#!/usr/bin/env bash
# Provision the AWS EC2 g6.xlarge OpenClaw backend as a boot-persistent system
# service. The service runs Qwen3.5-9B on the NVIDIA L4 and exposes vLLM only on
# 127.0.0.1:11434; clients connect through an SSH tunnel.
#
#   ./install-vllm-ec2.sh            # install and restart the service
#   ./install-vllm-ec2.sh --no-start # install/enable without touching a live run
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT="openclaw-vllm-ec2.service"
IMAGE="vllm/vllm-openai:v0.18.1"
CADDY_UNIT="openclaw-caddy-ec2.service"
CADDY_IMAGE="caddy:2.10.2"
ENV_FILE="/etc/openclaw-vllm-ec2.env"
NO_START="${1:-}"

if [[ -n "$NO_START" && "$NO_START" != "--no-start" ]]; then
  echo "Usage: $0 [--no-start]" >&2
  exit 2
fi

echo ">> Checking Docker and the NVIDIA container runtime ..."
command -v docker >/dev/null || { echo "!! docker not found" >&2; exit 1; }
docker info >/dev/null

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> Pulling $IMAGE (large, one-time) ..."
  docker pull "$IMAGE"
fi

docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE" \
  --query-gpu=name,memory.total --format=csv,noheader

if ! docker image inspect "$CADDY_IMAGE" >/dev/null 2>&1; then
  echo ">> Pulling $CADDY_IMAGE ..."
  docker pull "$CADDY_IMAGE"
fi

echo ">> Preparing the public hostname and root-only API key ..."
METADATA_TOKEN=$(curl -fsS -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  http://169.254.169.254/latest/api/token)
PUBLIC_IP=$(curl -fsS \
  -H "X-aws-ec2-metadata-token: $METADATA_TOKEN" \
  http://169.254.169.254/latest/meta-data/public-ipv4)
PUBLIC_HOST_DEFAULT="${PUBLIC_IP//./-}.sslip.io"
PUBLIC_HOST="${OPENCLAW_PUBLIC_HOST:-$PUBLIC_HOST_DEFAULT}"

API_KEY=""
if sudo test -s "$ENV_FILE"; then
  API_KEY=$(sudo sed -n 's/^VLLM_API_KEY=//p' "$ENV_FILE")
fi
if [[ -z "$API_KEY" ]]; then
  API_KEY=$(openssl rand -hex 32)
fi

ENV_TMP=$(mktemp)
trap 'rm -f "$ENV_TMP"' EXIT
printf 'VLLM_API_KEY=%s\nOPENCLAW_PUBLIC_HOST=%s\n' \
  "$API_KEY" "$PUBLIC_HOST" >"$ENV_TMP"
sudo install -m 0600 "$ENV_TMP" "$ENV_FILE"

echo ">> Installing $UNIT and $CADDY_UNIT ..."
sudo install -d -m 0755 /etc/openclaw /var/lib/openclaw-caddy/data \
  /var/lib/openclaw-caddy/config
sudo install -m 0644 "$HERE/vllm/Caddyfile.ec2" /etc/openclaw/Caddyfile
sudo install -m 0644 "$HERE/vllm/$UNIT" "/etc/systemd/system/$UNIT"
sudo install -m 0644 "$HERE/vllm/$CADDY_UNIT" "/etc/systemd/system/$CADDY_UNIT"
sudo systemctl daemon-reload
sudo systemctl enable "$UNIT" "$CADDY_UNIT"

if [[ "$NO_START" == "--no-start" ]]; then
  echo ">> Installed and enabled without changing the running container."
  echo ">> Start later with: sudo systemctl start $UNIT $CADDY_UNIT"
  exit 0
fi

echo ">> Starting $UNIT and $CADDY_UNIT ..."
sudo systemctl restart "$UNIT"
sudo systemctl restart "$CADDY_UNIT"

echo ">> Waiting for vLLM on 127.0.0.1:11434 ..."
for _ in $(seq 1 300); do
  curl -sf http://127.0.0.1:11434/health >/dev/null 2>&1 && break
  sleep 2
done

if curl -sf http://127.0.0.1:11434/health >/dev/null 2>&1; then
  echo ">> OK. Model=openclaw, base URL=http://127.0.0.1:11434/v1"
  echo ">> Public endpoint: https://$PUBLIC_HOST/v1/chat/completions"
  echo ">> Retrieve the bearer key with:"
  echo ">>   sudo sed -n 's/^VLLM_API_KEY=//p' $ENV_FILE"
else
  echo "!! vLLM did not become healthy. Check:" >&2
  echo "     sudo journalctl -u $UNIT -e" >&2
  exit 1
fi
