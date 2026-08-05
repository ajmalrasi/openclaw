#!/usr/bin/env bash
# Provision the AWS EC2 g6.xlarge OpenClaw backend as a boot-persistent system
# service. The service runs Gemma 4 12B QAT W4A16 on the NVIDIA L4 and exposes
# vLLM only on 127.0.0.1:11434; Caddy provides the authenticated HTTPS endpoint.
#
#   ./install-vllm-ec2.sh            # install and restart the service
#   ./install-vllm-ec2.sh --no-start # install/enable without touching a live run
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT="openclaw-vllm-ec2.service"
ACCOUNT_ID="740940193664"
ECR_REPOSITORY="openclaw/vllm-gemma4"
IMAGE_DIGEST="sha256:0ea4b07a909f78a5cc8a6a82e9d3dd3efa51b59a0f5421fcf2207e80a3aae53b"
CADDY_UNIT="openclaw-caddy-ec2.service"
CADDY_IMAGE="caddy:2.10.2"
INIT_UNIT="openclaw-ec2-init.service"
ENV_FILE="/etc/openclaw-vllm-ec2.env"
MODE="${1:-}"

if [[ -n "$MODE" && "$MODE" != "--no-start" ]]; then
  echo "Usage: $0 [--no-start]" >&2
  exit 2
fi

echo ">> Checking Docker and the NVIDIA container runtime ..."
command -v docker >/dev/null || { echo "!! docker not found" >&2; exit 1; }
command -v aws >/dev/null || { echo "!! aws CLI not found" >&2; exit 1; }
command -v curl >/dev/null || { echo "!! curl not found" >&2; exit 1; }
docker info >/dev/null

METADATA_TOKEN=$(curl -fsS -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  http://169.254.169.254/latest/api/token)
AWS_REGION=$(curl -fsS \
  -H "X-aws-ec2-metadata-token: $METADATA_TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/region)
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
IMAGE="$ECR_REGISTRY/$ECR_REPOSITORY@$IMAGE_DIGEST"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">> Authenticating with ECR and pulling $IMAGE (one-time) ..."
  trap 'docker logout "$ECR_REGISTRY" >/dev/null 2>&1 || true' EXIT
  aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_REGISTRY"
  docker pull "$IMAGE"
  docker logout "$ECR_REGISTRY" >/dev/null
  trap - EXIT
fi

docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE" \
  --query-gpu=name,memory.total --format=csv,noheader

if ! docker image inspect "$CADDY_IMAGE" >/dev/null 2>&1; then
  echo ">> Pulling $CADDY_IMAGE ..."
  docker pull "$CADDY_IMAGE"
fi

echo ">> Installing EC2 initialization and serving units ..."
sudo install -d -m 0755 /etc/openclaw /var/lib/openclaw-caddy/data \
  /var/lib/openclaw-caddy/config /usr/local/lib/openclaw
sudo install -m 0644 "$HERE/vllm/Caddyfile.ec2" /etc/openclaw/Caddyfile
sudo install -m 0755 "$HERE/vllm/openclaw-ec2-init.sh" \
  /usr/local/lib/openclaw/openclaw-ec2-init.sh
sudo install -m 0644 "$HERE/vllm/$INIT_UNIT" "/etc/systemd/system/$INIT_UNIT"
sudo install -m 0644 "$HERE/vllm/$UNIT" "/etc/systemd/system/$UNIT"
sudo install -m 0644 "$HERE/vllm/$CADDY_UNIT" "/etc/systemd/system/$CADDY_UNIT"
sudo systemctl daemon-reload
sudo systemctl enable "$INIT_UNIT" "$UNIT" "$CADDY_UNIT"

if [[ "$MODE" == "--no-start" ]]; then
  sudo systemctl start "$INIT_UNIT"
  echo ">> Installed and enabled without changing the running container."
  echo ">> Start later with: sudo systemctl start $UNIT $CADDY_UNIT"
  exit 0
fi

echo ">> Starting $UNIT and $CADDY_UNIT ..."
sudo systemctl restart "$INIT_UNIT"
sudo systemctl restart "$UNIT"
sudo systemctl restart "$CADDY_UNIT"

echo ">> Waiting for vLLM on 127.0.0.1:11434 ..."
for _ in $(seq 1 300); do
  curl -sf http://127.0.0.1:11434/health >/dev/null 2>&1 && break
  sleep 2
done

if curl -sf http://127.0.0.1:11434/health >/dev/null 2>&1; then
  PUBLIC_HOST=$(sudo sed -n 's/^OPENCLAW_PUBLIC_HOST=//p' "$ENV_FILE")
  echo ">> OK. Model=openclaw, base URL=http://127.0.0.1:11434/v1"
  echo ">> Public endpoint: https://$PUBLIC_HOST/v1/chat/completions"
  echo ">> Retrieve the bearer key with:"
  echo ">>   sudo sed -n 's/^VLLM_API_KEY=//p' $ENV_FILE"
else
  echo "!! vLLM did not become healthy. Check:" >&2
  echo "     sudo journalctl -u $UNIT -e" >&2
  exit 1
fi
