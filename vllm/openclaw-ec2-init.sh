#!/usr/bin/env bash
# Refresh machine-specific endpoint state on every EC2 boot. The bearer key is
# stable for a given VM, while the sslip.io hostname follows a changed public IP.
set -euo pipefail

ENV_FILE=/etc/openclaw-vllm-ec2.env
ENV_TMP=$(mktemp)
trap 'rm -f "$ENV_TMP"' EXIT

API_KEY=""
if [[ -s "$ENV_FILE" ]]; then
  API_KEY=$(sed -n 's/^VLLM_API_KEY=//p' "$ENV_FILE")
fi
if [[ -z "$API_KEY" ]]; then
  API_KEY=$(openssl rand -hex 32)
fi

PUBLIC_IP=""
AWS_REGION=""
for _ in $(seq 1 60); do
  TOKEN=$(curl -fsS --connect-timeout 2 -X PUT \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
    http://169.254.169.254/latest/api/token || true)
  if [[ -n "$TOKEN" ]]; then
    PUBLIC_IP=$(curl -fsS --connect-timeout 2 \
      -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/public-ipv4 || true)
    AWS_REGION=$(curl -fsS --connect-timeout 2 \
      -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/placement/region || true)
  fi
  [[ -n "$PUBLIC_IP" && -n "$AWS_REGION" ]] && break
  sleep 2
done

if [[ -z "$PUBLIC_IP" || -z "$AWS_REGION" ]]; then
  echo "EC2 public IPv4 address or region is unavailable" >&2
  exit 1
fi

PUBLIC_HOST="${PUBLIC_IP//./-}.sslip.io"
ECR_IMAGE="740940193664.dkr.ecr.${AWS_REGION}.amazonaws.com/openclaw/vllm-gemma4@sha256:0ea4b07a909f78a5cc8a6a82e9d3dd3efa51b59a0f5421fcf2207e80a3aae53b"
printf 'VLLM_API_KEY=%s\nOPENCLAW_PUBLIC_HOST=%s\nOPENCLAW_ECR_IMAGE=%s\n' \
  "$API_KEY" "$PUBLIC_HOST" "$ECR_IMAGE" >"$ENV_TMP"
install -o root -g root -m 0600 "$ENV_TMP" "$ENV_FILE"
