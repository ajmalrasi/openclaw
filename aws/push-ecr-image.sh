#!/usr/bin/env bash
# Build the pinned amd64 vLLM+Gemma image and push it to the private ECR repo.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_PROFILE="${AWS_PROFILE:-ml-prep-deploy}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-740940193664}"
REPOSITORY="openclaw/vllm-gemma4"
IMAGE_TAG="${IMAGE_TAG:-vllm-0.25.1-gemma4-12b-w4a16-$(date -u +%Y%m%d%H%M%S)}"
REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
IMAGE="$REGISTRY/$REPOSITORY:$IMAGE_TAG"

aws ecr describe-repositories --profile "$AWS_PROFILE" --region "$AWS_REGION" \
  --repository-names "$REPOSITORY" >/dev/null 2>&1 || \
  aws ecr create-repository --profile "$AWS_PROFILE" --region "$AWS_REGION" \
    --repository-name "$REPOSITORY" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 >/dev/null
aws ecr put-image-tag-mutability --profile "$AWS_PROFILE" --region "$AWS_REGION" \
  --repository-name "$REPOSITORY" --image-tag-mutability IMMUTABLE >/dev/null

aws ecr get-login-password --profile "$AWS_PROFILE" --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$REGISTRY"
trap 'docker logout "$REGISTRY" >/dev/null 2>&1 || true' EXIT

docker build --platform linux/amd64 --progress=plain \
  -f "$HERE/Dockerfile.ec2" -t "$IMAGE" "$HERE"
docker push "$IMAGE"
echo "Pushed $IMAGE"
