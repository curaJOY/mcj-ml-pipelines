#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION}"
: "${AWS_PROFILE:?set AWS_PROFILE}"
: "${ECR_REPO:=prodigy-service}"
: "${IMAGE_TAG:=v1}"

# Resolve account id from the logged-in profile
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

# Optional: choose architecture explicitly if your wheel is ARM
# docker build --platform=linux/arm64 -t "$ECR_REPO:$IMAGE_TAG" .
docker build -t "$ECR_REPO:$IMAGE_TAG" .

aws ecr describe-repositories --repository-names "$ECR_REPO" >/dev/null 2>&1 || \
aws ecr create-repository --repository-name "$ECR_REPO" >/dev/null

aws ecr get-login-password \
  | docker login --username AWS --password-stdin \
    "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker tag  "$ECR_REPO:$IMAGE_TAG" \
  "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"

docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"

echo "Pushed: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"
