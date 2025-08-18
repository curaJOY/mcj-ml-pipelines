#!/usr/bin/env bash
set -euo pipefail
: "${ROLE_NAME:=AppRunnerEcrSecretsRole}"

cat > trust.json <<'JSON'
{ "Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Principal":{"Service":"build.apprunner.amazonaws.com"},"Action":"sts:AssumeRole"},
  {"Effect":"Allow","Principal":{"Service":"tasks.apprunner.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON

aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file://trust.json >/dev/null 2>&1 || true
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AWSAppRunnerServicePolicyForECRAccess >/dev/null
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite >/dev/null
aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text
