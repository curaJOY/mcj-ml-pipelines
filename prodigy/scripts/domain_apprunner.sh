#!/usr/bin/env bash
set -euo pipefail
: "${SERVICE_ARN:?set SERVICE_ARN}"
: "${CUSTOM_DOMAIN:?set CUSTOM_DOMAIN}"  # dev.curajoy.org

# Associate custom domain and print validation records
aws apprunner associate-custom-domain \
  --service-arn "$SERVICE_ARN" \
  --domain-name "$CUSTOM_DOMAIN" \
  --query 'DNSTarget,CertificateValidationRecords' --output json
