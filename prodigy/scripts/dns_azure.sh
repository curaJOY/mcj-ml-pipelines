#!/usr/bin/env bash
set -euo pipefail
# Create Azure DNS records using values from App Runner
# Prereqs: az login; variables below set
: "${AZ_RG:?set AZ_RG}"            # Azure resource group with DNS zone
: "${AZ_ZONE:?set AZ_ZONE}"        # curajoy.org
: "${VALIDATION_NAME:?}"           # e.g. _xxxxxxxxxxxx.dev
: "${VALIDATION_TYPE:?}"           # CNAME or TXT
: "${VALIDATION_VALUE:?}"          # target value from App Runner
: "${APPRUNNER_CNAME:?}"           # e.g. randomhash.region.awsapprunner.com

TTL=300

if [[ "$VALIDATION_TYPE" == "CNAME" ]]; then
  az network dns record-set cname set-record \
    --resource-group "$AZ_RG" --zone-name "$AZ_ZONE" \
    --record-set-name "$VALIDATION_NAME" \
    --cname "$VALIDATION_VALUE" --ttl $TTL
else
  az network dns record-set txt add-record \
    --resource-group "$AZ_RG" --zone-name "$AZ_ZONE" \
    --record-set-name "$VALIDATION_NAME" \
    --value "$VALIDATION_VALUE" --ttl $TTL
fi

# Production CNAME: dev -> App Runner default domain
az network dns record-set cname set-record \
  --resource-group "$AZ_RG" --zone-name "$AZ_ZONE" \
  --record-set-name "dev" \
  --cname "$APPRUNNER_CNAME" --ttl $TTL
