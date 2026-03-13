#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Deploy the BigQuery Remote Function to Cloud Functions (gen2).
#
# Usage:
#   deploy.sh PROJECT [FUNCTION_REGION] [DATASET] [BQ_LOCATION]
#
# FUNCTION_REGION: where the Cloud Function runs (default: us-central1).
# BQ_LOCATION:     BigQuery dataset + connection location (default: US).
#                   Must match the dataset's location, NOT the function's
#                   region.  A multi-region dataset (US, EU) needs a
#                   multi-region connection in the same location.

set -euo pipefail

PROJECT="${1:?Usage: deploy.sh PROJECT [FUNCTION_REGION] [DATASET] [BQ_LOCATION]}"
FUNCTION_REGION="${2:-us-central1}"
DATASET="${3:-agent_analytics}"
BQ_LOCATION="${4:-US}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ------------------------------------------------------------------ #
# Stage deployment bundle                                              #
# ------------------------------------------------------------------ #
# Cloud Functions deploys only the --source directory.  We build a
# temporary staging area that contains:
#   1. main.py + dispatch.py (Cloud Function code)
#   2. requirements.txt (generated — not checked in, because it must
#      reference the local SDK wheel by filename)
#   3. The SDK wheel built from the repo working tree
# This ensures the deployed function always runs the checked-in SDK,
# not whatever version is on PyPI.
#
# NOTE: Do not deploy deploy/remote_function/ directly with gcloud.
#       Always use this script so the SDK wheel is bundled correctly.

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

echo "==> Building SDK wheel..."
python3 -m pip wheel --no-deps --wheel-dir "$STAGING" "$REPO_ROOT" -q

echo "==> Staging deployment bundle..."
cp "$SCRIPT_DIR/main.py" "$SCRIPT_DIR/dispatch.py" "$STAGING/"

# Write requirements.txt that installs the local wheel + runtime deps
WHEEL_FILE=$(basename "$STAGING"/bigquery_agent_analytics-*.whl)
cat > "$STAGING/requirements.txt" <<REQS
functions-framework==3.*
google-genai>=1.0.0
./${WHEEL_FILE}
REQS

# ------------------------------------------------------------------ #
# Deploy Cloud Function                                                #
# ------------------------------------------------------------------ #

echo "==> Deploying Cloud Function to $FUNCTION_REGION..."
gcloud functions deploy bq-agent-analytics \
  --gen2 --runtime python312 --region "$FUNCTION_REGION" \
  --entry-point handle_request \
  --source "$STAGING" \
  --trigger-http --no-allow-unauthenticated \
  --set-env-vars "BQ_AGENT_PROJECT=$PROJECT,BQ_AGENT_DATASET=$DATASET,BQ_AGENT_LOCATION=$BQ_LOCATION" \
  --memory 512MB --timeout 120s --min-instances 0

# ------------------------------------------------------------------ #
# BigQuery connection (must match DATASET location, not function region)
# ------------------------------------------------------------------ #

echo "==> Creating CLOUD_RESOURCE connection in $BQ_LOCATION..."
bq mk --connection --location="$BQ_LOCATION" --connection_type=CLOUD_RESOURCE \
  --project_id="$PROJECT" analytics-conn 2>/dev/null || true

echo "==> Granting invoker role to connection SA..."
CONNECTION_SA=$(bq show --connection --format=json \
  "$PROJECT.$BQ_LOCATION.analytics-conn" | jq -r '.cloudResource.serviceAccountId')
gcloud functions add-invoker-policy-binding bq-agent-analytics \
  --region="$FUNCTION_REGION" --member="serviceAccount:${CONNECTION_SA}"

# ------------------------------------------------------------------ #
# Print registration DDL                                               #
# ------------------------------------------------------------------ #

ENDPOINT="https://${FUNCTION_REGION}-${PROJECT}.cloudfunctions.net/bq-agent-analytics"

echo "==> Done. Register the function with:"
echo ""
echo "  CREATE OR REPLACE FUNCTION \`${PROJECT}.${DATASET}.agent_analytics\`("
echo "    operation STRING, params JSON"
echo "  ) RETURNS JSON"
echo "  REMOTE WITH CONNECTION \`${PROJECT}.${BQ_LOCATION}.analytics-conn\`"
echo "  OPTIONS ("
echo "    endpoint = '${ENDPOINT}',"
echo "    max_batching_rows = 50"
echo "  );"
