-- Copyright 2026 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Replace placeholders before running:
--   PROJECT         — GCP project ID
--   DATASET         — BigQuery dataset name
--   BQ_LOCATION     — BigQuery dataset location (e.g. US, EU, us-central1)
--   FUNCTION_REGION — Cloud Function region (e.g. us-central1)
--
-- The connection must already exist (deploy.sh creates it).
-- BQ_LOCATION must match the dataset's location, which may differ from
-- the Cloud Function's region.

CREATE OR REPLACE FUNCTION `PROJECT.DATASET.agent_analytics`(
  operation STRING, params JSON
) RETURNS JSON
REMOTE WITH CONNECTION `PROJECT.BQ_LOCATION.analytics-conn`
OPTIONS (
  endpoint = 'https://FUNCTION_REGION-PROJECT.cloudfunctions.net/bq-agent-analytics',
  max_batching_rows = 50
);
