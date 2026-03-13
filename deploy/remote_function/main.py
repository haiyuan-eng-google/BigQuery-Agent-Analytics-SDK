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

"""BigQuery Remote Function entry point.

Dispatches BigQuery Remote Function calls to SDK methods.
BigQuery sends batched requests as JSON with a ``calls`` array;
each element is ``[operation, params_json]``.  We return a ``replies``
array of the same length.

Supported operations::

    analyze   — retrieve and serialize a session trace
    evaluate  — run a code-based evaluator over traces
    judge     — run an LLM-as-judge evaluator over traces
    insights  — generate an insights report
    drift     — detect drift between golden and production datasets
"""

from __future__ import annotations

from dispatch import build_client_from_context
from dispatch import process_calls
from flask import jsonify
import functions_framework


@functions_framework.http
def handle_request(request):
  """HTTP entry point for BigQuery Remote Function."""
  body = request.get_json(silent=True)
  if not body or "calls" not in body:
    return jsonify({"errorMessage": "Missing 'calls' array"}), 400

  udc = body.get("userDefinedContext", {})
  try:
    client = build_client_from_context(udc)
  except ValueError as exc:
    return jsonify({"errorMessage": str(exc)}), 400

  replies = process_calls(client, body["calls"])
  return jsonify({"replies": replies})
