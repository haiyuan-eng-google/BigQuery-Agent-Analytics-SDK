-- ai_similarity_validation.sql
--
-- Side-by-side validation: AI.EMBED + ML.DISTANCE vs AI.SIMILARITY
--
-- AI.SIMILARITY provides a convenient text-to-text similarity score, but
-- re-embeds both inputs on every call.  In a cross join of N ground-truth
-- rows × M predicted rows this means O(N×M) embedding calls, compared to
-- O(N+M) when using AI.EMBED once per text and ML.DISTANCE on the
-- pre-computed vectors.
--
-- Use AI.SIMILARITY for:
--   • One-off pair comparisons or ad-hoc investigation
--   • Very small datasets where embedding cost is negligible
--
-- Use AI.EMBED + ML.DISTANCE for:
--   • Production drift workloads (the SDK's default)
--   • Any cross-join or large-batch scenario
--
-- Note: results are directionally similar but not guaranteed to be
-- numerically identical across implementations or model revisions.
--
-- Replace {project}, {dataset}, {model_id}, and {table} with your values.

-- 1. Current approach — AI.EMBED both sides, ML.DISTANCE on vectors
WITH ground_truth_embedded AS (
  SELECT
    session_id,
    question,
    ml_generate_embedding_result AS embedding
  FROM AI.EMBED(
    MODEL `{project}.{dataset}.{model_id}`,
    (SELECT session_id, question FROM `{project}.{dataset}.{table}_ground_truth`),
    STRUCT(TRUE AS flatten_json_output)
  )
),
predicted_embedded AS (
  SELECT
    session_id,
    question,
    ml_generate_embedding_result AS embedding
  FROM AI.EMBED(
    MODEL `{project}.{dataset}.{model_id}`,
    (SELECT session_id, question FROM `{project}.{dataset}.{table}_predicted`),
    STRUCT(TRUE AS flatten_json_output)
  )
),
embed_distance AS (
  SELECT
    g.session_id AS gt_session_id,
    p.session_id AS pred_session_id,
    ML.DISTANCE(g.embedding, p.embedding, 'COSINE') AS distance
  FROM ground_truth_embedded g
  CROSS JOIN predicted_embedded p
)
SELECT
  gt_session_id,
  pred_session_id,
  distance,
  1 - distance AS similarity_equiv
FROM embed_distance
ORDER BY gt_session_id, pred_session_id;

-- 2. AI.SIMILARITY approach — direct text-to-text comparison
WITH ai_sim AS (
  SELECT
    g.session_id AS gt_session_id,
    p.session_id AS pred_session_id,
    similarity
  FROM
    `{project}.{dataset}.{table}_ground_truth` g
  CROSS JOIN
    `{project}.{dataset}.{table}_predicted` p
  CROSS JOIN
    AI.SIMILARITY(
      MODEL `{project}.{dataset}.{model_id}`,
      g.question,
      p.question
    )
)
SELECT
  gt_session_id,
  pred_session_id,
  similarity
FROM ai_sim
ORDER BY gt_session_id, pred_session_id;

-- 3. Agreement check — join both results, compute correlation & max diff
WITH ground_truth_embedded AS (
  SELECT
    session_id,
    question,
    ml_generate_embedding_result AS embedding
  FROM AI.EMBED(
    MODEL `{project}.{dataset}.{model_id}`,
    (SELECT session_id, question FROM `{project}.{dataset}.{table}_ground_truth`),
    STRUCT(TRUE AS flatten_json_output)
  )
),
predicted_embedded AS (
  SELECT
    session_id,
    question,
    ml_generate_embedding_result AS embedding
  FROM AI.EMBED(
    MODEL `{project}.{dataset}.{model_id}`,
    (SELECT session_id, question FROM `{project}.{dataset}.{table}_predicted`),
    STRUCT(TRUE AS flatten_json_output)
  )
),
embed_results AS (
  SELECT
    g.session_id AS gt_session_id,
    p.session_id AS pred_session_id,
    1 - ML.DISTANCE(g.embedding, p.embedding, 'COSINE') AS similarity_from_embed
  FROM ground_truth_embedded g
  CROSS JOIN predicted_embedded p
),
ai_sim_results AS (
  SELECT
    g.session_id AS gt_session_id,
    p.session_id AS pred_session_id,
    similarity AS similarity_from_ai
  FROM
    `{project}.{dataset}.{table}_ground_truth` g
  CROSS JOIN
    `{project}.{dataset}.{table}_predicted` p
  CROSS JOIN
    AI.SIMILARITY(
      MODEL `{project}.{dataset}.{model_id}`,
      g.question,
      p.question
    )
)
SELECT
  CORR(e.similarity_from_embed, a.similarity_from_ai) AS pearson_correlation,
  MAX(ABS(e.similarity_from_embed - a.similarity_from_ai)) AS max_abs_difference,
  AVG(ABS(e.similarity_from_embed - a.similarity_from_ai)) AS avg_abs_difference,
  COUNT(*) AS num_pairs
FROM embed_results e
JOIN ai_sim_results a
  ON e.gt_session_id = a.gt_session_id
  AND e.pred_session_id = a.pred_session_id;
