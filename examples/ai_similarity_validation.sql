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
-- Replace {project}, {dataset}, {endpoint}, and {table} with your values.
-- {endpoint} is the embedding model endpoint, e.g. 'text-embedding-005'.

-- 1. Current approach — AI.EMBED both sides, ML.DISTANCE on vectors
WITH ground_truth_embedded AS (
  SELECT
    session_id,
    question,
    AI.EMBED(
      question,
      endpoint => '{endpoint}'
    ) AS embedding
  FROM `{project}.{dataset}.{table}_ground_truth`
),
predicted_embedded AS (
  SELECT
    session_id,
    question,
    AI.EMBED(
      question,
      endpoint => '{endpoint}'
    ) AS embedding
  FROM `{project}.{dataset}.{table}_predicted`
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
SELECT
  g.session_id AS gt_session_id,
  p.session_id AS pred_session_id,
  AI.SIMILARITY(
    content1 => g.question,
    content2 => p.question,
    endpoint => '{endpoint}'
  ) AS similarity
FROM
  `{project}.{dataset}.{table}_ground_truth` g
CROSS JOIN
  `{project}.{dataset}.{table}_predicted` p
ORDER BY gt_session_id, pred_session_id;

-- 3. Agreement check — join both results, compute correlation & max diff
WITH ground_truth_embedded AS (
  SELECT
    session_id,
    question,
    AI.EMBED(
      question,
      endpoint => '{endpoint}'
    ) AS embedding
  FROM `{project}.{dataset}.{table}_ground_truth`
),
predicted_embedded AS (
  SELECT
    session_id,
    question,
    AI.EMBED(
      question,
      endpoint => '{endpoint}'
    ) AS embedding
  FROM `{project}.{dataset}.{table}_predicted`
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
    AI.SIMILARITY(
      content1 => g.question,
      content2 => p.question,
      endpoint => '{endpoint}'
    ) AS similarity_from_ai
  FROM
    `{project}.{dataset}.{table}_ground_truth` g
  CROSS JOIN
    `{project}.{dataset}.{table}_predicted` p
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
