LOAD 'build/ducksemantics.duckdb_extension';
CREATE MACRO assert_true(v) AS CASE WHEN v THEN true ELSE error('generation stats contract failed') END;

SELECT assert_true(semantic_generate_stats(NULL::VARCHAR, 'Hello', 8) IS NULL);
SELECT assert_true(semantic_generate_stats('unused', NULL::VARCHAR, 8, 16) IS NULL);
SELECT assert_true(semantic_generate_stats('unused', 'Hello', NULL::INTEGER) IS NULL);
SELECT assert_true(semantic_generate_stats('unused', 'Hello', 8, NULL::INTEGER) IS NULL);
SELECT assert_true(typeof((semantic_generate_stats(NULL::VARCHAR, 'Hello', 8)).text) = 'VARCHAR');
SELECT assert_true(typeof((semantic_generate_stats(NULL::VARCHAR, 'Hello', 8)).prompt_tokens) = 'UBIGINT');
SELECT assert_true(typeof((semantic_generate_stats(NULL::VARCHAR, 'Hello', 8)).prefill_seconds) = 'DOUBLE');
SELECT assert_true(typeof((semantic_generate_stats(NULL::VARCHAR, 'Hello', 8)).decode_tokens_per_second) = 'DOUBLE');
SELECT assert_true((semantic_generate_stats(NULL::VARCHAR, 'Hello', 8)).text IS NULL);
SELECT assert_true((semantic_generate_stats(NULL::VARCHAR, 'Hello', 8)).decode_tokens IS NULL);

SELECT assert_true(count(*) = 4097)
FROM (
  SELECT semantic_generate_stats(NULL::VARCHAR, 'Hello', 8, 16) AS result
  FROM range(4097)
) runs
WHERE result IS NULL AND result.prefill_seconds IS NULL;

SELECT assert_true(count(*) = 2 AND bool_and(has_side_effects))
FROM duckdb_functions() WHERE function_name = 'semantic_generate_stats';
SELECT assert_true(bool_and(has_side_effects))
FROM duckdb_functions() WHERE function_name = 'semantic_generate';
