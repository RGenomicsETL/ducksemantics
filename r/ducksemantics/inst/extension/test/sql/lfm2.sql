SELECT CASE WHEN semantic_colbert_encode(NULL, 'BRCA1', 'query') IS NULL
  AND semantic_colbert_encode('/no/model.gguf', NULL, 'query') IS NULL
  AND semantic_colbert_encode('/no/model.gguf', 'BRCA1', NULL) IS NULL
  THEN true ELSE error('ColBERT NULL propagation') END;
SELECT CASE WHEN typeof(semantic_colbert_encode(NULL, NULL, NULL)) = 'FLOAT[][]'
  THEN true ELSE error('ColBERT nested float return type') END;
SELECT CASE WHEN count(*) FILTER (WHERE semantic_colbert_encode(NULL, text, role) IS NULL) = 3
  THEN true ELSE error('ColBERT NULL lanes') END
FROM (VALUES ('query', 'a'), ('document', 'b'), (NULL, NULL)) t(role, text);
SELECT CASE WHEN semantic_generate(NULL, 'Hello', 32) IS NULL
  AND semantic_generate('/no/model.gguf', NULL, 32) IS NULL
  AND semantic_generate('/no/model.gguf', 'Hello', NULL) IS NULL
  THEN true ELSE error('generation NULL propagation') END;
SELECT CASE WHEN semantic_next_token_logits(NULL, 'Hello', [0]) IS NULL
  AND semantic_next_token_logits('/no/model.gguf', NULL, [0]) IS NULL
  AND semantic_next_token_logits('/no/model.gguf', 'Hello', NULL) IS NULL
  THEN true ELSE error('logits NULL propagation') END;
SELECT CASE WHEN typeof(semantic_generate(NULL, NULL, NULL)) = 'VARCHAR'
  AND typeof(semantic_next_token_logits(NULL, NULL, NULL)) = 'FLOAT[]'
  THEN true ELSE error('generation and logits return types') END;
