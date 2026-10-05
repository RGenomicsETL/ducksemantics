SELECT CASE WHEN semantic_colbert_encode(NULL, 'BRCA1', 'query') IS NULL
  AND semantic_colbert_encode('/no/model.gguf', NULL, 'query') IS NULL
  AND semantic_colbert_encode('/no/model.gguf', 'BRCA1', NULL) IS NULL
  THEN true ELSE error('ColBERT NULL propagation') END;
SELECT CASE WHEN typeof(semantic_colbert_encode(NULL, NULL, NULL)) = 'FLOAT[][]'
  THEN true ELSE error('ColBERT nested float return type') END;
SELECT CASE WHEN count(*) FILTER (WHERE semantic_colbert_encode(NULL, text, role) IS NULL) = 3
  THEN true ELSE error('ColBERT NULL lanes') END
FROM (VALUES ('query', 'a'), ('document', 'b'), (NULL, NULL)) t(role, text);
