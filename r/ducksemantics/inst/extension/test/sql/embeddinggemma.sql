SELECT CASE WHEN typeof(semantic_embed(NULL, NULL, NULL, NULL, NULL)) = 'FLOAT[]'
    THEN true ELSE error('semantic_embed return type') END;
SELECT CASE WHEN semantic_embed(NULL, 'x', 'raw', 768, true) IS NULL
    AND semantic_embed('missing', NULL, 'invalid', 0, true) IS NULL
    AND semantic_embed('missing', 'x', NULL, 768, true) IS NULL
    AND semantic_embed('missing', 'x', 'raw', NULL, true) IS NULL
    AND semantic_embed('missing', 'x', 'raw', 768, NULL) IS NULL
    THEN true ELSE error('semantic_embed NULL propagation') END;
SELECT CASE WHEN count(*) = 2500 AND count(v) = 0 THEN true
    ELSE error('semantic_embed NULL chunks') END
FROM (SELECT semantic_embed('missing', CASE WHEN i % 2 = 0 THEN NULL ELSE 'x' END,
    CASE WHEN i % 2 = 0 THEN 'raw' ELSE NULL END, 768, true) AS v FROM range(2500) t(i));
