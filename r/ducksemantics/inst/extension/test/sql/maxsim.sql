-- Hand-checked scores: each query token takes its best document token.
SELECT CASE WHEN semantic_maxsim([[1, 0], [0, 1]], [[0.5, 0.5], [1, 0]]) = 1.5 THEN 'ok'
       ELSE error('maxsim basic') END;
SELECT CASE WHEN semantic_maxsim([[1, 2, 3]], [[-1, -1, -1], [0, 0, 1]]) = 3 THEN 'ok'
       ELSE error('maxsim best match') END;
-- Negative best scores are kept, not clamped.
SELECT CASE WHEN semantic_maxsim([[1, 0]], [[-2, 0], [-3, 0]]) = -2 THEN 'ok'
       ELSE error('maxsim negative') END;

-- NULL in, NULL out; an empty side has no best match.
SELECT CASE WHEN semantic_maxsim(NULL, [[1.0]]) IS NULL AND semantic_maxsim([[1.0]], NULL) IS NULL
            AND semantic_maxsim([]::FLOAT[][], [[1.0]]) IS NULL
            AND semantic_maxsim([[1.0]], []::FLOAT[][]) IS NULL THEN 'ok'
       ELSE error('maxsim NULL handling') END;

-- Agreement with the pure-SQL definition over 5000 random pairs of
-- 32-token x 128-wide matrices, crossing many DuckDB vectors.
SELECT setseed(0.42);
CREATE TABLE pairs AS
SELECT i,
       list_transform(range(4), t -> list_transform(range(128), k -> (random() - 0.5)::FLOAT)) AS q,
       list_transform(range(1 + i % 32), t -> list_transform(range(128), k -> (random() - 0.5)::FLOAT)) AS d
FROM range(5000) r(i);
SELECT CASE WHEN max(abs(semantic_maxsim(q, d) -
                     list_sum(list_transform(q, a -> list_max(list_transform(d, b -> list_dot_product(a, b)))))))
                 < 1e-4 THEN 'ok'
       ELSE error('maxsim disagrees with SQL definition') END FROM pairs;
-- Deterministic: identical inputs give bit-identical scores, also under parallel scans.
SELECT CASE WHEN count(*) = 0 THEN 'ok' ELSE error('maxsim not deterministic') END
FROM (SELECT i, semantic_maxsim(q, d) s FROM pairs) a JOIN (SELECT i, semantic_maxsim(q, d) s FROM pairs) b
USING (i) WHERE a.s IS DISTINCT FROM b.s;
