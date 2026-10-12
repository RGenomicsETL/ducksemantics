-- Scalar FLOAT[] results and element ranges can cross rows and blocks.
SELECT CASE WHEN gguf_tensor_values('build/fixtures/valid.gguf', 'w.f32', 2, 5)
                 = [2, 3, 4, 5, 6]::FLOAT[] THEN 'ok' ELSE error('f32 range') END;
SELECT CASE WHEN gguf_tensor_values('build/fixtures/valid.gguf', 'w.q8_0', 0, 32)
                 = list_transform(range(32), x -> x::FLOAT) THEN 'ok' ELSE error('q8_0 values') END;
SELECT CASE WHEN gguf_tensor_values('build/fixtures/valid.gguf', 'w.f32', 8, 0)
                 = []::FLOAT[] THEN 'ok' ELSE error('empty end range') END;
SELECT CASE WHEN gguf_tensor_matvec('build/fixtures/valid.gguf', 'w.f32', [1, 2, 3, 4])
                 = [20, 60]::FLOAT[] THEN 'ok' ELSE error('f32 matvec') END;
SELECT CASE WHEN gguf_tensor_values('build/fixtures/unaligned.gguf', 'half', 2, 4)
                 = [3, 4, 5, -6]::FLOAT[] THEN 'ok' ELSE error('half odd row') END;
SELECT CASE WHEN gguf_tensor_matvec('build/fixtures/unaligned.gguf', 'half', [1, 2, 3])
                 = [6, -4]::FLOAT[] THEN 'ok' ELSE error('half matvec') END;
SELECT CASE WHEN gguf_tensor_matvec('build/fixtures/unaligned.gguf', 'f32', [1, 2])
                 = [5, 11]::FLOAT[] THEN 'ok' ELSE error('unaligned f32 matvec') END;

-- Vectorized scalar calls include NULL inputs and empty results.
SELECT CASE WHEN count(*) = 3000 AND bool_and(v = [i::FLOAT]) THEN 'ok'
       ELSE error('scalar values across vectors') END
FROM (SELECT i, gguf_tensor_values('build/fixtures/many.gguf', printf('t.%05d', i), 0, 1) v
      FROM range(3000) t(i));
SELECT CASE WHEN list(v ORDER BY i) = [[0, 1]::FLOAT[], NULL, []::FLOAT[], [6, 7]::FLOAT[]]
       THEN 'ok' ELSE error('NULL scalar list') END
FROM (SELECT i, gguf_tensor_values('build/fixtures/valid.gguf', 'w.f32', a, b) v
      FROM (VALUES (0, 0::UBIGINT, 2::UBIGINT), (1, NULL, 1), (2, 8, 0), (3, 6, 2)) t(i,a,b));
SELECT CASE WHEN gguf_tensor_matvec(NULL, 'w.f32', [1, 2, 3, 4]) IS NULL
       AND gguf_tensor_matvec('build/fixtures/valid.gguf', 'w.f32', NULL) IS NULL
       THEN 'ok' ELSE error('NULL matvec') END;

-- Zero activation visits every format's dot and scratch path.
SELECT CASE WHEN count(*) = 10 AND bool_and(y = [0, 0, 0, 0, 0, 0, 0]::FLOAT[])
       THEN 'ok' ELSE error('zero matvec formats') END
FROM (SELECT gguf_tensor_matvec('build/fixtures/quant.gguf', name,
                               list_transform(range(512), x -> 0::FLOAT)) y
      FROM gguf_tensors('build/fixtures/quant.gguf'));
SELECT CASE WHEN gguf_tensor_matvec('build/fixtures/rank3.gguf', 'w', [1, 2])
                 = [5, 11, 17, 23]::FLOAT[] THEN 'ok' ELSE error('flatten leading dimensions') END;
