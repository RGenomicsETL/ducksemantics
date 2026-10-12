-- Header geometry of the synthetic fixture.
SELECT CASE WHEN version = 3 AND tensor_count = 3 AND metadata_count = 18 AND alignment = 64
            AND data_offset % 64 = 0 AND tensor_bytes = 32 + 34 THEN 'ok'
       ELSE error('gguf_header: ' || version || ' ' || tensor_count || ' ' || metadata_count) END
FROM gguf_header('build/fixtures/valid.gguf');

-- Every scalar type renders exactly; arrays report type and length, not a value.
CREATE TABLE md AS SELECT * FROM gguf_metadata('build/fixtures/valid.gguf');
CREATE TABLE expected (key VARCHAR, value_type VARCHAR, value VARCHAR);
INSERT INTO expected VALUES
  ('general.architecture', 'string', 'fixture'), ('general.alignment', 'uint32', '64'),
  ('general.name', 'string', 'Ünïcode ✓ model'),
  ('t.u8', 'uint8', '255'), ('t.i8', 'int8', '-128'), ('t.u16', 'uint16', '65535'),
  ('t.i16', 'int16', '-32768'), ('t.u32', 'uint32', '4294967295'),
  ('t.i32', 'int32', '-2147483648'), ('t.f32', 'float32', '0.100000001'),
  ('t.bool', 'bool', 'true'), ('t.u64', 'uint64', '18446744073709551615'),
  ('t.i64', 'int64', '-9223372036854775808'), ('t.f64', 'float64', '2.5e-300');
SELECT CASE WHEN count(*) = 14 THEN 'ok' ELSE error('scalar metadata mismatch') END
FROM expected e JOIN md m USING (key, value_type, value) WHERE m.array_type IS NULL;
SELECT CASE WHEN list(key || ':' || array_type || ':' || array_length ORDER BY key)
                 = ['t.empty:int32:0', 't.nested:array:2', 't.scores:float32:3', 't.tokens:string:4']
            AND bool_and(value IS NULL) THEN 'ok' ELSE error('array metadata mismatch') END
FROM md WHERE value_type = 'array';
-- Metadata keeps file order.
SELECT CASE WHEN first(key) = 'general.architecture' AND last(key) = 't.nested' THEN 'ok'
       ELSE error('metadata order') END FROM md;

-- Array elements, including empty and multi-byte strings and nested arrays.
SELECT CASE WHEN list(value ORDER BY array_index) = ['<s>', '▁hello', '', '✓'] THEN 'ok'
       ELSE error('string array') END
FROM gguf_metadata_array('build/fixtures/valid.gguf', 't.tokens');
SELECT CASE WHEN list(value ORDER BY array_index) = ['0', '-1.5', '3.25'] THEN 'ok'
       ELSE error('float array') END
FROM gguf_metadata_array('build/fixtures/valid.gguf', 't.scores');
SELECT CASE WHEN count(*) = 0 THEN 'ok' ELSE error('empty array') END
FROM gguf_metadata_array('build/fixtures/valid.gguf', 't.empty');
SELECT CASE WHEN count(*) = 2 AND bool_and(value IS NULL) THEN 'ok' ELSE error('nested array') END
FROM gguf_metadata_array('build/fixtures/valid.gguf', 't.nested');

-- Tensors: known types carry byte sizes, an unknown type id is listed with NULLs.
CREATE TABLE t AS SELECT * FROM gguf_tensors('build/fixtures/valid.gguf');
SELECT CASE WHEN list(name ORDER BY tensor_index) = ['w.f32', 'w.q8_0', 'w.future']
            AND list(ggml_type ORDER BY tensor_index) = ['f32', 'q8_0', NULL]
            AND list(ggml_type_id ORDER BY tensor_index) = [0, 8, 99]
            AND list(dims ORDER BY tensor_index) = [[4, 2], [32], [7]]
            AND list(n_elements ORDER BY tensor_index) = [8, 32, 7]
            AND list(n_bytes ORDER BY tensor_index) = [32, 34, NULL]
            AND bool_and(file_offset % 64 = 0) THEN 'ok' ELSE error('tensor listing') END FROM t;
SELECT CASE WHEN min(t.file_offset) = h.data_offset THEN 'ok' ELSE error('tensor offsets') END
FROM t, gguf_header('build/fixtures/valid.gguf') h GROUP BY h.data_offset;

-- An empty but well-formed file.
SELECT CASE WHEN tensor_count = 0 AND metadata_count = 0 AND tensor_bytes = 0 THEN 'ok'
       ELSE error('empty file') END FROM gguf_header('build/fixtures/empty.gguf');
SELECT CASE WHEN count(*) = 0 THEN 'ok' ELSE error('empty listing') END
FROM gguf_tensors('build/fixtures/empty.gguf');

-- Scans resume correctly across DuckDB vectors (3000 rows > 2048).
SELECT CASE WHEN count(*) = 3001 AND count(DISTINCT key) = 3001
            AND sum(TRY_CAST(value AS BIGINT)) FILTER (WHERE key LIKE 'k.%') = 4498500 THEN 'ok'
       ELSE error('metadata across vectors') END FROM gguf_metadata('build/fixtures/many.gguf');
SELECT CASE WHEN count(*) = 3000 AND sum(len(dims)) = 6000 AND sum(n_bytes) = 12000
            AND bool_and(file_offset = (SELECT data_offset FROM gguf_header('build/fixtures/many.gguf')) + 4 * tensor_index)
            AND bool_and(name = printf('t.%05d', tensor_index))
            AND bool_and(len(dims) = 1 + tensor_index % 3) THEN 'ok'
       ELSE error('tensors across vectors') END FROM gguf_tensors('build/fixtures/many.gguf');
