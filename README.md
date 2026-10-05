# DuckSemantics

A CPU-only DuckDB extension in C for semantic retrieval and typed model
decisions over local GGUF models. It targets DuckDB's stable C extension API
(v1.2.0), so one artifact loads in every DuckDB release that supports that ABI
(checked: CLI 1.5.1, Python 1.5.2, R 1.5.5).

This is the native layer planned for `ducksemantics`: relational work stays in
SQL, model kernels run in the extension, and R keeps schema ownership, writes,
and provider protocols.

## Functions

| Function | Returns |
|---|---|
| `gguf_header(path)` | version, tensor and metadata counts, alignment, data offset, file and tensor bytes |
| `gguf_metadata(path)` | one row per key: `key`, `value_type`, `array_type`, `array_length`, scalar `value` as text |
| `gguf_metadata_array(path, key)` | one row per array element: `array_index`, `value` (e.g. a tokenizer vocabulary) |
| `gguf_tensors(path)` | `tensor_index`, `name`, `ggml_type`, `ggml_type_id`, `dims`, `n_elements`, `file_offset`, `n_bytes` |
| `semantic_maxsim(query FLOAT[][], document FLOAT[][])` | ColBERT late-interaction score: the sum over query tokens of the best dot product with any document token |

```sql
LOAD 'build/ducksemantics.duckdb_extension';
SELECT ggml_type, count(*), sum(n_bytes) FROM gguf_tensors('model.gguf') GROUP BY ALL;
SELECT value FROM gguf_metadata_array('model.gguf', 'tokenizer.ggml.tokens') LIMIT 5;
SELECT semantic_maxsim([[1, 0], [0, 1]], [[0.5, 0.5], [1, 0]]);  -- 1.5
```

The GGUF reader memory-maps the file and validates every length, count,
offset and alignment before use. It never decodes payloads eagerly. An unknown
ggml type id is listed with a NULL type name and byte size instead of failing.
`semantic_maxsim` uses raw dot products: pass L2-normalized vectors for cosine
MaxSim. Its reduction order is fixed and the build disables FMA contraction,
so scores are bit-identical across runs and CPUs. Empty or NULL matrices give
NULL; mismatched widths or NULL tokens raise errors.

## Build and test

```sh
make          # build/ducksemantics.duckdb_extension (cc + python3, no CMake)
make test     # SQL assertions and expected-error cases on the DuckDB CLI
```

Loading needs `allow_unsigned_extensions` (`duckdb -unsigned`).

Further verification scripts:

- `scripts/audit_gguf.py EXT FILE...` compares every header field, metadata
  value, array element and tensor against gguf-py's `GGUFReader`. It matched
  exactly on six real models: tiny Llama (f32 and bf16), ESM-2, LFM2.5-ColBERT
  350M (Q4_K_M), EmbeddingGemma 300M (Q8_0) and LFM2.5-8B-A1B (Q4_K_M). That is
  2.07M array elements and 853 tensors. Ternary-Bonsai-27B, which gguf-py
  cannot open, lists all 851 tensors (498 `q2_0`) tiling the file exactly.
- `scripts/fuzz_gguf.py` mutates fixtures and queries every view in a fresh
  process. 3,000 mutations ran under ASan/UBSan with no crash or report.

## Direction

Next phases, each validated before the next starts:

1. **Forward pass on CPU.** One decoder family at a time in plain C, as in
   antirez/ds4: GGUF weights in place, ggml-compatible dequantization and dot
   products for f32/f16/bf16/q8_0/q4_K/q6_K, logits checked against a
   reference implementation, starting with the one-layer tiny Llama.
2. **Typed decisions.** A Jev-compatible `semantic_system_one(state,
   questions)` returns per-option probabilities for yes/no, choice and score
   questions. It scores only the allowed option tokens, with no text
   generation or parsing, and returns a receipt of model digest, prompt
   template digest, option token ids and kernel build.
3. **KV reuse inside DuckDB chunks.** The cache for the shared instruction
   prefix is computed once. Within a chunk, rows that share a document reuse
   its KV state and branch per question. A per-thread cache carries a run
   across chunk boundaries, keyed by a hash of the rendered byte prefix as in
   ds4's KV store. Several questions on one document are evaluated inside one
   call in cost order, so an AI filter short-circuits on cached state.
4. **Encoders and embeddings.** A Laya-style encoder with a decision head,
   plus EmbeddingGemma and ColBERT token encoders behind the same receipts.

Model weights are never bundled; the caller supplies a path and the receipt
records its digest.
