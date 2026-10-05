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
| `gguf_tensor_values(path, tensor, first_element, count)` | scalar `FLOAT[]` slice |
| `gguf_tensor_matvec(path, tensor, x FLOAT[])` | scalar `FLOAT[]` matrix-vector product |
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

## CPU quant kernels

`src/quant.h` exposes allocation-free scalar C kernels for F32, F16, BF16,
Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Q8_K. `quant_view_init()` creates a
borrowed view of a validated `gguf_tensor`; `quant_view_row()` checks row
bounds. The owning `gguf_file` must remain open. Payload access is
little-endian and alignment-independent.

`quant_dequantize_row()` decodes complete rows; `quant_values()` handles
arbitrary element slices. `quant_vec_dot()` uses Q8_0 activations for Q8_0
weights, Q8_K for K weights, and native f32 otherwise. K dots reduce integer
sub-dots before applying scales, with one f32 contribution per block and a
left-to-right block sum. Floating dots sum products left-to-right in f32.
Compile with `-ffp-contract=off`, without fast-math, and use IEEE
round-to-nearest. There are no kernel threads or SIMD-specific paths.

`quant_matvec()` accepts a row range, output capacity, and caller-owned
activation scratch sized by `quant_matvec_scratch()`. It quantizes x once
per call. Its output starts at `y[0]` for the requested first row. The kernels
allocate nothing; the SQL adapter owns mapping and scratch lifetimes.
GGML/FP16 notices are in `src/quant.LICENSE`.

### SQL validation surface

Both functions are scalars returning `FLOAT[]`, registered together by
`ducksemantics_register_quant()`:

```sql
SELECT gguf_tensor_values('model.gguf', 'token_embd.weight',
                          0::UBIGINT, 16::UBIGINT);
SELECT gguf_tensor_matvec('model.gguf', 'blk.0.attn_q.weight',
                          [/* dim[0] f32 activations */]::FLOAT[]);
```

Element indices are zero-based in GGUF storage order (`dim[0]` varies
fastest). Slices may cross rows and quant blocks. An empty slice at the end
of a tensor is valid. Matvec requires rank >= 2 and `length(x) = dim[0]`;
leading dimensions are flattened into rows, including MoE expert tensors.
NULL scalar arguments return NULL. Missing tensors, unsupported types,
invalid ranges, NULL activation elements, nonfinite x, and unrepresentable
activation scales produce DuckDB errors. Files are mapped per scalar input
row; these functions provide a validation surface without a model cache.

Q8 reference quantizers reproduce GGML's packed output for finite inputs
with finite reciprocal scales. Q8_K zero blocks include zeroed bsums (GGML
leaves these unused fields untouched). The low-level Q8_0 quantizer can encode
an infinite fp16 scale as GGML does; matvec rejects such scales.

### Quant oracle

```sh
make test quant-oracle
OPENBLAS_NUM_THREADS=1 Rscript scripts/audit_dequant.R
```

`test/make_quant_fixtures.c`, run by `make fixtures`, supplies deterministic
packed fields for every format, odd-width half rows, and a rank-3 tensor.
The R audit loads the built extension through DuckDB R and compares f32 bits
for every sampled row: all rows when there are <= 16, otherwise first, last,
and six seeded random rows. References are
`Rgguf::gguf_tensor(..., as = "numeric")` for tensors with <= 1 million
elements, and direct vendored GGML `dequantize_row_*` calls for larger tensors
and Q8_K. This bounds memory while using the same external to_float oracle.
The C oracle is a developer-only build from `GGML_DIR` (default
`/root/Rfmalloc/packages/Rggml/inst/ggml`); it is not linked into the extension.
Reports are written to `build/oracle/audit.tsv`.

Verified with Rgguf 0.1.0 and DuckDB R 1.5.5: all 747 tensors in the five
model files (256 LFM2.5-8B-A1B, 149 LFM2.5-ColBERT-350M, 316 EmbeddingGemma,
and 13 each tiny_llama / tiny_llama_bf16), plus 13 fixture tensors. All
**5,226,850 sampled values / 4,167 rows were bit-exact**, with 538 tensors
referenced through Rgguf and 222 through the C GGML oracle.

| Type | Tensors | Sampled rows | Exact values | Matvec tensors | Max relative error | Max scaled error |
|---|---:|---:|---:|---:|---:|---:|
| F32 | 343 | 806 | 597,532 | 62 | 5.83029e-5 | 1.21255e-7 |
| F16 | 2 | 9 | 3,590 | 2 | 1.48099e-6 | 1.02880e-7 |
| BF16 | 10 | 119 | 13,120 | 10 | 3.63640e-6 | 2.47605e-7 |
| Q8_0 | 172 | 1,375 | 1,146,368 | 172 | 31.4376 | 1.03118e-3 |
| Q2_K | 1 | 7 | 3,584 | 1 | 1.88024e-2 | 1.23019e-3 |
| Q3_K | 1 | 7 | 3,584 | 1 | 2.49661e-2 | 1.00579e-3 |
| Q4_K | 202 | 1,615 | 2,821,632 | 202 | 5.27793 | 1.34678e-3 |
| Q5_K | 1 | 7 | 3,584 | 1 | 2.67896e-2 | 5.82691e-4 |
| Q6_K | 27 | 215 | 630,272 | 27 | 0.482418 | 1.43799e-3 |
| Q8_K | 1 | 7 | 3,584 | 1 | 2.64251e-2 | 7.80302e-4 |

F16, Q2_K, Q3_K, Q5_K and Q8_K coverage comes from synthetic fixtures.
Conversion tests compare all bits, including NaN payloads and signed zero:
65,536 widening patterns and 362,144 narrowing patterns per fp16/bf16
converter. Q8_0 and Q8_K activation tests each compare 109 vectors / 111,616
values byte-for-byte, covering zeros, ties, signed maxima and extreme scales.
There were zero mismatches.

Matvec evaluates every rank >= 2 tensor (479 tensors, 3,886 sampled output
rows) against R double-precision `dequantize(W) %*% x`. Relative error means
`abs(error)/abs(reference)`; cancellation near zero makes it unbounded by
any format-wide percentage. Scaled error uses `sum(abs(W) * abs(x))` instead.
The largest scaled error was **0.143799%**. Each row must satisfy

```text
abs(error) <= sum(abs(W) * abs(xq - x))
              + gamma_(n+8) * sum(abs(W) * abs(xq))
              + 8 * epsilon_f32 * sum(abs(W) * abs(x))
gamma_m = m * epsilon_f32 / (1 - m * epsilon_f32)
```

Here xq is independently quantized/dequantized by GGML. The first term is
Q8 activation error by the triangle inequality; the remaining conservative
terms cover fixed f32 reduction and scale regrouping relative to rounded
GGML weights. Ordinary Q8 rounding contributes at most about
`block_amax/254` per element, with fp16 scale rounding also contributing for
Q8_0. All rows passed; the largest error/bound ratio was **0.179762**.

Bebelm's Rust Q4_K/Q6_K block grouping is the arithmetic reference for the
K dots. Its activation convention differs from GGML (positive scale and
half-away rounding), and its Q8_0 matvec uses unquantized f32 activations.
Thus full matvec bit equality to bebelm is not promised. The installed
Rbebelm exports no raw dot/matvec/dequant operation; a direct comparison
through its R API remains unverified. Other-platform execution and SIMD
implementations are also unverified.

`make test` and the specified ASan/UBSan extension suite passed, including
39 expected-error cases. The native C API tests also passed under ASan/UBSan,
covering row-range equivalence, unaligned scratch and short-buffer errors.

### One-core microbenchmark

```sh
make build/bench_quant
taskset -c 0 build/bench_quant \
  /root/modular/max/tests/integration/architectures/llama3/testdata/tiny_llama.gguf \
  /root/modular/max/tests/integration/architectures/llama3/testdata/tiny_llama_bf16.gguf \
  /root/bebelm/LFM2.5-8B-A1B-Q4_K_M.gguf \
  /root/bebelm/embeddinggemma-300M-Q8_0.gguf build/fixtures/quant.gguf
```

Intel Core i5-13500, CPU 0, GCC 12.2.0 `-O3 -ffp-contract=off`, portable
scalar path. One untimed matvec touches the pages, followed by at least
0.5 seconds of timed calls. GB/s counts weight payload bytes divided by
elapsed time, including activation quantization, not mapping. Small fixtures
fit in cache; the 215 MB Q6_K tensor exceeds CPU caches. These are
shape/data-dependent payload rates, not DRAM bandwidth measurements or
performance assertions.

| Type | Rows x columns | Tensor | GB/s |
|---|---:|---|---:|
| F32 | 128256 x 16 | tiny_llama token_embd.weight | 4.111 |
| BF16 | 128256 x 16 | tiny_llama_bf16 token_embd.weight | 0.081 |
| Q6_K | 128000 x 2048 | LFM2.5-8B token_embd.weight | 0.341 |
| Q4_K | 7168 x 2048 | LFM2.5-8B blk.0.ffn_gate.weight | 0.398 |
| Q8_0 | 3072 x 768 | EmbeddingGemma dense_2.weight | 1.637 |
| F16 | 7 x 512 | fixture w.1 | 1.743 |
| Q2_K | 7 x 512 | fixture w.10 | 0.184 |
| Q3_K | 7 x 512 | fixture w.11 | 0.204 |
| Q5_K | 7 x 512 | fixture w.13 | 0.322 |
| Q8_K | 7 x 512 | fixture w.15 | 1.205 |

The tiny BF16 tensor includes f32 subnormals (2,466 of its first 100,000
values are nonzero subnormals). Floating-point underflow behavior is preserved;
its unusually low rate is not representative of ordinary BF16 model weights.

## Direction

Next phases, each validated before the next starts:

1. **Forward pass on CPU.** One decoder family at a time in plain C, as in
   antirez/ds4: GGUF weights in place, the CPU quant kernels, and logits checked
   against a reference implementation, starting with the one-layer tiny Llama.
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
