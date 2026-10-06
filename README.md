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
| `semantic_tokenize(path VARCHAR, text VARCHAR, add_special BOOLEAN)` | `INTEGER[]` token ids |
| `semantic_detokenize(path VARCHAR, ids INTEGER[])` | UTF-8 `VARCHAR` |
| `semantic_embed(path VARCHAR, text VARCHAR, task VARCHAR, dimensions INTEGER, normalize BOOLEAN)` | EmbeddingGemma `FLOAT[]` |
| `semantic_colbert_encode(path VARCHAR, text VARCHAR, role VARCHAR)` | L2-normalized token vectors as `FLOAT[][]`; role is `'query'` or `'document'` |
| `semantic_generate(path VARCHAR, prompt VARCHAR, max_tokens INTEGER)` | Greedy LFM2-MoE output as UTF-8 `VARCHAR` |
| `semantic_next_token_logits(path VARCHAR, prompt VARCHAR, token_ids INTEGER[])` | Raw next-token logits in requested-id order as `FLOAT[]` |

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

## ColBERT encoder

`semantic_colbert_encode` runs the closed LFM2.5-ColBERT-350M CPU profile:
16 layers, 1,024 hidden channels, centered gated short convolutions,
bidirectional GQA attention with QK RMSNorm and NeoX RoPE, SwiGLU, final
RMSNorm, and a 128-dimensional token projection. It adds `[Q] ` or `[D] `
and BOS. Queries retain exactly 32 positions, including PAD expansion vectors;
documents contextualize at most 512 positions and then remove the token ids
obtained by encoding individual ASCII punctuation characters without BOS.
All retained vectors are L2-normalized. Empty text and invalid roles raise
errors; NULL scalar arguments propagate NULL.

```sql
SELECT semantic_maxsim(
  semantic_colbert_encode('colbert.gguf', 'BRCA1 breast cancer', 'query'),
  semantic_colbert_encode('colbert.gguf', 'BRCA1 variant interpretation', 'document')
);
```

`src/lfm2.c` owns immutable process-lifetime model handles behind a mutex,
keyed by path, size and nanosecond mtime. Loading validates profile metadata,
tensor names, shapes and dtypes. Weights remain mmapped; only small F32 norm,
convolution and routing-bias weights are copied. Each call owns its bounded workspace
(32 query positions or 512 document positions, at most 1 MiB input including
the retrieval prefix). Superseded identities stay alive for in-flight callers;
there is no eviction. Model files must remain unchanged while mapped.

The additive `quant_bebel_*` kernels use bebelm's positive amax/127 scales,
half-away rounding and per-32 sums for K-quant integer dots. Batched products
unpack each weight block once across tokens. SSE2 integer dots and packed
single-token K-quant dots preserve the scalar arithmetic; non-SSE2 builds use
the scalar path. Existing GGML-compatible kernels
retain their arithmetic. RMSNorm sums f32 squares in f64; softmax sums f32
exponentials in f64. RoPE advances angles through iterative f32 multiplication.
Attention dots use explicit `fmaf` and a fixed four-accumulator reduction;
`-ffp-contract=off` prevents implicit contraction. Notices are in
`src/lfm2.LICENSE`.

The native kernel, NULL/error SQL, ASan/UBSan and 40,000-mutation GGUF fuzz
checks run through `make test`, `make sanitize` and `make fuzz`. Model-weight
verification is separate:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 2 \
  Rscript scripts/audit_colbert.R
```

The audit writes per-encoding vector errors, pair scores, timings and a summary
under `build/colbert-audit`. Its corpus includes biomedical queries, PubMed
abstracts, punctuation, Unicode, PAD expansion and inputs exceeding the query
and document limits. It compares vectors and token counts through
`colbert_encode_query`, `colbert_encode_document` and `colbert_embedding_vectors`,
pair scores through `colbert_maxsim`, and rankings through `colbert_rank`.

Verified on Linux x86-64, R 4.6.0, DuckDB R 1.5.5 and Rbebelm
0.3.6.0.1.0 (AVX2, one thread), with `LFM2.5-ColBERT-350M-Q4_K_M.gguf`:
**50 encodings / 1,913 retained token vectors / 624 query-document pairs**.
Token counts were identical, minimum per-token cosine was **1**, and maximum
absolute and element-relative vector errors were **0**. MaxSim maximum absolute
error was **8.64267349243164e-6**, maximum relative error
**2.98494769515267e-7**; all **24/24 rankings** matched.

On an Intel Core i5-13500, pinned to CPU 4, the C build
(`-O3 -ffp-contract=off`, SSE2 integer dots) produced **29.5965 query vectors/s**
and **26.6149 retained document vectors/s**, measured over this corpus including
loading and allocation. RSS after the C encodings and before loading the
Rbebelm reference model was **324,348 KiB** (316.746 MiB), including the R/DuckDB
host. These are observed wall-clock rates, not performance guarantees.
Other CPUs, Rbebelm backends and weight quantizations are unverified.

## LFM2-MoE generation

The closed LFM2.5-8B-A1B CPU profile has 24 layers, 2,048 hidden channels,
32 query heads / 8 KV heads, six causal attention layers with QK norms and
NeoX RoPE, and length-three causal gated convolutions. The first two FFNs are
dense SwiGLU; the other 22 route to four of 32 experts. Routing ranks sigmoid
scores plus expert bias, but mixes the selected experts using bias-free sigmoid
scores divided by their sum plus `1e-6`. The output projection is tied to the
mapped token embeddings.

```sql
SELECT semantic_generate('lfm2moe.gguf', 'The first ten prime numbers are', 32);
SELECT semantic_next_token_logits('lfm2moe.gguf', 'Answer: ', [32, 33, 34, 35]);
```

Raw-prompt semantics match `Rbebelm::bebel_generate`: BOS is added and supplied
ChatML framing is honored verbatim. Greedy decoding has no temperature, top-k
filter or repetition penalty. EOS and sequence-boundary controls stop decoding
without appearing in the returned text. A generated `</think>` masks those
controls for the following decision, matching bebelm's answer transition.

The context workspace is capped at **4,096 positions** including generation;
`max_tokens` is **1..1,024**, and raw prompt input is at most **1 MiB**.
Requests exceeding these bounds raise errors. There is no sliding-window
execution. Requested logits accept at most 128,000 non-NULL vocabulary ids;
order and duplicates are preserved, and an empty request returns `FLOAT[]`.
NULL scalar arguments propagate NULL. Scores are raw model logits, with no
generation-control masking or probability normalization.

`lfm2_state_new`, `lfm2_prefill`, `lfm2_state_logits`, `lfm2_decode` and
`lfm2_state_fork` expose the C state surface in `src/lfm2.h`. Each state owns KV,
convolution history, absolute position, normalized final hidden state and
scratch. Forking copies active KV, both history columns and position into
independent allocations; immutable model weights are borrowed. Decoder state
leaves the final emitted id pending, as in bebelm; prefill that id before
continuing. Constructors and operations require initialized non-NULL handles.
A failed forward operation invalidates the state for further use.

The weight-free cache isolation and packed/unpacked integer-dot tests are in
`test/lfm2_test.c`. The model-weight oracle builds a native sequence/timing
runner, compares all generated ids and SQL text to Rbebelm, records the first
divergence and C top-two logit margin, and verifies two warm-prefix branches
against cold full-prompt runs:

```sh
make build/lfm2_oracle
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 2 \
  Rscript scripts/audit_lfm2moe.R
```

Artifacts are written under `build/lfm2moe-audit`. A completed native run has
an input/artifact receipt; `--reuse-native` checks that receipt and still runs
every R-reference and SQL comparison. Each model is loaded once per process.
RSS comes from the native C process, excluding the R reference model. Prefill
includes the prompt's final logits projection; decode includes sampling and
the subsequent forward steps, matching bebelm's timing split.

Verified on Linux x86-64, R 4.6.0, DuckDB R 1.5.5 and Rbebelm 0.3.6.0.1.0,
with `LFM2.5-8B-A1B-Q4_K_M.gguf`: **52/52 greedy token sequences and SQL texts
matched**, including **50 prompts × 32 tokens (1,600 tokens)** plus two raw
completion prompts, for **1,664 generated tokens** in total. There was **no
first divergence**; the minimum C top-two logit margin was
**0.000213623047**. Requested-logit order, duplicate ids and typed empty lists
also passed. Both shared-prefix continuations matched cold full-prompt runs:
all five requested logits were bit-exact and all **8 generated ids per branch**
were identical, with independent KV/convolution/position ownership.

On the Intel Core i5-13500 pinned to CPU 2, one thread, `-O3 -ffp-contract=off`
with SSE2 integer dots: native **prefill 7.64155646 tokens/s**, native **decode
6.46806702 tokens/s**, and maximum native RSS **4,991,412 KiB**. SQL throughput
including prefill and loading was **3.63160192 generated tokens/s**. The tested
extension SHA256 was
`9b3b19c65beed67c9b1c5e24218ba4872955b6f777703ec9538ae44219c776f2`.
These are observed corpus rates; other CPUs, backends and weight quantizations
are unverified.

## Tokenizers

Vocabulary, ranked merges, scores, token types and BOS/EOS/unknown ids come
from validated GGUF metadata. Supported profiles are LFM2/LFM2MoE with the
`gpt2`/`lfm2` tokenizer and EmbeddingGemma with the `llama` tokenizer.
Hash tables and a priority queue implement ranked byte BPE and score-priority
SentencePiece-style BPE, with leftmost tie-breaking. There is no regex runtime,
embedded vocabulary, or kernel threading.

`add_special = true` prepends BOS for LFM and applies the GGUF BOS/EOS policy
for Gemma. LFM control/user-defined literals are atomic regardless of this flag.
Gemma replaces ASCII spaces with U+2581 and adds no dummy leading space.
Tokenization does not truncate, pad, format chat prompts, or add retrieval
prefixes. For raw ColBERT tokenization, callers supply `[Q] ` or `[D] `
explicitly; `semantic_colbert_encode` owns retrieval prefixes, query padding,
document truncation and punctuation filtering.

LFM decoding follows Rbebelm's byte-character inverse, including representable
special-token literals. Gemma decoding omits control tokens, converts U+2581 to
ASCII space and joins byte-fallback tokens. A literal U+2581 input therefore
normalizes to a space on Gemma round-trip. Invalid decoded byte sequences become
U+FFFD; invalid UTF-8 input, out-of-range ids and NULL list elements raise errors.
NULL scalar arguments propagate NULL. Empty text and empty id lists are valid.

An immutable, mutex-protected process cache owns each tokenizer and GGUF mapping,
keyed by the supplied path, file size and nanosecond mtime. Vocabulary is loaded
once per identity. Entries, including superseded identities, remain mapped until
process teardown; workloads using many distinct files retain their vocabulary
memory. Loading holds the cache lock; encoding and decoding use per-call scratch
storage without holding it. Model files must remain unchanged while mapped.

### Unicode and verification

The pre-tokenizer uses Unicode 17.0.0 `Alphabetic`, numeric categories
`Nd`/`Nl`/`No`, and `White_Space`, matching the Rust character predicates.
Regenerate the committed ranges with:

```sh
Rscript scripts/generate_tokenizer_unicode.R
```

The generator downloads `DerivedCoreProperties.txt`, `UnicodeData.txt` and
`PropList.txt` from the versioned Unicode UCD into `build/unicode-data`, verifies
pinned checksums, and writes `src/tokenizer_unicode.h`. These are development
inputs, not build or runtime dependencies.

```sh
Rscript scripts/audit_tokenizer.R build/ducksemantics.duckdb_extension
```

The installed Rbebelm 0.3.6.0.1.0 oracle, through DuckDB R 1.5.5 on R 4.6.0,
matched this 426-string corpus:

| Profile / comparison | Exact sequences | Maximum id error |
|---|---:|---:|
| LFM2.5-8B-A1B, with BOS / without BOS | 426 / 426 | 0 |
| LFM2.5-ColBERT-350M, query / retained document ids | 425 / 425 | 0 |
| EmbeddingGemma-300M, full with specials / without specials | 424 / 424 | 0 |
| EmbeddingGemma, context-truncated sequences | 2 | 0 |

LFM decode matched 430 reference strings, including four additional id sequences.
The corpus covers mixed whitespace, digits, punctuation, emoji, CJK, Arabic,
accented Latin, combining marks, empty text, code and inputs over 10,000
characters. Biomedical inputs include titles from the RClinVarbitration XML
fixtures, ClinVar-style names, and sourced PubMed abstracts in
`scripts/tokenizer_biomedical.tsv` (PMIDs 8524414 and 28976996).
Each model is loaded once. Receipts and session information are written under
`build/tokenizer-audit` as RDS/text files.

ColBERT's R encoder rejects empty text and exposes only padded/truncated retained
ids, not full raw tokenization or decode. Gemma's R wrapper limits results to
2,048 tokens and has no sequence decoder: two over-context inputs verify the
truncation policy, not their full tails. Two other inputs over 10,000 characters
fit within that context and match in full. Gemma's no-special comparison removes
BOS/EOS from the reference output. External acceptance uses Rbebelm.

Warm-cache SQL throughput, one DuckDB thread, GCC `-O3`, Intel Core i5-13500;
200 corpus repetitions, excluding vocabulary/model loading:

| Profile | Tokens processed | Seconds | Tokens/s |
|---|---:|---:|---:|
| LFM2.5-8B-A1B | 5,108,600 | 1.097 | 4,656,882 |
| LFM2.5-ColBERT-350M | 5,869,600 | 1.087 | 5,399,816 |
| EmbeddingGemma-300M | 5,434,400 | 3.044 | 1,785,283 |

`make test` passes the SQL assertions, expected-error cases and native cache/
UTF-8 tests. The prescribed ASan/UBSan shared-library run also passes; the native
cache test additionally passes with ASan leak detection enabled. Tokenizer
fixtures are produced by `test/make_tokenizer_fixtures.c` through `make fixtures`.

## EmbeddingGemma encoder

`semantic_embed` implements Rbebelm's EmbeddingGemma-300M forward pass for the
`gemma-embedding` GGUF profile with Q8_0/F32 weights: 24 bidirectional layers,
768 hidden values, three 256-wide query heads and one KV head, GeGLU, and RMS
norms using the stored gains. Five symmetric local-attention layers (radius
256, up to 513 keys) alternate with one global layer. Local/global RoPE bases
are 10,000/1,000,000. Pooling averages all retained tokens, including BOS/EOS;
two linear heads project 768 → 3,072 → 768, without an intervening activation.

```sql
SELECT semantic_embed('/root/bebelm/embeddinggemma-300M-Q8_0.gguf',
                      'BRCA1 c.5266dup variant interpretation',
                      'retrieval_query', 256, true);
```

Task names and prefixes match Rbebelm exactly:

| Task | Input prefix |
|---|---|
| `retrieval_query` | `task: search result \| query: ` |
| `retrieval_document` | `title: none \| text: ` |
| `question_answering` | `task: question answering \| query: ` |
| `fact_verification` | `task: fact checking \| query: ` |
| `classification` | `task: classification \| query: ` |
| `clustering` | `task: clustering \| query: ` |
| `semantic_similarity` | `task: sentence similarity \| query: ` |
| `code_retrieval` | `task: code retrieval \| query: ` |
| `summarization` | `task: summarization \| query: ` |
| `raw` | empty; input is already formatted |

Formatting precedes tokenization. Inputs beyond 2,048 tokens are truncated,
with EOS retained as the last token. Dimensions must be 768, 512, 256 or 128:
select the leading dimensions, then optionally L2-normalize that subvector.
Empty text is valid; NULL arguments yield NULL. Unknown tasks, dimensions,
architectures, tensor shapes and tensor types raise errors. The formatted input
limit is 16 MiB. Custom document titles can be formatted explicitly with `raw`.

Weights stay in the read-only GGUF mapping; only the small norm vectors are
cached as floats. Model handles and their tokenizers are immutable process-cache
entries keyed by supplied path, size and nanosecond mtime, protected by a mutex.
Old identities remain valid until process teardown; many identities retain
vocabulary memory and mappings. Files must remain unchanged while mapped.
Encoding uses per-call scratch without the cache lock, with at most 2,048
packed tokens. Attention and positions are isolated per sequence. Consecutive
identical token sequences share the full projection across dimension/normalization
requests within a chunk. Encoder scratch is approximately 56 MiB, plus a 6 MiB
projection buffer and DuckDB output lists.

`quant_bebelm_matmul` and `quant_bebelm_dot` are additive Rbebelm-convention
kernels in `src/quant.[ch]`; GGML-convention kernels are unchanged. Q8_0 and F32
projections use floating products, without quantizing activations. Dots use four
8-lane accumulators and an explicit fused multiply-add, with fixed reduction
order. AVX2/FMA dispatch accelerates that arithmetic; the portable path uses
`fmaf`. Other operations compile with `-ffp-contract=off`, without fast-math.
Matrix work is tiled over 32 tokens. The encoder is single-threaded.

```sh
Rscript scripts/audit_embeddinggemma.R build/ducksemantics.duckdb_extension
OPENBLAS_NUM_THREADS=1 taskset -c 10 Rscript scripts/audit_embeddinggemma.R --benchmark
```

The oracle uses 200 texts: ASCII, Unicode, ClinVar-style variant names, synthetic
PubMed-style abstracts, and two inputs requiring truncation. It compares every
task and dimension, with and without normalization, against installed Rbebelm;
query/document convenience wrappers are also checked. It requires cosine
≥ 0.99999, maximum absolute error ≤ 1e-4 for every vector, and identical ordered
top-10 cosine neighbours over the corpus (excluding self, ties by corpus id).
Rbebelm manages its own packed reference batches. Metrics, corpus, worst-error
locations and runtime information are written under `build/embeddinggemma-audit`.
`RBEBELM_AUDIT_TASKS` and `RBEBELM_AUDIT_DIMENSIONS` select comma-separated tasks
and dimensions for independent oracle shards. `--summary` requires the complete
10-task × 4-dimension × 2-normalization grid before aggregating their metrics.

Verified against **Rbebelm 0.3.6.0.1.0**, R 4.6.0 and DuckDB R 1.5.5 on AVX2/FMA:

| Comparison | Count | Observed worst |
|---|---:|---:|
| Primary vectors | 16,000 vectors / 6,656,000 values | absolute error **0**, relative error **0**, minimum cosine **1** |
| Query/document wrappers | 3,200 vectors / 1,331,200 values | absolute error **0** |
| Ordered top-10 cosine neighbours | 16,000 rankings / 160,000 positions | **0 mismatches** |

All 200 texts were checked for every task, dimension and normalization setting;
the two long inputs truncated in every setting. Detailed tables are
`build/embeddinggemma-audit/metrics-all.tsv` and `wrappers-all.tsv`. Model MD5:
`29ebd01c9362065f36405ef185c7755d`; audited and rebuilt extension MD5:
`3363724715a7da6587b1dc49d9534634`. `make test`, `make sanitize` and `make fuzz`
pass; the fuzz gate covers 40,000 mutations over two seeds.

Warm-cache forward throughput on one pinned Intel Core i5-13500 P-core (logical
CPU 10), GCC `-O3`, AVX2/FMA, DuckDB R 1.5.5 / R 4.6.0, excluding model loading:
198 nontruncated retrieval queries, 8,639 tokens including prompts and BOS/EOS.
Median of three runs: **80.129 s, 107.81365 tokens/s, 2.47102 texts/s**. Median
CPU time was 80.125 s (107.81903 tokens/CPU-second). Other oracle processes ran
on different physical cores; this is the observed throughput under that load.
Samples and runtime details are in `build/embeddinggemma-audit/benchmark`.

The forward oracle targets AVX2/FMA on this host. Non-AVX2 end-to-end parity
and performance require separate testing; scalar fused-dot arithmetic is
checked against the dispatched kernel by the native test.

## Build and test

```sh
make           # build/ducksemantics.duckdb_extension (C compiler + POSIX sh only)
make test      # C-generated fixtures; SQL assertions and expected errors on the DuckDB CLI
make sanitize  # the same suite against an ASan/UBSan build of the extension
make fuzz      # 20,000 mutations per seed through the reader, under ASan/UBSan
make audit     # R oracle comparison on real models (needs duckdb and Rgguf for R)
```

Loading needs `allow_unsigned_extensions` (`duckdb -unsigned`). The repository
contains C, shell and R only.

Verification:

- `make audit` runs `scripts/audit_gguf.R` against Rgguf, which parses with
  ggml's official GGUF implementation. It compares every metadata value, array
  element and tensor field. It matched on seven real models: tiny Llama (f32
  and bf16), ESM-2, EmbeddingGemma 300M (Q8_0), LFM2.5-ColBERT 350M (Q4_K_M),
  LFM2.5-8B-A1B (Q4_K_M) and Ternary-Bonsai-27B (`q2_0`). That is 2.8M array
  elements and 1,704 tensors. Bonsai's 851 tensors tile its 7.6 GB file
  exactly. Dims are reported as stored; ggml drops trailing 1s, and the audit
  compares under that convention.
- `make fuzz` (`test/fuzz_gguf.c`) links the reader directly, mutates
  fixtures, and touches every scalar, array element, tensor field and payload
  boundary. 40,000 mutations ran under ASan/UBSan with no fault.

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
Thus `gguf_tensor_matvec` bit equality to bebelm is not promised. The installed
Rbebelm exports no raw dot/matvec/dequant operation; a direct comparison
through its R API remains unverified. Other-platform execution and SIMD
implementations are also unverified.

`make test` and the specified ASan/UBSan extension suite passed, including
the expected-error cases. The native C API tests also passed under ASan/UBSan,
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

1. **Typed decisions.** A Jev-compatible `semantic_system_one(state,
   questions)` returns per-option probabilities for yes/no, choice and score
   questions. It scores only the allowed option tokens, with no text
   generation or parsing, and returns a receipt of model digest, prompt
   template digest, option token ids and kernel build.
2. **KV reuse inside DuckDB chunks.** The cache for the shared instruction
   prefix is computed once. Within a chunk, rows that share a document reuse
   its KV state and branch per question. A per-thread cache carries a run
   across chunk boundaries, keyed by a hash of the rendered byte prefix as in
   ds4's KV store. Several questions on one document are evaluated inside one
   call in cost order, so an AI filter short-circuits on cached state.
3. **Encoder-backed decisions and receipts.** A Laya-style encoder with a
   decision head, with EmbeddingGemma and ColBERT results behind the same
   receipts.

Model weights are never bundled; the caller supplies a path and the receipt
records its digest.
