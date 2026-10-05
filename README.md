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
| `semantic_tokenize(path VARCHAR, text VARCHAR, add_special BOOLEAN)` | `INTEGER[]` token ids |
| `semantic_detokenize(path VARCHAR, ids INTEGER[])` | UTF-8 `VARCHAR` |

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
prefixes. ColBERT callers supply `[Q] ` or `[D] ` explicitly; its encoder applies
query padding, document truncation and punctuation filtering separately.

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

`make test` passes the SQL assertions, 36 expected-error cases and native cache/
UTF-8 tests. The prescribed ASan/UBSan shared-library run also passes; the native
cache test additionally passes with ASan leak detection enabled. Tokenizer
fixtures are produced by `test/make_tokenizer_fixtures.c` through `make fixtures`.

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
