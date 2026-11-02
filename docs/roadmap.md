# DuckSemantics roadmap

This roadmap describes candidate work. The native-kernel reference documents
only implemented and verified behavior.

## Typed decisions

A typed decision API can score only allowed option tokens for yes/no, choice,
and bounded-score questions. A result must include the model digest, prompt
template digest, option token ids, and kernel build receipt. This work must stay
within C11 and DuckDB's stable C ABI. It must not add generated-text parsing,
C++, ONNX Runtime, or DuckDB internal APIs.

## Explicit shared-prefix generation

The C state API can prefill a prefix and fork an independent continuation. A
future SQL API must make that reuse explicit rather than infer it from elapsed
time. The API decision remains between:

1. a table function that prefills one shared prefix, forks one state per
   continuation, and reports branch telemetry.
2. Independent SQL rows whose telemetry explicitly states that no cross-row KV
   reuse occurred.

The shared-prefix design has these implementation constraints:

- Tokenize the shared prefix once and prefill it once.
- Store KV and convolution state in contiguous arenas.
- Use flat branch descriptors and integer indices instead of pointer graphs or
  hash tables.
- Copy only the active state required by each branch.
- Keep invocation-local ownership and avoid a global prompt cache.
- Use structure-of-arrays layouts when branch phases consume separate fields.
- Keep small branch metadata inline when a bounded stack capacity is practical.
- Use bitsets for dense branch flags instead of byte-per-flag arrays.
- Report physical prompt work separately from avoided logical token work.

Telemetry must report shared-prefix tokens, physical prefill tokens, logical
reused tokens, forked branches, and copied state bytes. It must also report
fork/copy time, branch decode tokens, total phase times, and derived token and
row rates. Parity,
no-mutation-on-failure behavior, and bounded memory use are release conditions.

## Encoder-backed decisions

An encoder with a decision head can place EmbeddingGemma and ColBERT-backed
results behind the same typed receipts. Retrieval scores remain ranking signals,
not evidence authority. VariantStoryBench continues to own evaluation.

Model weights are caller-supplied. DuckSemantics does not bundle or download
them automatically.
