# Construct token embedding rows

Construct token embedding rows

## Usage

``` r
ducksemantics_token_embedding_batch(
  embeddings,
  subject_id,
  subject_kind = "node",
  provider = "embedding",
  token_index = NULL,
  block_id = NULL,
  token = NULL,
  start_offset = NULL,
  end_offset = NULL,
  attrs = NULL
)
```

## Arguments

- embeddings:

  Finite numeric matrix with one row per subject.

- subject_id:

  Non-empty subject identifiers.

- subject_kind:

  Subject type.

- provider:

  Provider identity.

- token_index:

  Zero-based token index within a document block.

- block_id:

  Optional block identifier.

- token:

  Optional token text.

- start_offset, end_offset:

  Optional zero-based half-open source offsets.

- attrs:

  Optional metadata text per row.

## Value

A data frame suitable for
[`ducksemantics_write_token_embeddings()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_write_token_embeddings.md).
