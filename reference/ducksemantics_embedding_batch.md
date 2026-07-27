# Construct embedding rows

Bulk embedding values are an ordinary data frame with one
`FLOAT[]`-ready numeric vector per row; they are not an S7 value object.

## Usage

``` r
ducksemantics_embedding_batch(
  embeddings,
  subject_id,
  subject_kind = "node",
  provider = "embedding",
  text = NULL,
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

- text:

  Optional source text per row.

- attrs:

  Optional metadata text per row.

## Value

A data frame suitable for
[`ducksemantics_write_embeddings()`](https://sounkou-bioinfo.github.io/ducksemantics/reference/ducksemantics_write_embeddings.md).
