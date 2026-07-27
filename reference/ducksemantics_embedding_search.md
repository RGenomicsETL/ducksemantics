# Search dense embeddings

Search dense embeddings

## Usage

``` r
ducksemantics_embedding_search(
  conn,
  embedding,
  provider = NULL,
  subject_kind = NULL,
  top_k = 10L,
  metric = c("cosine", "cosine_distance", "l2", "inner_product"),
  table = NULL
)
```

## Arguments

- conn:

  DBI connection.

- embedding:

  Numeric query vector.

- provider:

  Optional provider identity.

- subject_kind:

  Optional subject-kind filter.

- top_k:

  Number of results.

- metric:

  `"cosine"`, `"cosine_distance"`, `"l2"`, or `"inner_product"`.

- table:

  Caller-selected embedding table.

## Value

Ranked embedding rows.
