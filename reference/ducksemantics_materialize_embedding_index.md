# Materialize a fixed-dimension embedding table

Materialize a fixed-dimension embedding table

## Usage

``` r
ducksemantics_materialize_embedding_index(
  conn,
  dimensions,
  provider = NULL,
  subject_kind = NULL,
  table = NULL,
  source_table = "semantic_embeddings",
  hnsw = FALSE,
  metric = "cosine"
)
```

## Arguments

- conn:

  DBI connection.

- dimensions:

  Embedding width.

- provider:

  Optional provider filter.

- subject_kind:

  Optional subject kind filter.

- table:

  Target table name.

- source_table:

  Source embedding table.

- hnsw:

  Create an optional DuckDB HNSW index.

- metric:

  HNSW metric.

## Value

Target table name.
