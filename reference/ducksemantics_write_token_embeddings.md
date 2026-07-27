# Store token embedding rows in DuckDB

Store token embedding rows in DuckDB

## Usage

``` r
ducksemantics_write_token_embeddings(
  conn,
  embeddings,
  prefix = "semantic",
  replace = FALSE
)
```

## Arguments

- conn:

  DBI connection.

- embeddings:

  Data frame from
  [`ducksemantics_token_embedding_batch()`](https://sounkou-bioinfo.github.io/ducksemantics/reference/ducksemantics_token_embedding_batch.md).

- prefix:

  Table prefix.

- replace:

  Replace existing rows with the same subject and provider.

## Value

Invisibly, written rows.
