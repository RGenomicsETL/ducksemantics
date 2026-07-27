# Write ontology graph rows

Write ontology graph rows

## Usage

``` r
ducksemantics_write_graph(
  conn,
  nodes = NULL,
  aliases = NULL,
  edges = NULL,
  prefix = "semantic",
  replace = FALSE,
  index = TRUE
)
```

## Arguments

- conn:

  DBI connection.

- nodes:

  Data frame with `node_id` and `family`.

- aliases:

  Data frame with `node_id` and `alias`.

- edges:

  Data frame with `from_id`, `predicate`, and `to_id`.

- prefix:

  Table prefix.

- replace:

  Replace supplied target relations.

- index:

  Rebuild lexical alias index.

## Value

Invisibly, semantic table names.
