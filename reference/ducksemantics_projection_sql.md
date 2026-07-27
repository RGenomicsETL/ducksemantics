# Project edge-shaped data to graph SQL

Project edge-shaped data to graph SQL

## Usage

``` r
ducksemantics_projection_sql(
  source_table,
  from,
  predicate,
  to,
  target_table = "semantic_edges",
  attrs = NULL,
  trust = NULL
)
```

## Arguments

- source_table:

  Source relation name.

- from, predicate, to:

  Source columns.

- target_table:

  Target table.

- attrs, trust:

  Optional source columns.

## Value

SQL script.
