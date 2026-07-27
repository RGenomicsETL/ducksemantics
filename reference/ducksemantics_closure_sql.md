# Materialize transitive closure SQL

Materialize transitive closure SQL

## Usage

``` r
ducksemantics_closure_sql(
  transitive_predicates,
  source_table = "semantic_edges",
  target_table = "semantic_entailed_edges"
)
```

## Arguments

- transitive_predicates:

  Non-empty predicate names, or
  [`character()`](https://rdrr.io/r/base/character.html).

- source_table:

  Source edge relation.

- target_table:

  Target closure relation.

## Value

SQL script.
