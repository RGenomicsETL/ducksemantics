# Validate accepted HPO observations against source documents

Validate accepted HPO observations against source documents

## Usage

``` r
ducksemantics_hpo_observations(documents, observations, conn = NULL)
```

## Arguments

- documents:

  Data frame (or a caller-owned DuckDB table name with `conn`)
  containing `document_id` and `source_text`.

- observations:

  Data frame (or caller-owned DuckDB table name) following
  [`ducksemantics_hpo_observation_contract()`](https://sounkou-bioinfo.github.io/ducksemantics/reference/ducksemantics_hpo_observation_contract.md).

- conn:

  Optional DBI connection for a lazy caller-owned table name.

## Value

Validated accepted-observation data frame.
