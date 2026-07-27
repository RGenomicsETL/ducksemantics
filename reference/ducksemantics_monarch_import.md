# Import caller-supplied Monarch relations

This function never downloads Monarch data. A typed release catalog is
required to prove that every fact has a known provider/release identity.

## Usage

``` r
ducksemantics_monarch_import(
  facts,
  releases,
  relation = c("gene_phenotype", "gene_disease"),
  conn = NULL,
  table = NULL,
  replace = FALSE
)
```

## Arguments

- facts:

  Gene-phenotype or gene-disease data frame, or a caller-owned DuckDB
  table name with `conn`.

- releases:

  Typed catalog from
  [`ducksemantics_monarch_release_catalog()`](https://sounkou-bioinfo.github.io/ducksemantics/reference/ducksemantics_monarch_release_catalog.md).

- relation:

  Either `"gene_phenotype"` or `"gene_disease"`.

- conn:

  Optional DBI connection.

- table:

  Optional destination table when `conn` is supplied.

- replace:

  Replace destination rows.

## Value

Normalized Monarch relation.
