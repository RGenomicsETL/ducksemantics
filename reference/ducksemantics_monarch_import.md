# Import caller-supplied Monarch relations

This function never downloads Monarch data. A typed release catalog is
required to prove that every fact has a known provider/release identity.

## Usage

``` r
ducksemantics_monarch_import(
  facts,
  releases,
  relation = c("gene_phenotype", "gene_disease", "disease_phenotype"),
  conn = NULL,
  table = NULL,
  replace = FALSE
)
```

## Arguments

- facts:

  Gene-phenotype, gene-disease, or disease-phenotype data frame, or a
  caller-owned DuckDB table name with `conn`.

- releases:

  Typed catalog from
  [`ducksemantics_monarch_release_catalog()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_monarch_release_catalog.md).

- relation:

  One of `"gene_phenotype"`, `"gene_disease"`, or `"disease_phenotype"`.

- conn:

  Optional DBI connection.

- table:

  Optional destination table when `conn` is supplied.

- replace:

  Replace destination rows.

## Value

Normalized Monarch relation.
