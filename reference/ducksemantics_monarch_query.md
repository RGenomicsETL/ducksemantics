# Select one cataloged Monarch release

Select one cataloged Monarch release

## Usage

``` r
ducksemantics_monarch_query(
  facts,
  releases,
  relation = c("gene_phenotype", "gene_disease"),
  provider_id,
  release_id = NULL,
  as_of = NULL,
  as_of_ordinal = NULL,
  conn = NULL
)
```

## Arguments

- facts:

  Monarch fact relation or a caller-owned DuckDB table name.

- releases:

  Typed Monarch release catalog.

- relation:

  Relation type.

- provider_id:

  Provider identity to bind.

- release_id:

  Exact catalog release identity.

- as_of:

  Typed `Date` or `POSIXct` cutoff.

- as_of_ordinal:

  Numeric source-order cutoff.

- conn:

  Optional DBI connection for table relations.

## Value

Facts from one selected provider/release. Selection metadata is held in
`release_selected` and `provider_selected` attributes.
