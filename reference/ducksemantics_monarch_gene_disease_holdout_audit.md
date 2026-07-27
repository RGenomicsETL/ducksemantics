# Audit a historical Monarch gene-disease holdout

The selected training and holdout releases must be exact rows of one
typed provider catalog. The result is a provenance-preserving anti-join.

## Usage

``` r
ducksemantics_monarch_gene_disease_holdout_audit(
  facts,
  releases,
  provider_id,
  train_release_id,
  holdout_release_id,
  conn = NULL
)
```

## Arguments

- facts:

  Caller-supplied gene-disease relation.

- releases:

  Typed Monarch release catalog.

- provider_id:

  Provider identity.

- train_release_id:

  Exact training release identity.

- holdout_release_id:

  Exact holdout release identity.

- conn:

  Optional DBI connection for table relations.

## Value

Holdout rows absent from the historical relation.
