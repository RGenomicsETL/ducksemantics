# Validate an append-only literature snapshot catalog

`RClinVarbitration` owns article and section storage. This catalog
identifies a provider snapshot by its source-order cutoff; it creates no
shadow literature table.

## Usage

``` r
ducksemantics_literature_snapshot_catalog(snapshots, conn = NULL)
```

## Arguments

- snapshots:

  Data frame or caller-owned DuckDB table name containing `provider_id`,
  `snapshot_id`, and finite integer `high_water_ordinal`; `effective_at`
  is an optional typed `POSIXct` timestamp.

- conn:

  Optional DBI connection for a table relation.

## Value

Validated source snapshot catalog.
