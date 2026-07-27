# Validate a typed Monarch release catalog

A catalog row identifies one provider release. It must carry either a
typed `effective_date` (`Date` or `POSIXct`) or a finite numeric
`source_ordinal`; both may be supplied when the provider declares both.

## Usage

``` r
ducksemantics_monarch_release_catalog(releases, conn = NULL)
```

## Arguments

- releases:

  Data frame or caller-owned DuckDB table name containing `release_id`,
  `provider_id`, and one or both ordering columns.

- conn:

  Optional DBI connection for a table relation.

## Value

Validated release catalog.
