# Connect to a DuckDB semantic store

Connect to a DuckDB semantic store

## Usage

``` r
ducksemantics_connect(dbdir = ":memory:", read_only = FALSE, array = "matrix")
```

## Arguments

- dbdir:

  DuckDB path or `":memory:"`.

- read_only:

  Open read-only.

- array:

  DuckDB array conversion mode.

## Value

A DBI connection.
