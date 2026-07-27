# Exact token-level late-interaction search

Exact token-level late-interaction search

## Usage

``` r
ducksemantics_late_interaction_search(
  conn,
  embeddings,
  provider = NULL,
  subject_kind = NULL,
  top_k = 10L,
  table = NULL,
  candidate_subject_id = NULL
)
```

## Arguments

- conn:

  DBI connection.

- embeddings:

  Numeric query-token matrix.

- provider:

  Optional provider identity.

- subject_kind:

  Optional subject-kind filter.

- top_k:

  Number of blocks to return.

- table:

  Caller-selected token embedding table.

- candidate_subject_id:

  Optional candidate restriction.

## Value

Ranked exact MaxSim rows.
