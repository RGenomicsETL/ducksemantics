# Write an OBO ontology into the graph

Write an OBO ontology into the graph

## Usage

``` r
ducksemantics_write_obo(
  conn,
  path,
  family,
  source = basename(path),
  prefix = "semantic",
  replace = FALSE,
  index = TRUE,
  include_obsolete = FALSE
)
```

## Arguments

- conn:

  DBI connection.

- path:

  OBO file path.

- family:

  Ontology family.

- source:

  Alias source identity.

- prefix:

  Table prefix.

- replace:

  Replace graph data.

- index:

  Rebuild alias index.

- include_obsolete:

  Include obsolete terms.

## Value

Parsed graph relations, invisibly.
