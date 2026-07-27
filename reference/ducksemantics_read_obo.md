# Read an OBO ontology into graph relations

Read an OBO ontology into graph relations

## Usage

``` r
ducksemantics_read_obo(
  path,
  family,
  source = basename(path),
  include_obsolete = FALSE
)
```

## Arguments

- path:

  OBO file path.

- family:

  Ontology family.

- source:

  Alias source identity.

- include_obsolete:

  Include obsolete terms.

## Value

A list of `nodes`, `aliases`, and `edges` data frames.
