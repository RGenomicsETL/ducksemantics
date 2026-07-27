# Generate lexical HPO/ontology candidates

These candidates are not accepted observations. Use
[`ducksemantics_hpo_observations()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_hpo_observations.md)
to validate a separately accepted HPO observation relation against
source documents.

## Usage

``` r
ducksemantics_annotate(
  conn,
  text,
  document_id = NULL,
  prefix = "semantic",
  longest_match = TRUE,
  record = FALSE
)
```

## Arguments

- conn:

  DBI connection.

- text:

  Source text.

- document_id:

  Optional source-document identity.

- prefix:

  Table prefix.

- longest_match:

  Drop spans contained by a longer span.

- record:

  Append candidate rows to the semantic mention table.

## Value

Candidate data frame.
