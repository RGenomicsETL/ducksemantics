# HPO observation contract

The final observation relation has exactly these required semantic
fields: `document_id`, `hpo_id`, `start_offset`, `end_offset`,
`source_text`, `context_status`, `method`, `provider_id`,
`provider_version`, `confidence`, and `status`. Offsets are zero-based,
half-open R character offsets. Lexical candidates returned by
[`ducksemantics_annotate()`](https://sounkou-bioinfo.github.io/ducksemantics/reference/ducksemantics_annotate.md)
are not observations and cannot be substituted for this relation.

## Usage

``` r
ducksemantics_hpo_observation_contract()
```

## Value

Required HPO observation column names.
