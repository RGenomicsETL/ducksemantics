# Provider protocol generics

These S7 generics are reserved for real pluggable providers. All bulk
input and output remains an ordinary data frame or matrix.

## Usage

``` r
ducksemantics_run(provider, prompt, ...)

ducksemantics_embed(provider, text, ...)

ducksemantics_token_embed(provider, text, ...)

ducksemantics_parse(parser, response, ...)

ducksemantics_ground(
  annotator,
  conn,
  text,
  document_id = NULL,
  prefix = "semantic",
  longest_match = TRUE,
  record = FALSE,
  ...
)
```

## Arguments

- provider:

  Prompt or embedding provider.

- prompt:

  Prompt text.

- ...:

  Provider-specific arguments.

- text:

  Source text.

- parser:

  Judgment parser.

- response:

  Raw response text.

- annotator:

  Grounding provider.

- conn:

  DBI connection.

- document_id:

  Optional document identifier.

- prefix:

  Semantic-table prefix.

- longest_match:

  Drop nested lexical candidates.

- record:

  Persist candidates.

## Value

Provider-specific text, matrix, or data frame.
