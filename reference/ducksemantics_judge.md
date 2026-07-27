# Judge lexical candidates with a provider

Judge lexical candidates with a provider

## Usage

``` r
ducksemantics_judge(
  text,
  mentions,
  runner,
  conn = NULL,
  prefix = "semantic",
  graph_context = NULL,
  instructions = ducksemantics_default_judgment_instructions(),
  prompt_builder = ducksemantics_judgment_prompt,
  parser = ducksemantics_json_judgment_parser(),
  record = !is.null(conn),
  model = "semantic-runner",
  ...
)
```

## Arguments

- text:

  Source text.

- mentions:

  Candidate relation.

- runner:

  Prompt provider.

- conn:

  Optional DBI connection to record judgments.

- prefix:

  Table prefix.

- graph_context:

  Optional graph context.

- instructions:

  Explicit judgment instructions.

- prompt_builder:

  Function building the prompt.

- parser:

  Judgment parser provider.

- record:

  Record result when `conn` is supplied.

- model:

  Provider/model identity.

- ...:

  Arguments passed to `prompt_builder`.

## Value

Source-grounded judgment relation.
