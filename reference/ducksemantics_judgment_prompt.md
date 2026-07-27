# Build a source-grounded judgment prompt

Build a source-grounded judgment prompt

## Usage

``` r
ducksemantics_judgment_prompt(
  text,
  mentions,
  graph_context = NULL,
  instructions = ducksemantics_default_judgment_instructions()
)
```

## Arguments

- text:

  Source text.

- mentions:

  Candidate data frame.

- graph_context:

  Optional graph context relation.

- instructions:

  Explicit judgment policy.

## Value

Prompt text.
