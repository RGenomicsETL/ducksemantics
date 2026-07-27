# Wrap a prompt function as a prompt provider

Wrap a prompt function as a prompt provider

## Usage

``` r
ducksemantics_prompt_runner(fun, label = "function")
```

## Arguments

- fun:

  Function accepting a prompt and returning response text.

- label:

  Provider identity.

## Value

An object implementing
[DucksemanticsPromptRunner](https://RGenomicsETL.github.io/ducksemantics/reference/DucksemanticsPromptRunner.md).
