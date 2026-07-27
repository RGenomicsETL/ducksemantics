# Wrap a token embedding function as a token provider

Wrap a token embedding function as a token provider

## Usage

``` r
ducksemantics_token_embedding_provider(fun, label = "function-token")
```

## Arguments

- fun:

  Function accepting a prompt and returning response text.

- label:

  Provider identity.

## Value

An object implementing
[DucksemanticsTokenEmbeddingProvider](https://RGenomicsETL.github.io/ducksemantics/reference/DucksemanticsTokenEmbeddingProvider.md).
