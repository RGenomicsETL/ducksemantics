# Wrap an embedding function as an embedding provider

Wrap an embedding function as an embedding provider

## Usage

``` r
ducksemantics_embedding_provider(fun, label = "function")
```

## Arguments

- fun:

  Function accepting a prompt and returning response text.

- label:

  Provider identity.

## Value

An object implementing
[DucksemanticsEmbeddingProvider](https://RGenomicsETL.github.io/ducksemantics/reference/DucksemanticsEmbeddingProvider.md).
