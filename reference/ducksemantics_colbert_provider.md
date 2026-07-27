# Create a native ColBERT provider

Create a native ColBERT provider

## Usage

``` r
ducksemantics_colbert_provider(
  model,
  role = c("document", "query"),
  label = "Rbebelm ColBERT"
)
```

## Arguments

- model:

  An `Rbebelm` `ColbertModel`.

- role:

  Query or document encoding role.

- label:

  Provider identity.

## Value

An object implementing
[DucksemanticsTokenEmbeddingProvider](https://sounkou-bioinfo.github.io/ducksemantics/reference/DucksemanticsTokenEmbeddingProvider.md).
