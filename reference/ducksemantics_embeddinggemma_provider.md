# Create an EmbeddingGemma provider

Create an EmbeddingGemma provider

## Usage

``` r
ducksemantics_embeddinggemma_provider(
  model,
  label = "Rbebelm EmbeddingGemma",
  task = "semantic_similarity",
  title = NULL,
  dimensions = 768L,
  normalize = TRUE,
  truncate = TRUE,
  check_interrupt = TRUE
)
```

## Arguments

- model:

  An `Rbebelm` `EmbeddingGemmaModel`.

- label:

  Provider identity.

- task:

  EmbeddingGemma task.

- title:

  Optional document title.

- dimensions:

  Matryoshka dimension.

- normalize:

  L2-normalize output rows.

- truncate:

  Truncate overly long input.

- check_interrupt:

  Poll for R interrupts.

## Value

An object implementing
[DucksemanticsEmbeddingProvider](https://RGenomicsETL.github.io/ducksemantics/reference/DucksemanticsEmbeddingProvider.md).
