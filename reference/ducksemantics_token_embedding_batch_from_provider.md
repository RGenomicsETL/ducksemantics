# Construct token embedding rows from a provider

Construct token embedding rows from a provider

## Usage

``` r
ducksemantics_token_embedding_batch_from_provider(
  text,
  provider,
  subject_id = text,
  subject_kind = "node",
  provider_label = NULL,
  block_id = NULL,
  attrs = NULL,
  ...
)
```

## Arguments

- text:

  Character vector to embed.

- provider:

  Object implementing
  [DucksemanticsTokenEmbeddingProvider](https://sounkou-bioinfo.github.io/ducksemantics/reference/DucksemanticsTokenEmbeddingProvider.md).

- subject_id:

  Subject identifiers for input texts.

- subject_kind:

  Subject type for stored rows.

- provider_label:

  Stored provider identity.

- block_id:

  Optional block identifier per input text.

- attrs:

  Optional metadata per input text.

- ...:

  Arguments forwarded to
  [`ducksemantics_token_embed()`](https://sounkou-bioinfo.github.io/ducksemantics/reference/ducksemantics_provider_generics.md).

## Value

A token-embedding data frame.
