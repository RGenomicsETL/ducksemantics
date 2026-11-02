# Example resources

`source-releases.csv` declares the pinned Mondo and DisMech source receipts.
`mondo-subset.obo` contains complete upstream term stanzas.
`dismech-marfan.yaml` contains the upstream Marfan syndrome record.
The import APIs verify their bytes against the declared SHA-256 digests.

The `readme-*.csv` files are synthetic examples for the R-package walkthrough:

- `readme-hpo-*` provides one document and its exact phenotype observation.
- `readme-monarch-*` provides two miniature release snapshots and their facts.
  These rows demonstrate release selection, not biological associations.
- `readme-literature-*` provides two versions of one synthetic article.
  Its identifiers and text are not a publication evidence source.

Model weights are not included.
