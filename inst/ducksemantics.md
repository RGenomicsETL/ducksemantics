# Ducksemantics Direction

**Current architecture:** ducksemantics owns ontology graphs, HPO extraction
observations, typed release-cataloged Monarch gene/phenotype/disease relations,
semantic and literature retrieval, source-grounded judgments, and real
model/provider protocols. The source of truth is the source checkout's
`ARCHITECTURE.md`.

Bulk values are data frames or caller-owned DuckDB relations. S7 is reserved
for genuine prompt, embedding, parser, and annotator protocols; it is not used
for connections, relations, matrices, batches, queries, indexes, or clustering
specifications.

## Relational boundaries

- Lexical ontology matching produces candidates only. Final HPO observations
  have validated zero-based half-open spans, exact source text, contextual
  status, provider provenance, confidence, and `status = "accepted"`.
- Monarch facts are supplied by the caller with a typed catalog keyed by
  `provider_id` and `release_id`. Catalog order is a `Date`/`POSIXct`
  effective date or numeric source ordinal. Exact/as-of release queries and
  holdouts retain provider/release identity. An official dated pack may be
  caller-attached and projected only through connection-local `TEMP VIEW`s:
  the projection accepts no `latest`, does not attach/download/collect/copy the
  pack, and binds one exact catalog row.
- `RClinVarbitration` owns append-only PubMed article and section events plus
  the provider snapshot catalog. Literature retrieval binds a provider and an
  exact source-order cutoff, selects each article's latest event at that
  ordinal, drops latest deletion events, and joins only same-version sections.
  It creates no literature table and does not load PubMed XML or bulk full
  text.
- Attached Monarch role projections retain every raw edge and normalize
  `gene_*`, `disease_*`, and `phenotype_*` only from explicit Biolink
  subject/object categories. Missing or malformed role data, missing
  predicates, negation, and unsupported orientations remain explicit statuses;
  they are never causal defaults, votes, or merged assertions.
- RClinVarbitration owns ClinVar/PubMed source relations. VariantStory owns
  case policy/ranking. VariantStoryBench owns evaluation.

## Graph and retrieval substrate

The DuckDB graph remains `semantic_nodes`, `semantic_aliases`,
`semantic_edges`, `semantic_entailed_edges`, `semantic_mentions`, and
`semantic_judgments`. Dense and token embeddings are regular data-frame rows
written to `semantic_embeddings` and `semantic_token_embeddings`; exact token
MaxSim reranking is retained. Optional local Rbebelm providers remain genuine
provider protocols, never mandatory remote services.

## RClinVarbitration literature projection

Literature retrieval directly consumes this read-only append-only projection:

```text
snapshots: provider_id, snapshot_id, high_water_ordinal, effective_at
articles:  provider_id, article_id, pmid, version_id, source_ordinal, is_deleted
sections:  provider_id, article_id, pmid, version_id, source_ordinal, section, text
```

`high_water_ordinal` and `source_ordinal` are finite integers;
`effective_at` is optional `POSIXct`; `is_deleted` is logical. At a cataloged
provider/snapshot, `ducksemantics` chooses the greatest article
`source_ordinal` not exceeding `high_water_ordinal`, drops that article if the
selected event is deleted, then joins sections on the complete version key.
The relations map read-only from RClinVarbitration's `pubmed_sources`,
`pubmed_articles`, and `pubmed_abstracts` relations and are never copied into
package-owned storage.
