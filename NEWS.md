# ducksemantics 0.1.0

- Sealed package ownership around ontology graphs, validated HPO observations,
  typed release-cataloged Monarch relations, snapshot-catalog-bound literature
  retrieval, and source-grounded semantic/provider protocols.
- Added the final relational HPO observation contract. It validates exact,
  non-empty zero-based half-open source spans and records context, method,
  provider identity/version, confidence, and explicit accepted status.
- Added caller-supplied Monarch gene-phenotype/gene-disease facts with a required
  typed provider/release catalog. Exact and Date/order as-of queries reject
  unknown releases; historical gene-disease holdouts are provenance-preserving
  anti-join audits.
- Literature retrieval now consumes RClinVarbitration's caller-owned
  append-only source-order projection: cataloged integer cutoffs select the
  latest article event, discard latest deletion events, and join exact
  same-version sections before returning source spans. ducksemantics neither
  imports nor stores literature relations.
- Removed ducksemantics benchmark APIs (owned by VariantStoryBench), generic
  cache APIs, and S7 wrappers for relations, connections, matrices, batches,
  queries, indexes, and clustering specifications. Embedding and reranking now
  use ordinary data frames, matrices, and scalar arguments.
