
<!-- README.md is generated from README.Rmd. Please edit that file. -->

# ducksemantics

`ducksemantics` is a DuckDB-native semantic and retrieval substrate for
R. It owns ontology graphs, validated HPO observations, cataloged
Monarch relations, snapshot-bound literature retrieval, source-grounded
judgments, and optional local model/provider protocols. It does not own
ClinVar/PubMed source storage, case ranking, or evaluation.

Bulk values are ordinary data frames or caller-owned DuckDB relations.
S7 is reserved for real prompt, embedding, parser, and annotator
provider protocols. See [ARCHITECTURE.md](ARCHITECTURE.md) for the
current boundary.

## Install

``` r
install.packages(
  "ducksemantics",
  repos = c("https://sounkou-bioinfo.r-universe.dev", "https://cloud.r-project.org")
)
```

## Ontology candidates and accepted HPO observations

Lexical candidates are intentionally distinct from accepted
observations. An accepted observation must carry its HPO identifier,
exact zero-based half-open source span, context, method/provider
provenance, confidence, and explicit `accepted` status.

``` r
library(ducksemantics)

documents <- data.frame(
  document_id = "case-001",
  source_text = "The patient has seizures.",
  stringsAsFactors = FALSE
)

observations <- data.frame(
  document_id = "case-001",
  hpo_id = "HP:0001250",
  start_offset = 16L,
  end_offset = 24L,
  source_text = "seizures",
  context_status = "present",
  method = "lexical_alias",
  provider_id = "local-lexicon",
  provider_version = "1",
  confidence = 1,
  status = "accepted",
  stringsAsFactors = FALSE
)

ducksemantics_hpo_observations(documents, observations)
```

    ##   document_id     hpo_id start_offset end_offset source_text context_status
    ## 1    case-001 HP:0001250           16         24    seizures        present
    ##          method   provider_id provider_version confidence   status
    ## 1 lexical_alias local-lexicon                1          1 accepted

The validator rejects hallucinated text, empty spans, out-of-bounds
spans, and any context outside `present`, `absent/negated`,
`family_history`, `uncertain`, `conflict`, or `unsupported`.

## Cataloged Monarch facts

Callers provide both facts and a typed release catalog. A catalog row
identifies one provider/release and supplies a `Date`/`POSIXct`
effective date or a numeric source ordinal. Exact and as-of queries bind
the provider plus a cataloged release; historical gene-disease holdouts
are explicit anti-join audits.

``` r
releases <- data.frame(
  release_id = c("2024-01", "2025-01"),
  provider_id = "Monarch",
  effective_date = as.Date(c("2024-01-01", "2025-01-01")),
  source_ordinal = c(1, 2),
  stringsAsFactors = FALSE
)
gene_disease <- data.frame(
  gene_id = c("HGNC:1100", "HGNC:1100"),
  disease_id = c("MONDO:0000001", "MONDO:0000002"),
  release_id = c("2024-01", "2025-01"),
  provider_id = "Monarch",
  stringsAsFactors = FALSE
)

gene_disease <- ducksemantics_monarch_import(
  gene_disease, releases, relation = "gene_disease"
)
ducksemantics_monarch_gene_disease_holdout_audit(
  gene_disease, releases, provider_id = "Monarch",
  train_release_id = "2024-01", holdout_release_id = "2025-01"
)
```

    ##     gene_id    disease_id               predicate release_id provider_id
    ## 1 HGNC:1100 MONDO:0000002 biolink:associated_with    2025-01     Monarch
    ##   source_version attrs train_release_id holdout_release_id
    ## 1           <NA>  <NA>          2024-01            2025-01

## Snapshot-bound literature retrieval

`RClinVarbitration` owns append-only PubMed source history and
`ducksemantics` neither imports nor stores literature. Retrieval
consumes the same provider-scoped source-order projection:

- **snapshots:** `provider_id`, `snapshot_id`, finite integer
  `high_water_ordinal`, and optional `POSIXct` `effective_at`;
- **article versions:** `provider_id`, `article_id`, `pmid`,
  `version_id`, finite integer `source_ordinal`, and logical
  `is_deleted`;
- **version-bound sections:** `provider_id`, `article_id`, `pmid`,
  `version_id`, finite integer `source_ordinal`, `section`, and exact
  non-empty `text`.

For a cataloged provider/snapshot, it selects the maximum
`source_ordinal <= high_water_ordinal` per article, excludes an article
when that selected event is deleted, and joins sections only on all
version keys. Thus order is numeric (for example 10 follows 9), not
lexical. Exact nonempty zero-based half-open source spans are returned.

This is directly derivable from RClinVarbitration without a storage
adapter: `pubmed_sources` supplies provider, snapshot, source ordinal,
and application timestamp; `pubmed_articles` supplies the event and
deletion flag; and `pubmed_abstracts` supplies same-source version-bound
sections. A caller may also project `article_title` into a `title`
section. The relations remain caller-owned; `ducksemantics` does not
write or materialize a shadow copy.

``` r
snapshots <- data.frame(
  provider_id = c("pubmed", "pubmed"),
  snapshot_id = c("baseline", "update"),
  high_water_ordinal = c(9, 10),
  effective_at = as.POSIXct(c("2025-01-09", "2025-01-10"), tz = "UTC"),
  stringsAsFactors = FALSE
)
articles <- data.frame(
  provider_id = "pubmed",
  article_id = c("pmid:123", "pmid:123"),
  pmid = "123",
  version_id = c("source-9", "source-10"),
  source_ordinal = c(9, 10),
  is_deleted = c(FALSE, FALSE),
  stringsAsFactors = FALSE
)
sections <- data.frame(
  provider_id = "pubmed",
  article_id = c("pmid:123", "pmid:123"),
  pmid = "123",
  version_id = c("source-9", "source-10"),
  source_ordinal = c(9, 10),
  section = "BACKGROUND",
  text = c("Seizure phenotype in a family", "Updated seizure phenotype"),
  stringsAsFactors = FALSE
)

ducksemantics_literature_retrieve(
  articles, sections, snapshots,
  query = "seizure",
  provider_id = "pubmed",
  snapshot_id = "update"
)
```

    ##   provider_id snapshot_id high_water_ordinal effective_at article_id pmid
    ## 1      pubmed      update                 10   2025-01-10   pmid:123  123
    ##   version_id source_ordinal    section start_offset end_offset source_text
    ## 1  source-10             10 BACKGROUND            8         15     seizure
    ##                section_text
    ## 1 Updated seizure phenotype
