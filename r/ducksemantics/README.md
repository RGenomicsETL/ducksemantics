
<!-- README.md is generated from README.Rmd. Edit README.Rmd. -->

# ducksemantics

`ducksemantics` turns phenotype mentions, Monarch associations and
literature passages into DuckDB-ready relations that downstream tools
can check. A phenotype observation is accepted only when its text
matches the exact span it cites, a Monarch fact only within a cataloged
provider release, and a literature hit only from the article version
visible at a named snapshot.

These contracts exist because each project has one authority.
RClinVarbitration owns ClinVar and PubMed source history.
`ducksemantics` validates and queries semantic relations without copying
that history. VariantStoryBench owns case ranking and evaluation. This
walkthrough executes the acceptance and rejection paths that sit between
them.

## Install

``` r
install.packages(
  "ducksemantics",
  repos = c("https://rgenomicsetl.r-universe.dev", "https://cloud.r-project.org")
)
```

Bulk relations are ordinary data frames or caller-owned DuckDB tables.
See the [semantic relation
guide](https://github.com/RGenomicsETL/ducksemantics/blob/main/docs/semantic-relations.md)
for provider protocols and ownership boundaries.

## Phenotype observations cite their text

An accepted HPO observation names the source document, the exact
zero-based half-open span, its context and method provenance.

``` r
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

accepted <- ducksemantics_hpo_observations(documents, observations)
accepted[c("document_id", "hpo_id", "source_text", "context_status")]
#>   document_id     hpo_id source_text context_status
#> 1    case-001 HP:0001250    seizures        present
```

Changing the asserted text while keeping the offsets is a contract
failure, not a second interpretation of the document.

``` r
invalid <- observations
invalid$source_text <- "seizure disorder"
span_error <- tryCatch(
  ducksemantics_hpo_observations(documents, invalid),
  error = identity
)
stopifnot(inherits(span_error, "error"))
conditionMessage(span_error)
#> [1] "HPO observation `source_text` must exactly equal its source-document span."
```

The same validator distinguishes `present`, `absent/negated`,
`family_history`, `uncertain`, `conflict` and `unsupported`. Family
history is retained as context; it is never silently converted to a
proband-present phenotype.

## Monarch facts belong to a release

The following miniature catalog is synthetic and exists only to
demonstrate the release contract. Facts cannot be queried as `latest`:
callers select an exact release or a typed date/source-order cutoff.

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
  gene_disease,
  releases,
  relation = "gene_disease"
)
```

The same relation produces different visible facts at two cutoffs. The
selected release is attached to each result as an explicit attribute.

``` r
early <- ducksemantics_monarch_query(
  gene_disease,
  releases,
  relation = "gene_disease",
  provider_id = "Monarch",
  as_of = as.Date("2024-06-01")
)
late <- ducksemantics_monarch_query(
  gene_disease,
  releases,
  relation = "gene_disease",
  provider_id = "Monarch",
  as_of = as.Date("2025-06-01")
)

selection <- data.frame(
  as_of = as.Date(c("2024-06-01", "2025-06-01")),
  release_id = c(attr(early, "release_selected"), attr(late, "release_selected")),
  disease_id = c(early$disease_id, late$disease_id),
  stringsAsFactors = FALSE
)
stopifnot(selection$release_id == c("2024-01", "2025-01"))
selection
#>        as_of release_id    disease_id
#> 1 2024-06-01    2024-01 MONDO:0000001
#> 2 2025-06-01    2025-01 MONDO:0000002
```

A historical holdout is an anti-join between two exact releases, with
both release identities preserved.

``` r
holdout <- ducksemantics_monarch_gene_disease_holdout_audit(
  gene_disease,
  releases,
  provider_id = "Monarch",
  train_release_id = "2024-01",
  holdout_release_id = "2025-01"
)
holdout[c("gene_id", "disease_id", "train_release_id", "holdout_release_id")]
#>     gene_id    disease_id train_release_id holdout_release_id
#> 1 HGNC:1100 MONDO:0000002          2024-01            2025-01
```

Official dated Monarch DuckDB packs remain caller-owned. The package can
project a verified, already-attached pack into temporary views; it does
not download, copy or merge provider edges. See
`?ducksemantics_monarch_project_pack` for the pack contract.

## Literature is selected as of a snapshot

RClinVarbitration supplies append-only article events and sections.
`ducksemantics_literature_retrieve()` selects the highest source ordinal
visible at the requested snapshot, then joins sections on the complete
article-version key.

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
```

The same query resolves to the article version current at each snapshot.

``` r
retrieve <- function(snapshot_id) {
  ducksemantics_literature_retrieve(
    articles,
    sections,
    snapshots,
    query = "seizure",
    provider_id = "pubmed",
    snapshot_id = snapshot_id
  )
}
hits <- rbind(retrieve("baseline"), retrieve("update"))
stopifnot(hits$version_id == c("source-9", "source-10"))
hits[c(
  "snapshot_id", "version_id", "source_ordinal", "section",
  "start_offset", "end_offset", "source_text"
)]
#>   snapshot_id version_id source_ordinal    section start_offset end_offset
#> 1    baseline   source-9              9 BACKGROUND            0          7
#> 2      update  source-10             10 BACKGROUND            8         15
#>   source_text
#> 1     Seizure
#> 2     seizure
```

Deleted selected versions are excluded. Optional subsection labels and
source entity ordinals remain attached to returned hits. The package
never writes a second PubMed source store.

## Where the relations go

``` text
RClinVarbitration source history
              |
              v
ducksemantics validated semantic relations
              |
              v
VariantStoryBench candidates, judgments and metrics
```

`ducksemantics` validates the handoff; it does not score the benchmark.
See the [VariantStoryBench
contract](https://github.com/RGenomicsETL/VariantStoryBench/blob/main/benchmark-contract.md)
for candidate, evaluation and grounding relations.

## Reference and verification

- The [semantic relation
  guide](https://github.com/RGenomicsETL/ducksemantics/blob/main/docs/semantic-relations.md)
  defines ownership, relation and provider boundaries.
- `?ducksemantics_hpo_observations` documents span validation.
- `?ducksemantics_monarch_query` documents exact and cutoff selection.
- `?ducksemantics_literature_retrieve` documents snapshot semantics.

Render this file with `Rscript tools/render-readme.R`. The renderer
rejects disabled example chunks. Package checks execute the same
relation contracts with offline fixtures.
