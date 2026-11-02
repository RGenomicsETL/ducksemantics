
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

From a source checkout, install the package with its bundled stable-ABI
extension:

``` sh
make r-install
```

Published builds are also available from R-universe:

``` r
install.packages(
  "ducksemantics",
  repos = c("https://rgenomicsetl.r-universe.dev", "https://cloud.r-project.org")
)
```

Bulk relations are ordinary data frames or caller-owned DuckDB tables.
See the [semantic relation guide](../../docs/semantic-relations.md) for
provider protocols and ownership boundaries.

## Phenotype observations cite their text

An accepted HPO observation names the source document, the exact
zero-based half-open span, its context and method provenance.

``` r
documents <- fixture_csv("readme-hpo-documents.csv")
observations <- fixture_csv(
  "readme-hpo-observations.csv", colClasses = c(provider_version = "character")
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
conditionMessage(span_error)
#> [1] "HPO observation `source_text` must exactly equal its source-document span."
```

The same validator distinguishes `present`, `absent/negated`,
`family_history`, `uncertain`, `conflict` and `unsupported`. Family
history is retained as context; it is never silently converted to a
proband-present phenotype.

## Monarch facts belong to a release

The packaged synthetic catalog demonstrates the release contract. Facts
cannot be queried as `latest`: callers select an exact release or a
typed date/source-order cutoff.

``` r
releases <- fixture_csv("readme-monarch-releases.csv")
releases$effective_date <- as.Date(releases$effective_date)
gene_disease <- fixture_csv("readme-monarch-gene-disease.csv")

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
snapshots <- fixture_csv("readme-literature-snapshots.csv")
snapshots$effective_at <- as.POSIXct(
  snapshots$effective_at, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"
)
articles <- fixture_csv(
  "readme-literature-articles.csv", colClasses = c(pmid = "character")
)
sections <- fixture_csv(
  "readme-literature-sections.csv", colClasses = c(pmid = "character")
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

## Reviewed hypotheses retain exact source spans

Mondo and DisMech imports use a selected release row as the receipt
authority. The caller can override receipt fields, but the importer
verifies the file against the resulting SHA-256 value.

``` r
source_releases <- fixture_csv("source-releases.csv")
source_releases$available_at <- as.POSIXct(
  source_releases$available_at, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"
)
dismech_release <- ducksemantics_source_release_select(
  source_releases,
  provider_id = "DisMech",
  release_id = "fac61860115b1fdb194de4f4cacef3a8f6a398af"
)
dismech <- ducksemantics_dismech_import(
  fixture_path("dismech-marfan.yaml"),
  dismech_release
)
dismech$artifact[c("provider_id", "release_id", "source_id", "sha256")]
#>   provider_id                               release_id
#> 2     DisMech fac61860115b1fdb194de4f4cacef3a8f6a398af
#>                           source_id
#> 2 kb/disorders/Marfan_Syndrome.yaml
#>                                                             sha256
#> 2 9c098d1020e99b9c3ab6b9c28c2c4a884a0a122625e7f9385c5d6307e6924b29
```

A hypothesis is a reviewed observation. Its source link is an exact
subspan of one versioned field, with zero-based half-open offsets.

``` r
hypothesis <- ducksemantics_hypothesis(
  hypothesis_id = "walkthrough-h1",
  run_id = "walkthrough-run",
  case_id = "walkthrough-case",
  target_type = "gene",
  candidate_id = "HGNC:3603",
  proposal_type = "candidate",
  claim_text = "Defects in FBN1 can cause Marfan syndrome.",
  mechanism_text = "FBN1 defects cause Marfan syndrome",
  proposer_type = "human",
  proposer_id = "walkthrough-author",
  proposer_version = "1",
  confidence = 0.9,
  judgment = "supports",
  review_status = "accepted",
  reviewer_id = "walkthrough-reviewer",
  reviewed_at = as.POSIXct("2026-10-15", tz = "UTC"),
  review_note = "The quoted source text supports the causal claim.",
  recorded_at = as.POSIXct("2026-10-14", tz = "UTC")
)
source <- ducksemantics_hypothesis_source(
  hypothesis_id = hypothesis$hypothesis_id,
  source_fields = dismech$fields,
  source_relation = "dismech_fields",
  field_path = "/pathophysiology/0/evidence/0/snippet",
  text = paste(
    "The gene encoding for fibrillin, the FBN1 gene on chromosome 15,",
    "has been identified as the defective gene causing the Marfan syndrome."
  ),
  citation = "PMID:8180508"
)
bundle <- ducksemantics_hypotheses(hypothesis, source, dismech$fields)
bundle$hypotheses[c(
  "hypothesis_id", "candidate_id", "claim_text", "review_status"
)]
#>    hypothesis_id candidate_id                                 claim_text
#> 1 walkthrough-h1    HGNC:3603 Defects in FBN1 can cause Marfan syndrome.
#>   review_status
#> 1      accepted
bundle$sources[c(
  "source_field", "start_offset", "end_offset", "source_span", "citation"
)]
#>                            source_field start_offset end_offset
#> 1 /pathophysiology/0/evidence/0/snippet            0        135
#>                                                                                                                               source_span
#> 1 The gene encoding for fibrillin, the FBN1 gene on chromosome 15, has been identified as the defective gene causing the Marfan syndrome.
#>       citation
#> 1 PMID:8180508
```

Review state and evidence judgment remain separate:

| Review state | Example                                             | Required record                                                                     |
|--------------|-----------------------------------------------------|-------------------------------------------------------------------------------------|
| `proposed`   | An unreviewed FBN1 claim                            | Proposer provenance and exact source spans                                          |
| `accepted`   | The causal claim above                              | Reviewer identity and time, plus supporting source spans                            |
| `revised`    | A narrower claim about pathogenic FBN1 defects      | A new row naming its predecessor in `supersedes_id`                                 |
| `rejected`   | A claim that FBN1 explains every observed phenotype | Review provenance; an `insufficient_evidence` judgment also records what is missing |
| `abstained`  | No directional claim about an additional mechanism  | No confidence or source spans                                                       |

The cited passage does not establish a specific downstream signaling
mechanism. An abstention records that limit without inventing a
confidence value:

``` r
abstention <- ducksemantics_hypothesis(
  hypothesis_id = "state-abstained",
  run_id = "walkthrough-run",
  case_id = "walkthrough-case",
  target_type = "gene",
  candidate_id = "HGNC:3603",
  proposal_type = "refinement",
  claim_text = "The cited passage does not establish a downstream signaling mechanism.",
  proposer_type = "human",
  proposer_id = "walkthrough-author",
  proposer_version = "1",
  judgment = "abstain",
  review_status = "abstained",
  recorded_at = as.POSIXct("2026-10-18", tz = "UTC")
)
abstention[c("hypothesis_id", "judgment", "review_status", "confidence")]
#>     hypothesis_id judgment review_status confidence
#> 1 state-abstained  abstain     abstained         NA
```

Only accepted or revised hypotheses can become contribution artifacts.
External schema validation, review, and submission remain
human-controlled steps.

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

- The [semantic relation guide](../../docs/semantic-relations.md)
  defines ownership, relation and provider boundaries.
- `?ducksemantics_hpo_observations` documents span validation.
- `?ducksemantics_monarch_query` documents exact and cutoff selection.
- `?ducksemantics_literature_retrieve` documents snapshot semantics.

Render this file with `Rscript tools/render-readme.R`. The renderer
rejects disabled example chunks. Package checks execute the same
relation contracts with offline fixtures.
