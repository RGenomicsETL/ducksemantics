# Semantic relation contracts

`ducksemantics` owns ontology graphs, HPO extraction observations,
release-cataloged Monarch gene/phenotype/disease relations, including narrow
read-only projections of caller-attached dated Monarch packs, semantic and
literature retrieval, source-grounded judgments, reviewed hypothesis
observations, and real model/provider protocols. Bulk values are data frames or caller-owned DuckDB relations; S7 is
reserved for prompt, embedding, parser, and annotator provider protocols.

`RClinVarbitration` owns ClinVar and append-only PubMed/Europe PMC source
relations; ducksemantics consumes their caller-owned source-order literature
projection without writing a copy.
`VariantStoryBench` owns all evaluation and benchmark APIs.

The semantic authority is the documented relational contracts and their
validators: lexical HPO candidates are not observations; observations validate
their exact source spans; Monarch facts retain provider/release identity through
a typed catalog; and literature retrieval binds a provider plus exact cataloged
source-order cutoff, selects each article's latest event, drops deletion events,
and joins only that event's sections.

## Mondo and DisMech source imports

`ducksemantics_source_release_catalog()` retains caller-declared receipts and
POSIXct provider-availability epochs. Selection delegates to the existing
Monarch release-selection authority with availability as its time axis.
Ingestion time, commit author date, and version-name order are not substitutes
for provider availability.

Source import verifies each local artifact against its declared SHA-256.
Mondo imports retain obsolete terms, aliases, raw xrefs, explicit SKOS mappings,
replacement/consider tags, and stanza-relative source-line locators. Imported
non-Mondo terms are excluded from the disease projection; links to external
identities remain visible. These release-bound relations do not overwrite the
unversioned lexical graph.

DisMech imports retain the complete YAML record and scalar fields at JSON-pointer
locations. The record path and field locator are source keys, not invented
upstream evidence accessions. Imports preserve source claims and snippets;
they do not independently verify cited publications.

## Reviewed hypothesis observations

`ducksemantics_hypotheses()` validates caller-supplied hypothesis and source-link
relations. A non-abstaining observation records its case and candidate, proposal
type, claim, inheritance or mechanism, proposer identity and version,
confidence, evidence judgment, missing or contradictory evidence, and review
outcome. Accepted, revised, rejected, proposed, and abstained observations remain
explicit. A revised row names the observation it supersedes rather than
replacing it.

Each non-abstaining observation cites an exact zero-based half-open span in a
versioned source field. Provider, release, source, record, and field keys must
resolve to that field, and the cited text must match it exactly. Abstentions do
not carry confidence or source spans. Source facts remain separate from model or
human proposals.

VariantStoryBench consumes these observations through
`bench_ducksemantics_grounding()`. The adapter maps non-abstaining rows to the
existing `judgments` and `judgment_sources` relations, retains the complete
DuckSemantics receipts, and leaves abstained candidates without judgments.
VariantStoryBench remains the evaluation authority.

`ducksemantics_contribution()` accepts only reviewed, supporting observations
without unresolved missing evidence or contradiction. It validates bounded
payloads against pinned Mondo issue-template fields, DisMech LinkML fields and
enums, or Monarch Biolink/KGX edge fields and provenance enums. The returned
schema receipt names the pinned upstream revision and validation scope. Upstream
project validators and human submission remain required.

## Attached Monarch annotation packs

`ducksemantics_monarch_project_pack()` accepts only an already attached,
read-only named DuckDB Monarch catalog and an exact provider/release row from
the typed catalog. That row is a caller-owned binding, not a computed file hash;
the localization layer remains responsible for verifying the immutable pack
receipt before attachment.
It never attaches a URL, accepts `latest`, downloads, collects, copies, rewrites,
or coalesces the pack. Before creating any view it checks the official base
tables and the required official projection columns/types, including Monarch's serialized
text `negated` field. It then creates only connection-local temporary views: a
native association view, three total role projections, and nodes,
node-has-phenotype, and information-content views.

The native association view keeps the source edge rows and all source fields,
including list provenance, qualifiers, raw originals, taxa, predicate,
association category, and direction. Role views add normalized
`gene_*`/`disease_*`/`phenotype_*` columns only when explicit Biolink subject
and object categories establish one supported orientation. They retain the raw
source subject/object and expose `association_status`; malformed values,
missing predicates, negated edges, and unsupported orientation are never
silently promoted to support. A separate `negation_status` retains whether the
serialized qualifier was `not_provided`, explicitly `not_negated`, `negated`, or
`malformed`; Monarch's omitted optional qualifier does not erase the positive
source assertion. There is no cross-provider vote, merge, ranking, or causal
default.
