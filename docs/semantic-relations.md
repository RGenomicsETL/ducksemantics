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

| Relation or operation | Authority | DuckSemantics responsibility |
|---|---|---|
| ClinVar and publication event history | RClinVarbitration | Consume caller-owned versioned projections without copying them |
| HPO observations | DuckSemantics | Validate exact text spans, context, method, and provider provenance |
| Mondo and DisMech imports | DuckSemantics | Verify local artifacts and retain versioned source receipts |
| Monarch source packs | Monarch and the caller's localization layer | Validate a selected release and expose read-only temporary projections |
| Reviewed hypotheses and contribution artifacts | DuckSemantics | Keep proposals separate from source facts and enforce evidence links |
| Candidates, judgments, metrics, and benchmark splits | VariantStoryBench | Receive validated relations; DuckSemantics does not evaluate rankings |

Validators reject an invalid relation as a whole. They do not repair text,
select an implicit latest release, discard unsupported rows, or mutate caller
relations.

## HPO observations

`ducksemantics_hpo_observations()` accepts a document relation and an
observation relation. The document key is `document_id`; the document text is
`source_text`. Each observation contains `document_id`, `hpo_id`,
`start_offset`, `end_offset`, `source_text`, `context_status`, `method`,
`provider_id`, `provider_version`, `confidence`, and `status`.

Offsets are zero-based and half-open. The cited text must equal the exact
substring of the named document. Context remains one of `present`,
`absent/negated`, `family_history`, `uncertain`, `conflict`, or `unsupported`.
Family history is not converted to a proband-present phenotype.

## Release-cataloged Monarch relations

A Monarch release catalog contains `release_id`, `provider_id`,
`effective_date`, and `source_ordinal`. Exact release selection is preferred.
Date or source-order cutoffs select the latest cataloged release visible at the
cutoff. No API accepts `latest` as a release identity.

Imported facts retain `provider_id` and `release_id`. Queries attach the
selected release identity to their result. Historical holdouts preserve both
the training and holdout release identities and use an anti-join; they do not
reinterpret a newer fact as if it existed in the earlier release.

## Literature retrieval

`ducksemantics_literature_retrieve()` consumes caller-owned snapshot, article
event, and article-section relations. A snapshot binds `provider_id`,
`snapshot_id`, `high_water_ordinal`, and `effective_at`. Article sections join
only on the complete provider, article, version, and source-order key.

For each article, retrieval selects the highest event ordinal visible at the
chosen snapshot, excludes a selected deletion event, and returns exact section
spans. It never creates a second publication source store.

## Mondo and DisMech source imports

`ducksemantics_source_release_catalog()` retains caller-declared receipts and
POSIXct provider-availability epochs. A selected row contains the provider and
release identity plus `source_id`, `source_uri`, and `sha256`. Mondo and DisMech
imports use those receipt values by default. Callers can override them
explicitly, and the importer verifies the local file against the resulting
SHA-256 value.

Selection delegates to the existing Monarch release-selection authority with
availability as its time axis. Ingestion time, commit author date, and
version-name order are not substitutes for provider availability.

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

`ducksemantics_hypothesis()` constructs the exact hypothesis schema.
`ducksemantics_hypothesis_source()` selects one versioned source field and
constructs an exact source span. `ducksemantics_hypotheses()` validates the two
relations together.

The hypothesis relation has these columns:

| Columns | Meaning |
|---|---|
| `hypothesis_id`, `run_id`, `case_id` | Observation, run, and case identity |
| `target_type`, `candidate_id`, `proposal_type` | Proposed target and whether the row is a candidate or refinement |
| `claim_text`, `inheritance_id`, `mechanism_text` | The proposal; inheritance and mechanism are optional individually |
| `proposer_type`, `proposer_id`, `proposer_version` | Human or model provenance |
| `confidence`, `judgment` | Bounded confidence and evidence judgment |
| `missing_evidence`, `contradiction` | Required explanations for the corresponding judgment |
| `review_status`, `reviewer_id`, `reviewed_at`, `review_note` | Review state and provenance |
| `supersedes_id`, `recorded_at` | Revision lineage and observation time |

`hypothesis_id` is unique. Required identity text is nonblank. A
non-abstaining row has finite confidence in `[0, 1]` and names an inheritance or
mechanism. `insufficient_evidence` requires `missing_evidence`; `contradicts`
requires `contradiction`. Accepted, revised, and rejected rows require reviewer
identity and time. A revised row names a different observation in
`supersedes_id`. Proposed and abstained rows do not carry reviewer metadata.

The source-link relation has these columns:

| Columns | Meaning |
|---|---|
| `hypothesis_id` | Referenced hypothesis |
| `provider_id`, `release_id`, `source_id` | Versioned source receipt |
| `source_relation`, `source_record_id`, `source_field` | Relation, record, and field identity |
| `start_offset`, `end_offset`, `source_span` | Zero-based half-open exact span |
| `citation`, `evidence_role` | Citation and support, contradiction, or context role |

Provider, release, source, record, and field keys must resolve to one row in the
source-field relation. The cited text must equal the exact substring at the
recorded offsets. Extra source-field columns are accepted and ignored by the
span validator. `ducksemantics_hypothesis_source()` cites the complete field
when `text = NULL`; otherwise the fixed substring must occur exactly once.
Missing, ambiguous, or empty text is rejected.

Accepted, revised, rejected, proposed, and abstained observations remain
explicit. A revision adds a row rather than replacing its predecessor.
Abstentions use `judgment = "abstain"`, `review_status = "abstained"`, no
confidence, and no source spans. Source facts remain separate from model or
human proposals.

VariantStoryBench consumes these observations through
`bench_ducksemantics_grounding()`. The adapter maps non-abstaining rows to the
existing `judgments` and `judgment_sources` relations, retains the complete
DuckSemantics receipts, and leaves abstained candidates without judgments.
VariantStoryBench remains the evaluation authority.

`ducksemantics_contribution()` accepts only accepted or revised supporting
observations without unresolved missing evidence or contradiction. It validates
bounded payloads against pinned Mondo issue-template fields, DisMech LinkML
fields and enums, or Monarch Biolink/KGX edge fields and provenance enums. A
DisMech reference and quotation must match the same supporting source receipt.
The returned schema receipt names the pinned upstream revision and validation
scope. Upstream project validators, project review, and human submission remain
required.

## Rejection behavior

The relation APIs fail with an error before returning an output when a contract
is violated. Common failures include:

- offsets that do not reproduce the cited source text;
- missing, duplicate, or unresolved source keys;
- uncataloged provider or release identities;
- a cutoff before the first available release;
- incompatible hypothesis judgment, confidence, and review states;
- contribution evidence that does not match one supporting receipt; and
- a local source artifact whose SHA-256 does not match its receipt.

Retrieval similarity is not evidence authority. A high-scoring passage can lack
a source citation, and a lower-scoring passage can contain the supporting
receipt. Callers inspect source relations; VariantStoryBench evaluates outcomes.

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
