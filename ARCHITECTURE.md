# Architecture

**Current.** `ducksemantics` owns ontology graphs, HPO extraction
observations, release-cataloged Monarch gene/phenotype/disease
relations, including narrow read-only projections of caller-attached
dated Monarch packs, semantic and literature retrieval, source-grounded
judgments, and real model/provider protocols. Bulk values are data
frames or caller-owned DuckDB relations; S7 is reserved for prompt,
embedding, parser, and annotator provider protocols.

`RClinVarbitration` owns ClinVar and append-only PubMed/Europe PMC
source relations; ducksemantics consumes their caller-owned source-order
literature projection without writing a copy. `VariantStoryBench` owns
all evaluation and benchmark APIs.

The semantic authority is the documented relational contracts and their
validators: lexical HPO candidates are not observations; observations
validate their exact source spans; Monarch facts retain provider/release
identity through a typed catalog; and literature retrieval binds a
provider plus exact cataloged source-order cutoff, selects each
article’s latest event, drops deletion events, and joins only that
event’s sections.

## Attached Monarch annotation packs

[`ducksemantics_monarch_project_pack()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_monarch_project_pack.md)
accepts only an already attached, read-only named DuckDB Monarch catalog
and an exact provider/release row from the typed catalog. That row is a
caller-owned binding, not a computed file hash; the localization layer
remains responsible for verifying the immutable pack receipt before
attachment. It never attaches a URL, accepts `latest`, downloads,
collects, copies, rewrites, or coalesces the pack. Before creating any
view it checks the official base tables and the required official
projection columns/types, including Monarch’s serialized text `negated`
field. It then creates only connection-local temporary views: a native
association view, three total role projections, and nodes,
node-has-phenotype, and information-content views.

The native association view keeps the source edge rows and all source
fields, including list provenance, qualifiers, raw originals, taxa,
predicate, association category, and direction. Role views add
normalized `gene_*`/`disease_*`/`phenotype_*` columns only when explicit
Biolink subject and object categories establish one supported
orientation. They retain the raw source subject/object and expose
`association_status`; malformed values, missing predicates, negated
edges, and unsupported orientation are never silently promoted to
support. A separate `negation_status` retains whether the serialized
qualifier was `not_provided`, explicitly `not_negated`, `negated`, or
`malformed`; Monarch’s omitted optional qualifier does not erase the
positive source assertion. There is no cross-provider vote, merge,
ranking, or causal default.
