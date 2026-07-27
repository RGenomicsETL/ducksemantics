# Project one attached, dated Monarch annotation pack

Projects an already attached read-only official Monarch DuckDB pack into
connection-local temporary views. It never attaches, downloads,
collects, copies, rewrites, or merges pack rows. `source_catalog` must
name an attached DuckDB catalog whose `main` schema contains the
official `denormalized_edges`, `nodes`, `information_content`, and
`node_has_phenotype` tables. The edge table is checked before any view
is made, including its scalar role categories, provider-serialized text
negation, and array provenance fields. The official required `nodes`,
`node_has_phenotype`, and `information_content` projection columns/types
are also validated.

## Usage

``` r
ducksemantics_monarch_project_pack(
  conn,
  source_catalog,
  releases,
  provider_id,
  release_id,
  prefix = "semantic_monarch"
)
```

## Arguments

- conn:

  Valid DBI connection with the pack already attached.

- source_catalog:

  Attached DuckDB catalog name, not the current catalog.

- releases:

  Typed Monarch release catalog or caller-owned catalog table. This
  binds provider/release semantics; the caller must verify the immutable
  file receipt before attaching the pack.

- provider_id:

  Exact provider identity in `releases`.

- release_id:

  Exact, dated release identity in `releases`; `"latest"` is rejected.

- prefix:

  Safe unqualified prefix for temporary view names.

## Value

A data frame naming the seven connection-local temporary views, with
`relation`, `view_name`, `provider_id`, and `release_id` columns. It
does not return pack contents.

## Details

`associations` retains every source edge and all source columns exactly,
then adds the selected provider and release. Each role view is likewise
a total projection: it retains raw source columns and adds normalized
role columns, `source_subject`, `source_object`, `source_direction`, and
`association_status`. Only `supported_subject_to_object` and
`supported_object_to_subject` rows have non-missing normalized roles.
`missing_endpoint`, `missing_predicate`, `malformed_negation`,
`missing_role_category`, `malformed_role_category`, `negated`, and
`unsupported_orientation` are not support. `negation_status` separately
distinguishes `not_provided`, `not_negated`, `negated`, and `malformed`:
Monarch omits the optional qualifier on many positive assertion rows, so
omission is retained but does not erase the assertion. In particular, a
missing predicate never becomes a causal predicate.
