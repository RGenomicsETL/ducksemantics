# Retrieve exact literature spans from append-only caller-owned relations

`RClinVarbitration` owns the article and section relations. Retrieval
binds a provider and exact cataloged source-order cutoff, selects the
maximum article `source_ordinal` at or below that cutoff per
provider/article, drops a latest deletion event, and joins only sections
from that exact version. It never imports, writes, or fetches
literature.

## Usage

``` r
ducksemantics_literature_retrieve(
  articles,
  sections,
  snapshots,
  query,
  provider_id,
  snapshot_id,
  section_names = NULL,
  conn = NULL
)
```

## Arguments

- articles:

  Caller-owned relation with `provider_id`, `article_id`, `pmid`,
  `version_id`, finite integer `source_ordinal`, and logical
  `is_deleted`.

- sections:

  Caller-owned relation with `provider_id`, `article_id`, `pmid`,
  `version_id`, finite integer `source_ordinal`, `section`, and exact
  non-empty `text`.

- snapshots:

  Source snapshot catalog from
  [`ducksemantics_literature_snapshot_catalog()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_literature_snapshot_catalog.md).

- query:

  Non-empty lexical query text.

- provider_id:

  Provider identity.

- snapshot_id:

  Exact provider snapshot identity.

- section_names:

  Optional non-empty section names to search; `NULL` searches every
  version-bound section.

- conn:

  Optional DBI connection for table relations.

## Value

Exact non-empty source spans from non-deleted article versions visible
at the cataloged source-order cutoff.

## Details

A read-only RClinVarbitration projection joins `pubmed_sources` to
`pubmed_articles` and `pubmed_abstracts`: use source provider as
`provider_id`, source id as `version_id`/`snapshot_id`, source ordinal
for both ordinal fields, and `source_applied_at` as `effective_at`. A
section must retain the exact article event's version id and source
ordinal.
