
<!-- README.md is generated from README.Rmd by `make readme`. Edit README.Rmd. -->

# DuckSemantics

DuckSemantics lets DuckDB read local GGUF model files, tokenize and
embed text, and score source passages in the same database as the
versioned records those passages came from. The extension and its
supported model kernels are pure C11 and use DuckDB’s stable C extension
ABI. You provide model weights as a local file; the package does not
bundle or download them.

The purpose of that design is traceability. A similarity score tells you
which passage to inspect first, not whether the passage is true. Mondo
and DisMech imports therefore keep their provider, release and SHA-256
receipt. Scores, hypotheses and reviewed contribution artifacts remain
separate relations that point back to exact source rows and spans.

This page is one executed session against a real EmbeddingGemma-300M
model and verbatim upstream fixtures. The render requires the model
named by `DUCKSEMANTICS_EMBED_MODEL`; no result is precomputed.

## What this session demonstrates

| Property                                                     | Executed result                                    |
|--------------------------------------------------------------|----------------------------------------------------|
| A caller-supplied GGUF runs through the pure-C extension     | model metadata, tokenization and embeddings below  |
| Imported source rows are release-bound and byte-verified     | Mondo and DisMech receipts, plus a rejected digest |
| Retrieval does not manufacture evidence                      | ranked mechanisms joined to their own citations    |
| A reviewed hypothesis cites exact source text                | hypothesis and source-span validation              |
| Contribution output is reviewable, not automatic publication | pinned DisMech schema receipt                      |

## Install and connect

From a repository checkout, install the package and its bundled
extension:

``` sh
make r-bootstrap
R CMD INSTALL r/ducksemantics
```

The same canonical C source list builds the standalone extension with
`make` or CMake. Model weights remain outside the package.

``` r
library(ducksemantics)
conn <- ducksemantics_connect()
```

``` sql
SELECT extension_name, loaded
FROM duckdb_extensions()
WHERE extension_name = 'ducksemantics';
```

<div class="knitsql-table">

| extension_name | loaded |
|:---------------|:-------|
| ducksemantics  | TRUE   |

</div>

## 1. Bring your own model

Set `DUCKSEMANTICS_EMBED_MODEL` to an EmbeddingGemma-300M GGUF file. The
render fails here instead of silently replacing the model example with
fixture output.

``` r
embed_model <- normalizePath(
  Sys.getenv("DUCKSEMANTICS_EMBED_MODEL"),
  mustWork = TRUE
)
embed_model_sha256 <- digest::digest(file = embed_model, algo = "sha256")
basename(embed_model)
#> [1] "embeddinggemma-300M-Q8_0.gguf"
embed_model_sha256
#> [1] "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"
```

The architecture key selects the native kernel. Width and block count
are read from the file, not hard-coded from its filename.

``` sql
SELECT key, value
FROM gguf_metadata(?embed_model)
WHERE key IN ('general.architecture', 'general.name',
              'gemma-embedding.embedding_length',
              'gemma-embedding.block_count', 'tokenizer.ggml.model')
ORDER BY key;
```

## 2. Tokens and vectors

Tokenization uses the vocabulary stored in the same GGUF file. Gemma
control tokens are omitted during decoding, so the input text
round-trips.

``` sql
WITH encoded AS (
  SELECT semantic_tokenize(
    ?embed_model, 'Marfan syndrome aortic dilation', true
  ) AS ids
),
pieces AS (
  SELECT generate_subscripts(ids, 1) AS position, unnest(ids) AS id
  FROM encoded
)
SELECT position, id, vocab.value AS piece
FROM pieces
JOIN gguf_metadata_array(?embed_model, 'tokenizer.ggml.tokens') AS vocab
  ON vocab.array_index = pieces.id
ORDER BY position;
```

<div class="knitsql-table">

| position |    id | piece     |
|:---------|------:|:----------|
| 1        |     2 | <bos>     |
| 2        |  9585 | Mar       |
| 3        | 23935 | fan       |
| 4        | 22595 | ▁syndrome |
| 5        | 80838 | ▁aortic   |
| 6        | 58692 | ▁dilation |
| 7        |     1 | <eos>     |

</div>

``` sql
SELECT semantic_detokenize(
  ?embed_model,
  semantic_tokenize(?embed_model, 'Marfan syndrome aortic dilation', true)
) AS text;
```

`semantic_embed(path, text, task, dimensions, normalize)` runs the
native EmbeddingGemma forward pass. Query and document task prefixes
remain explicit.

``` r
embed_task <- "retrieval_query"
embed_dimensions <- 256L
```

``` sql
CREATE TABLE short_texts AS
SELECT text_id, text,
       semantic_embed(
         ?embed_model, text, ?embed_task, ?embed_dimensions, true
       ) AS embedding
FROM (VALUES
  ('marfan',    'Marfan syndrome'),
  ('fibrillin', 'defective fibrillin-1 microfibrils'),
  ('diabetes',  'insulin resistance in type 2 diabetes')
) AS t(text_id, text);
```

``` sql
SELECT text_id,
       typeof(embedding) AS type,
       len(embedding) AS dimensions,
       sqrt(list_dot_product(embedding, embedding)) AS norm,
       list_value(
         round(embedding[1], 4),
         round(embedding[2], 4),
         round(embedding[3], 4),
         round(embedding[4], 4)
       )::VARCHAR AS first_values
FROM short_texts
ORDER BY text_id;
```

These vectors are local model outputs. Their dot products are
similarities, not calibrated probabilities.

## 3. Load source records at a date

The bundled Mondo fixture contains complete term stanzas from release
`v2026-10-06`. The DisMech fixture is the verbatim Marfan syndrome
record at a pinned commit. A catalog selects releases by provider
availability time, and each importer verifies the input bytes before
parsing them.

``` r
fixture <- function(name) {
  system.file("extdata", name, package = "ducksemantics", mustWork = TRUE)
}
receipts <- read.csv(fixture("source-releases.csv"), stringsAsFactors = FALSE)
receipts$available_at <- as.POSIXct(
  receipts$available_at, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"
)
catalog <- ducksemantics_source_release_catalog(receipts)

mondo_release <- ducksemantics_source_release_select(
  catalog, "Mondo", as_of = as.POSIXct("2026-10-07", tz = "UTC")
)
mondo <- ducksemantics_mondo_import(
  fixture("mondo-subset.obo"), mondo_release
)

dismech_release <- ducksemantics_source_release_select(
  catalog, "DisMech",
  release_id = "fac61860115b1fdb194de4f4cacef3a8f6a398af"
)
dismech <- ducksemantics_dismech_import(
  fixture("dismech-marfan.yaml"), dismech_release
)

DBI::dbWriteTable(conn, "mondo_terms", mondo$terms)
DBI::dbWriteTable(conn, "mondo_obsolescence", mondo$obsolescence)
DBI::dbWriteTable(conn, "dismech_fields", dismech$fields)
```

``` sql
SELECT provider_id, release_id, receipt_id, count(*) AS rows
FROM (
  SELECT provider_id, release_id, receipt_id FROM mondo_terms
  UNION ALL
  SELECT provider_id, release_id, receipt_id FROM dismech_fields
)
GROUP BY ALL
ORDER BY provider_id;
```

<div class="knitsql-table">

| provider_id | release_id                               | receipt_id               | rows |
|:------------|:-----------------------------------------|:-------------------------|-----:|
| DisMech     | fac61860115b1fdb194de4f4cacef3a8f6a398af | dismech-fac61860-marfan  | 1412 |
| Mondo       | v2026-10-06                              | mondo-v2026-10-06-subset |    3 |

</div>

The Mondo relation keeps obsolescence explicit instead of silently
replacing an identifier.

``` sql
SELECT t.node_id, t.label, t.is_obsolete, o.value AS replaced_by
FROM mondo_terms t
LEFT JOIN mondo_obsolescence o
  ON o.node_id = t.node_id AND o.tag = 'replaced_by'
ORDER BY t.node_id;
```

<div class="knitsql-table">

| node_id       | label                       | is_obsolete | replaced_by   |
|:--------------|:----------------------------|:------------|:--------------|
| MONDO:0000002 | obsolete 46,XX sex reversal | TRUE        | MONDO:0009299 |
| MONDO:0007947 | Marfan syndrome             | FALSE       | NA            |
| MONDO:0009299 | 46 XX gonadal dysgenesis    | FALSE       | NA            |

</div>

A wrong digest is rejected before the parser can emit rows.

``` r
tamper_error <- tryCatch(
  ducksemantics_mondo_import(
    fixture("mondo-subset.obo"), mondo_release,
    sha256 = paste(rep("0", 64L), collapse = "")
  ),
  error = identity
)
conditionMessage(tamper_error)
#> [1] "Source artifact SHA-256 does not match its receipt."
```

## 4. Ask a question

The DisMech record contains six pathophysiology descriptions. Document
vectors use `retrieval_document`; the question uses `retrieval_query`.
The vector and score relations retain the source row keys and model
digest.

``` sql
CREATE TABLE mechanisms AS
SELECT regexp_extract(d.field_path, '^(/pathophysiology/[0-9]+)/', 1) AS field_prefix,
       n.value AS mechanism,
       d.value AS description,
       d.provider_id, d.release_id, d.source_id, d.receipt_id, d.source_row_id
FROM dismech_fields d
JOIN dismech_fields n
  ON n.provider_id = d.provider_id AND n.release_id = d.release_id
 AND n.source_id = d.source_id
 AND n.field_path = regexp_replace(d.field_path, '/description$', '/name')
WHERE regexp_matches(d.field_path, '^/pathophysiology/[0-9]+/description$');
```

``` sql
CREATE TABLE mechanism_vectors AS
SELECT provider_id, release_id, source_id, source_row_id,
       ?embed_model_sha256 AS model_sha256,
       'retrieval_document' AS task,
       ?embed_dimensions AS dimensions,
       true AS normalized,
       semantic_embed(
         ?embed_model, description, 'retrieval_document',
         ?embed_dimensions, true
       ) AS embedding
FROM mechanisms;
```

``` r
question <- "Why does the aortic wall weaken and dilate?"
```

``` sql
CREATE TABLE mechanism_scores AS
WITH query AS (
  SELECT semantic_embed(
    ?embed_model, ?question, ?embed_task, ?embed_dimensions, true
  ) AS vector
)
SELECT ?question AS question, ?embed_task AS query_task,
       v.provider_id, v.release_id, v.source_id, v.source_row_id,
       v.model_sha256, v.dimensions, v.normalized,
       list_dot_product(query.vector, v.embedding) AS cosine
FROM mechanism_vectors v, query;
```

``` sql
SELECT m.mechanism, round(s.cosine, 4) AS cosine
FROM mechanism_scores s
JOIN mechanisms m USING (provider_id, release_id, source_id, source_row_id)
ORDER BY s.cosine DESC;
```

<div class="knitsql-table">

| mechanism                       | cosine |
|:--------------------------------|-------:|
| Extracellular Matrix Remodeling | 0.7187 |
| Vascular Inflammation           | 0.4436 |
| Impaired Mechanotransduction    | 0.4220 |
| Mitochondrial Dysfunction       | 0.4121 |
| Dysregulated TGF-beta Signaling | 0.3600 |
| FBN1 Gene Mutation              | 0.2979 |

</div>

## 5. A score is not evidence

Citation fields are joined from the source record after ranking.
Retrieval cannot add a citation that the source does not contain.

``` sql
CREATE TABLE mechanism_evidence AS
SELECT m.provider_id, m.release_id, m.source_id, m.source_row_id,
       m.mechanism, m.field_prefix, e.field_path, e.value AS reference
FROM mechanisms m
JOIN dismech_fields e
  ON e.provider_id = m.provider_id AND e.release_id = m.release_id
 AND e.source_id = m.source_id
 AND regexp_matches(
   e.field_path,
   '^' || m.field_prefix || '/evidence/[0-9]+/reference$'
 );
```

``` sql
SELECT m.mechanism, round(s.cosine, 4) AS cosine,
       count(e.reference) AS cited_references
FROM mechanism_scores s
JOIN mechanisms m USING (provider_id, release_id, source_id, source_row_id)
LEFT JOIN mechanism_evidence e
  USING (provider_id, release_id, source_id, source_row_id)
GROUP BY m.mechanism, s.cosine
ORDER BY s.cosine DESC;
```

The highest-ranked passage, Extracellular Matrix Remodeling, has no
cited reference in this source record. The score is useful for
navigation and invalid as a substitute for evidence.

The best-ranked mechanism that does carry citations retains the source
paths and reference identifiers:

``` sql
WITH best AS (
  SELECT m.provider_id, m.release_id, m.source_id, m.source_row_id
  FROM mechanism_scores s
  JOIN mechanisms m USING (provider_id, release_id, source_id, source_row_id)
  WHERE EXISTS (
    SELECT 1
    FROM mechanism_evidence e
    WHERE e.provider_id = m.provider_id
      AND e.release_id = m.release_id
      AND e.source_id = m.source_id
      AND e.source_row_id = m.source_row_id
  )
  ORDER BY s.cosine DESC
  LIMIT 1
)
SELECT e.mechanism, e.reference, support.value AS supports, e.field_path
FROM mechanism_evidence e
JOIN best USING (provider_id, release_id, source_id, source_row_id)
LEFT JOIN dismech_fields support
  ON support.provider_id = e.provider_id
 AND support.release_id = e.release_id
 AND support.source_id = e.source_id
 AND support.field_path = regexp_replace(e.field_path, '/reference$', '/supports')
ORDER BY e.field_path;
```

<div class="knitsql-table">

| mechanism          | reference     | supports | field_path                              |
|:-------------------|:--------------|:---------|:----------------------------------------|
| FBN1 Gene Mutation | PMID:8180508  | SUPPORT  | /pathophysiology/0/evidence/0/reference |
| FBN1 Gene Mutation | PMID:39096853 | SUPPORT  | /pathophysiology/0/evidence/1/reference |
| FBN1 Gene Mutation | PMID:17657824 | SUPPORT  | /pathophysiology/0/evidence/2/reference |
| FBN1 Gene Mutation | PMID:12938084 | SUPPORT  | /pathophysiology/0/evidence/3/reference |
| FBN1 Gene Mutation | PMID:24555249 | SUPPORT  | /pathophysiology/0/evidence/4/reference |
| FBN1 Gene Mutation | PMID:36004597 | SUPPORT  | /pathophysiology/0/evidence/5/reference |
| FBN1 Gene Mutation | PMID:25729264 | SUPPORT  | /pathophysiology/0/evidence/6/reference |
| FBN1 Gene Mutation | PMID:42350581 | SUPPORT  | /pathophysiology/0/evidence/7/reference |

</div>

The DisMech disease identifier also joins to a separately versioned
Mondo row. Both receipts remain visible.

``` sql
SELECT d.value AS disease_id,
       t.label AS mondo_label,
       d.release_id AS dismech_release,
       t.release_id AS mondo_release,
       d.receipt_id AS dismech_receipt,
       t.receipt_id AS mondo_receipt
FROM dismech_fields d
JOIN mondo_terms t ON t.node_id = d.value
WHERE d.field_path = '/disease_term/term/id';
```

<div class="knitsql-table">

| disease_id    | mondo_label     | dismech_release                          | mondo_release | dismech_receipt         | mondo_receipt            |
|:--------------|:----------------|:-----------------------------------------|:--------------|:------------------------|:-------------------------|
| MONDO:0007947 | Marfan syndrome | fac61860115b1fdb194de4f4cacef3a8f6a398af | v2026-10-06   | dismech-fac61860-marfan | mondo-v2026-10-06-subset |

</div>

## 6. Record a reviewed hypothesis

A hypothesis is an observation, not a source fact. This example records
a human-authored causal claim, its review state, and one exact source
subspan. The claim paraphrases the source. Inheritance is left
unspecified because the cited text does not establish it.

``` r
hypotheses <- ducksemantics_hypothesis(
  hypothesis_id = "readme-hypothesis-1",
  run_id = "readme-run-1",
  case_id = "readme-case-1",
  target_type = "gene",
  candidate_id = "HGNC:3603",
  proposal_type = "candidate",
  claim_text = "Defects in FBN1 can cause Marfan syndrome.",
  mechanism_text = "FBN1 defects cause Marfan syndrome",
  proposer_type = "human",
  proposer_id = "readme-author",
  proposer_version = "1",
  confidence = 0.9,
  judgment = "supports",
  review_status = "accepted",
  reviewer_id = "readme-reviewer",
  reviewed_at = as.POSIXct("2026-10-15", tz = "UTC"),
  review_note = "The quoted source text supports the causal claim.",
  recorded_at = as.POSIXct("2026-10-14", tz = "UTC")
)

hypothesis_sources <- ducksemantics_hypothesis_source(
  hypothesis_id = "readme-hypothesis-1",
  source_fields = dismech$fields,
  source_relation = "dismech_fields",
  field_path = "/pathophysiology/0/evidence/0/snippet",
  text = paste(
    "The gene encoding for fibrillin, the FBN1 gene on chromosome 15,",
    "has been identified as the defective gene causing the Marfan syndrome."
  ),
  citation = "PMID:8180508"
)

hypothesis_bundle <- ducksemantics_hypotheses(
  hypotheses,
  hypothesis_sources,
  dismech$fields
)
hypothesis_bundle$hypotheses[c(
  "hypothesis_id", "candidate_id", "claim_text", "judgment", "review_status"
)]
#>         hypothesis_id candidate_id                                 claim_text
#> 1 readme-hypothesis-1    HGNC:3603 Defects in FBN1 can cause Marfan syndrome.
#>   judgment review_status
#> 1 supports      accepted
hypothesis_bundle$sources[c(
  "source_field", "start_offset", "end_offset", "source_span", "citation"
)]
#>                            source_field start_offset end_offset
#> 1 /pathophysiology/0/evidence/0/snippet            0        135
#>                                                                                                                               source_span
#> 1 The gene encoding for fibrillin, the FBN1 gene on chromosome 15, has been identified as the defective gene causing the Marfan syndrome.
#>       citation
#> 1 PMID:8180508
```

The validator resolves the provider, release, source, record and field
keys, then checks the zero-based half-open span against the versioned
field text. Abstentions remain valid observations but carry neither
confidence nor source spans.

## 7. Build a reviewable contribution

Accepted or revised observations can become contribution artifacts. This
DisMech payload uses fields and enums from a pinned LinkML schema
revision. The package validates the supported subset and returns the
schema receipt; the upstream DisMech validator and a human submission
are still required.

``` r
dismech_payload <- list(
  name = "Marfan Syndrome",
  disease_term = list(id = "MONDO:0007947", label = "Marfan syndrome"),
  pathophysiology = list(list(
    name = "FBN1 Gene Mutation",
    biological_scale = "MOLECULAR",
    description = hypotheses$claim_text,
    evidence = list(list(
      reference = hypothesis_sources$citation,
      supports = "SUPPORT",
      evidence_source = "HUMAN_CLINICAL",
      quote_role = "BACKGROUND",
      snippet = hypothesis_sources$source_span,
      explanation = "The quoted source identifies FBN1 as causal in Marfan syndrome."
    ))
  ))
)

contribution <- ducksemantics_contribution(
  hypotheses,
  hypothesis_sources,
  dismech$fields,
  hypothesis_id = "readme-hypothesis-1",
  target = "dismech",
  payload = dismech_payload
)
contribution$schema
#>    target                                    schema_uri
#> 1 dismech https://github.com/monarch-initiative/dismech
#>                            schema_revision                     schema_path
#> 1 3e6cb8631b383355d9b590e5930a38f1813f9094 src/dismech/schema/dismech.yaml
#>                                                                                                   validation_scope
#> 1 Pinned disease, pathophysiology, and evidence fields/enums; run the upstream LinkML validator before submission.
```

The artifact is suitable for review. It is not inserted into the
imported DisMech relation and is not published automatically.

``` r
DBI::dbDisconnect(conn, shutdown = TRUE)
```

## Boundaries and reference

- [Native kernels](docs/native-kernels.md) documents SQL signatures,
  supported architectures, numerical tolerances and oracle commands.
- [Semantic relations](docs/semantic-relations.md) documents HPO,
  Monarch, literature, source-release, hypothesis and contribution
  contracts.
- [R package walkthrough](r/ducksemantics/README.md) exercises
  acceptance, rejection and release-cutoff behavior without model
  weights.
- [VariantStoryBench](https://github.com/RGenomicsETL/VariantStoryBench)
  owns candidate evaluation, grounding metrics and abstention
  accounting. Its `bench_ducksemantics_grounding()` adapter consumes the
  hypothesis relation.

DuckSemantics does not run arbitrary GGUF architectures. It does not
treat model scores as calibrated confidence, copy RClinVarbitration’s
source history, or publish ontology and knowledge-base contributions
without human action.

## Reproduce

Rendering requires `knitr`, `rmarkdown`, Pandoc and a local
EmbeddingGemma GGUF. The native and package checks use offline fixtures
and do not need model weights.

``` sh
make test
make sanitize
make r-check
make r-sync-check

DUCKSEMANTICS_EMBED_MODEL=/path/to/embeddinggemma-300M-Q8_0.gguf make readme
```

`make readme` executes every chunk, writes `README.md`, and builds the
static site from that rendered Markdown. The reference model is
revision-pinned and SHA-256-verified in CI but is not distributed with
the package. Mondo fixture content is CC BY 4.0; DisMech fixture content
retains its BSD 3-Clause notice. See
`r/ducksemantics/inst/LICENSE.note`.
