
<!-- README.md is generated from README.Rmd by `make readme`. Edit README.Rmd. -->

# DuckSemantics

DuckSemantics is a DuckDB extension for semantic computation over local
files. It inspects GGUF model files, tokenizes text, runs supported CPU
models, and scores vectors in ordinary SQL. The **ducksemantics** R
package builds and loads the extension. It also imports ontology and
knowledge-base records, with their release identities.

One rule runs through this walkthrough: **a retrieval score is not a
source fact**. Imported records keep their provider, release, and
receipt. Scores and model output go into their own relations, so they
never overwrite a source record.

The walkthrough below is a single executed session. Every output comes
from running it against a real EmbeddingGemma model and the bundled
source fixtures.

| Step                                               | What it shows                                          |
|----------------------------------------------------|--------------------------------------------------------|
| [1](#1-select-and-inspect-a-model)                 | Read the model file’s metadata and tensor layout       |
| [2](#2-tokenize-with-the-models-vocabulary)        | Turn text into the model’s token ids                   |
| [3](#3-embed-short-texts)                          | Produce vectors and check their type, width, and norm  |
| [4](#4-load-versioned-source-records)              | Import Mondo and DisMech records with release receipts |
| [5](#5-rank-source-records-for-a-question)         | Score DisMech mechanisms against a question in SQL     |
| [6](#6-trace-a-result-to-its-evidence-and-release) | Join the top result back to citations and releases     |

## Install and connect

The R package lives in `r/ducksemantics`. It bundles the C sources and
builds the extension during installation. It does not download headers
or model weights. Linux needs a C11 compiler and GNU make. The Linux
x86-64 build is verified locally; the distribution pipeline tests Linux
x86-64 and ARM64. macOS, Windows, and webR await native portability
verification.

From a repository checkout, with the R dependencies in
`r/ducksemantics/DESCRIPTION` installed (including `duckdb`):

``` sh
make r-bootstrap
R CMD INSTALL r/ducksemantics
```

For an isolated development library, use `make r-install` and start R
with `R_LIBS=build/r/library`. To build only the standalone extension,
run `make` and load `build/ducksemantics.duckdb_extension` in a DuckDB
CLI started with `-unsigned`. The extension uses DuckDB’s stable C
extension ABI **1.2.0**.

``` r
library(ducksemantics)
conn <- ducksemantics_connect()
```

`ducksemantics_connect()` opens an isolated DuckDB database, allows the
locally built unsigned extension, and loads it. These settings apply
only to that connection. The R and SQL chunks below share it. A SQL
chunk refers to an R variable as `?name`, and the value is quoted
safely.

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

## 1. Select and inspect a model

Package installation never bundles or downloads model weights. You
supply a local file through `DUCKSEMANTICS_EMBED_MODEL`. This README
needs EmbeddingGemma-300M in GGUF format (for example
`embeddinggemma-300M-Q8_0.gguf`). If the variable is not set, the render
stops here.

``` r
embed_model <- Sys.getenv("DUCKSEMANTICS_EMBED_MODEL")
stopifnot(
  "Set DUCKSEMANTICS_EMBED_MODEL to a local EmbeddingGemma-300M GGUF file." = nzchar(embed_model),
  "DUCKSEMANTICS_EMBED_MODEL must name an existing local file." = file.exists(embed_model)
)
embed_model_sha256 <- digest::digest(file = embed_model, algo = "sha256")
basename(embed_model)
#> [1] "embeddinggemma-300M-Q8_0.gguf"
embed_model_sha256
#> [1] "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"
```

A GGUF file holds metadata, a vocabulary, and tensor payloads. You can
read all three without running the model. The architecture key tells you
which kernel can run it.

``` sql
SELECT key, value
FROM gguf_metadata(?embed_model)
WHERE key IN ('general.architecture', 'general.name',
              'gemma-embedding.embedding_length',
              'gemma-embedding.block_count', 'tokenizer.ggml.model');
```

<div class="knitsql-table">

| key                              | value               |
|:---------------------------------|:--------------------|
| general.architecture             | gemma-embedding     |
| general.name                     | Embeddinggemma 300m |
| gemma-embedding.block_count      | 24                  |
| gemma-embedding.embedding_length | 768                 |
| tokenizer.ggml.model             | llama               |

</div>

``` sql
SELECT ggml_type, count(*) AS tensors,
       round(sum(n_bytes) / 2^20, 1) AS mebibytes
FROM gguf_tensors(?embed_model)
GROUP BY ggml_type
ORDER BY mebibytes DESC;
```

<div class="knitsql-table">

| ggml_type | tensors | mebibytes |
|:----------|--------:|----------:|
| q8_0      |     171 |     311.6 |
| f32       |     145 |       0.3 |

</div>

The package also includes small **synthetic** GGUF files. The offline
tests use them to exercise the parser and tokenizers. They are not
trained models. The header shows the difference:

``` r
parser_fixture <- system.file("extdata", "valid.gguf",
                              package = "ducksemantics", mustWork = TRUE)
```

``` sql
SELECT 'caller model' AS file, tensor_count, metadata_count, file_bytes
FROM gguf_header(?embed_model)
UNION ALL
SELECT 'synthetic fixture', tensor_count, metadata_count, file_bytes
FROM gguf_header(?parser_fixture);
```

<div class="knitsql-table">

| file              | tensor_count | metadata_count | file_bytes |
|:------------------|-------------:|---------------:|-----------:|
| caller model      |          316 |             41 |  333590944 |
| synthetic fixture |            3 |             18 |        960 |

</div>

## 2. Tokenize with the model’s vocabulary

Tokenization uses only the vocabulary and tokenizer metadata, not the
weights. With `add_special = true`, the Gemma profile adds BOS and EOS.
Joining each id to `tokenizer.ggml.tokens` shows the stored piece; `▁`
marks a word boundary.

``` sql
WITH encoded AS (
  SELECT semantic_tokenize(?embed_model, 'Marfan syndrome aortic dilation', true) AS ids
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
) AS round_trip;
```

<div class="knitsql-table">

| round_trip                      |
|:--------------------------------|
| Marfan syndrome aortic dilation |

</div>

The decoder omits control tokens, so the text round-trips.
`semantic_embed` adds the task prefix and tokenizes internally. You do
not pass these ids to it.

## 3. Embed short texts

`semantic_embed(path, text, task, dimensions, normalize)` runs the
EmbeddingGemma forward pass. The task selects the input prefix. Here,
`retrieval_query` means `task: search result | query:`. The model
returns 768 values. A smaller `dimensions` keeps the leading values, and
`normalize` then scales the subvector to unit length.

``` r
embed_task <- "retrieval_query"
embed_dimensions <- 256L
```

``` sql
CREATE TABLE short_texts AS
SELECT text_id, text,
       semantic_embed(?embed_model, text, ?embed_task, ?embed_dimensions, true) AS embedding
FROM (VALUES
  ('marfan',     'Marfan syndrome'),
  ('fibrillin',  'defective fibrillin-1 microfibrils'),
  ('diabetes',   'insulin resistance in type 2 diabetes')
) AS t(text_id, text);
```

Each result is a `FLOAT[]` list. Check its type, width, and norm before
using it:

``` sql
SELECT text_id,
       typeof(embedding) AS type,
       len(embedding) AS dimensions,
       sqrt(list_dot_product(embedding, embedding)) AS norm,
       list_transform(embedding[1:4], x -> round(x, 4))::VARCHAR AS first_values
FROM short_texts
ORDER BY text_id;
```

``` r
vector_shape
#>     text_id    type dimensions      norm                        first_values
#> 1  diabetes FLOAT[]        256 0.9999999  [-0.1815, 0.0743, -0.0354, 0.0624]
#> 2 fibrillin FLOAT[]        256 1.0000000 [-0.178, -0.0132, -0.0081, -0.0038]
#> 3    marfan FLOAT[]        256 1.0000001  [-0.233, -0.0015, 0.0897, -0.1025]
stopifnot(vector_shape$type == "FLOAT[]",
          vector_shape$dimensions == embed_dimensions,
          abs(vector_shape$norm - 1) < 1e-5)
```

For unit vectors, the dot product is the cosine similarity. These are
raw similarities between the three texts, not calibrated probabilities:

``` sql
SELECT a.text_id AS text_a, b.text_id AS text_b,
       round(list_dot_product(a.embedding, b.embedding), 4) AS cosine
FROM short_texts a
JOIN short_texts b ON a.text_id < b.text_id
ORDER BY cosine DESC;
```

<div class="knitsql-table">

| text_a    | text_b    | cosine |
|:----------|:----------|-------:|
| fibrillin | marfan    | 0.3841 |
| diabetes  | marfan    | 0.2886 |
| diabetes  | fibrillin | 0.2468 |

</div>

Vectors need the same model, width, and normalization, with the model’s
paired query/document task conventions. The tables below record those
settings next to each vector.

## 4. Load versioned source records

Disease labels change, and terms become obsolete. To answer “what did
the source say at a given date?”, the source catalog chooses a release
by **provider availability time**. It does not sort version strings or
commit dates. Each import checks the file’s bytes against a SHA-256
receipt before it reads them.

The bundled fixtures are real upstream content. The Mondo fixture holds
three complete term stanzas from release `v2026-10-06`. The DisMech
fixture is the verbatim Marfan syndrome record at commit `fac6186`. The
receipts record the upstream digest as well as the fixture digest.

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
  fixture("mondo-subset.obo"), mondo_release,
  source_id = mondo_release$source_id,
  source_uri = mondo_release$source_uri,
  sha256 = mondo_release$sha256
)

dismech_release <- ducksemantics_source_release_select(
  catalog, "DisMech", release_id = "fac61860115b1fdb194de4f4cacef3a8f6a398af"
)
dismech <- ducksemantics_dismech_import(
  fixture("dismech-marfan.yaml"), dismech_release,
  source_id = dismech_release$source_id,
  source_uri = dismech_release$source_uri,
  sha256 = dismech_release$sha256
)

DBI::dbWriteTable(conn, "mondo_terms", mondo$terms)
DBI::dbWriteTable(conn, "mondo_mappings", mondo$mappings)
DBI::dbWriteTable(conn, "mondo_obsolescence", mondo$obsolescence)
DBI::dbWriteTable(conn, "dismech_fields", dismech$fields)
```

Every imported row carries its release and receipt:

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

The Mondo subset includes an obsolete term and its replacement:

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

A Mondo `xref` alone does not mean exact equivalence. The importer keeps
each mapping assertion as written, so an explicit SKOS predicate stays
visible:

``` sql
SELECT tag, value
FROM mondo_mappings
WHERE node_id = 'MONDO:0007947'
  AND value = 'skos:exactMatch OMIM:154700';
```

<div class="knitsql-table">

| tag            | value                       |
|:---------------|:----------------------------|
| property_value | skos:exactMatch OMIM:154700 |

</div>

DisMech is imported as one row per scalar field. Each field has a JSON
pointer with zero-based array indices. The pointer locates a field in
the imported record. It is not a new upstream accession.

## 5. Rank source records for a question

The Marfan record describes six pathophysiology mechanisms. This query
gets each description and its name from the source fields:

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

Documents use the `retrieval_document` task, and questions use
`retrieval_query`. Both use the same model, width, and normalization.
The vector table stores the model digest and settings. It refers to the
source rows by `source_row_id` and does not copy them.

``` sql
CREATE TABLE mechanism_vectors AS
SELECT provider_id, release_id, source_id, source_row_id,
       ?embed_model_sha256 AS model_sha256,
       'retrieval_document' AS task,
       ?embed_dimensions AS dimensions, true AS normalized,
       semantic_embed(?embed_model, description, 'retrieval_document',
                      ?embed_dimensions, true) AS embedding
FROM mechanisms;
```

``` r
question <- "Why does the aortic wall weaken and dilate?"
```

Scores go into their own relation, with the question that produced them:

``` sql
CREATE TABLE mechanism_scores AS
WITH query AS (
  SELECT semantic_embed(?embed_model, ?question, ?embed_task,
                        ?embed_dimensions, true) AS vector
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

The ranking tells you which source passage to read first. It does not
show that the mechanism is correct. Those claims come from the record
and its citations.

## 6. Trace a result to its evidence and release

A score says nothing about evidence. Put the source’s own citation count
next to each score:

``` sql
CREATE TABLE mechanism_evidence AS
SELECT m.provider_id, m.release_id, m.source_id, m.source_row_id,
       m.mechanism, m.field_prefix, e.field_path, e.value AS reference
FROM mechanisms m
JOIN dismech_fields e
  ON e.provider_id = m.provider_id AND e.release_id = m.release_id
 AND e.source_id = m.source_id
 AND regexp_matches(e.field_path,
                    '^' || m.field_prefix || '/evidence/[0-9]+/reference$');
```

``` sql
SELECT m.mechanism, round(s.cosine, 4) AS cosine,
       count(e.reference) AS cited_references
FROM mechanism_scores s
JOIN mechanisms m USING (provider_id, release_id, source_id, source_row_id)
LEFT JOIN mechanism_evidence e USING (provider_id, release_id, source_id, source_row_id)
GROUP BY m.mechanism, s.cosine
ORDER BY s.cosine DESC;
```

<div class="knitsql-table">

| mechanism                       | cosine | cited_references |
|:--------------------------------|-------:|-----------------:|
| Extracellular Matrix Remodeling | 0.7187 |                0 |
| Vascular Inflammation           | 0.4436 |                0 |
| Impaired Mechanotransduction    | 0.4220 |                0 |
| Mitochondrial Dysfunction       | 0.4121 |                0 |
| Dysregulated TGF-beta Signaling | 0.3600 |                0 |
| FBN1 Gene Mutation              | 0.2979 |                8 |

</div>

In this record, only one mechanism has its own `evidence` entries. The
closest passage has none. Retrieval found relevant text, but it cannot
add a citation the source does not contain. The best-scoring mechanism
that does cite evidence has these references:

``` sql
WITH best AS (
  SELECT m.provider_id, m.release_id, m.source_id, m.source_row_id
  FROM mechanism_scores s
  JOIN mechanisms m USING (provider_id, release_id, source_id, source_row_id)
  WHERE EXISTS (
    SELECT 1 FROM mechanism_evidence e
    WHERE e.provider_id = m.provider_id AND e.release_id = m.release_id
      AND e.source_id = m.source_id AND e.source_row_id = m.source_row_id
  )
  ORDER BY s.cosine DESC
  LIMIT 1
)
SELECT e.mechanism, e.reference, s.value AS supports, e.field_path
FROM mechanism_evidence e
JOIN best USING (provider_id, release_id, source_id, source_row_id)
LEFT JOIN dismech_fields s
  ON s.provider_id = e.provider_id AND s.release_id = e.release_id
 AND s.source_id = e.source_id
 AND s.field_path = regexp_replace(e.field_path, '/reference$', '/supports')
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

The DisMech record names its disease with a Mondo identifier. Joining
that identifier to the Mondo import connects two separately versioned
sources. Each side keeps its own release and receipt:

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

Source facts, retrieval scores, and model-generated proposals stay in
separate relations. The DisMech importer does not validate a DisMech
contribution, and it does not check the cited papers. Model-generated
hypotheses and reviewed contribution export are open work. They need
their own relations and an executable Bench consumer before evaluation
([issue \#1](https://github.com/RGenomicsETL/ducksemantics/issues/1)).

## Check the late-interaction score

`semantic_maxsim(query, document)` scores ColBERT-style token vectors.
For each query token it takes the best dot product with any document
token, then adds these maxima:

``` text
MaxSim(Q, D) = sum over query tokens i of  max over document tokens j of  dot(q_i, d_j)
```

This is a raw dot-product sum. Normalized inputs make each dot product a
cosine similarity; the sum is neither a single cosine nor a calibrated
probability. Both inputs must have the same vector width. The toy
vectors below check the arithmetic only. They are not a biomedical
result. Two unit query vectors that match exactly score `2`. A document
that matches only one of them scores `1`.

``` sql
WITH documents(document_id, vectors) AS (
  VALUES
    ('both',  [[1.0, 0.0], [0.0, 1.0]]::FLOAT[][]),
    ('first', [[1.0, 0.0]]::FLOAT[][]),
    ('away',  [[-1.0, 0.0], [0.0, -1.0]]::FLOAT[][])
)
SELECT document_id,
       semantic_maxsim([[1.0, 0.0], [0.0, 1.0]], vectors) AS score
FROM documents
ORDER BY score DESC, document_id;
```

<div class="knitsql-table">

| document_id | score |
|:------------|------:|
| both        |     2 |
| first       |     1 |
| away        |     0 |

</div>

``` r
DBI::dbDisconnect(conn, shutdown = TRUE)
```

## Further reference

- [Native kernel reference](docs/native-kernels.md): SQL signatures, the
  EmbeddingGemma, LFM2.5-ColBERT-350M, and LFM2-MoE profiles, tokenizer
  limits, numerical tolerances, and oracle commands.
- [Semantic relation contracts](docs/semantic-relations.md): HPO
  validation, dated Monarch pack projections, and append-only literature
  snapshots.

Model support depends on the architecture. The extension does not run
arbitrary GGUF files. Unsupported architectures or tensor layouts raise
errors.

## Reproduce

Rendering needs `knitr`, `rmarkdown`, Pandoc, and a local EmbeddingGemma
GGUF file. The standalone SQL suite also needs the DuckDB CLI on `PATH`.

``` sh
make test          # Native C and standalone DuckDB SQL tests
make r-test        # Package build, install, and tinytest tests
make r-check       # R CMD check of the source tarball
make r-sync-check  # Check that bundled C sources match the canonical tree

git submodule update --init extension-ci-tools
make configure release test_release  # Distribution artifact and C/CLI contracts

DUCKSEMANTICS_EMBED_MODEL=/path/to/embeddinggemma-300M-Q8_0.gguf make readme
```

`make readme` runs every chunk in this document and stops on the first
error. It writes `README.md`. It then builds a static HTML copy of
`README.md` and `docs/` in `build/site`, without running the chunks
again. The offline test suites use only the synthetic fixtures and need
no model weights. R-package CI stages a separate, revision-pinned and
SHA-256-verified reference model for this executed document; weights are
governed by the Gemma Terms of Use and are not included in published
package artifacts.

The distribution workflow uses the pinned `extension-ci-tools` submodule
and its reusable stable-C-ABI pipeline. Native Make and CMake share
`src/sources.list`. Release and debug test targets run the existing C
oracles and CLI SQL suite against the distribution artifact, with no
wheel-based test skip. R source-package installation remains independent
of the CI helpers.

Mondo’s fixture is distributed under CC BY 4.0. DisMech’s fixture keeps
its upstream BSD 3-Clause notice. Source URIs, release identities,
selection details, and SHA-256 receipts are bundled with the fixtures.
See `r/ducksemantics/inst/LICENSE.note` for attribution.
