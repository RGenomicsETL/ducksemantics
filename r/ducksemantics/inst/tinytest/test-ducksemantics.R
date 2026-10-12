schema <- ducksemantics_schema_sql()
expect_true(any(grepl("semantic_nodes", schema, fixed = TRUE)))
expect_true(any(grepl("semantic_embeddings", schema, fixed = TRUE)))
expect_true(any(grepl("semantic_token_embeddings", schema, fixed = TRUE)))
expect_false(any(grepl("benchmark", schema, ignore.case = TRUE)))

tables <- ducksemantics_tables()
expect_equal(tables[["nodes"]], "semantic_nodes")
expect_equal(tables[["entailed_edges"]], "semantic_entailed_edges")
expect_equal(tables[["embeddings"]], "semantic_embeddings")

projection <- ducksemantics_projection_sql("source_edges", "subject", "predicate", "object", attrs = "attrs")
expect_true(grepl('"subject" AS from_id', projection, fixed = TRUE))
expect_true(grepl('"semantic_edges_subj_idx"', projection, fixed = TRUE))
closure <- ducksemantics_closure_sql(c("rdfs:subClassOf", "BFO:0000050"))
expect_true(grepl("WITH RECURSIVE closure", closure, fixed = TRUE))
expect_true(grepl("'BFO:0000050'", closure, fixed = TRUE))
empty_closure <- ducksemantics_closure_sql(character())
expect_true(grepl("CREATE OR REPLACE TABLE", empty_closure, fixed = TRUE))
expect_true(grepl('"semantic_entailed_edges_subj_idx"', empty_closure, fixed = TRUE))
expect_true(grepl('"semantic_entailed_edges_obj_idx"', empty_closure, fixed = TRUE))
expect_equal(ducksemantics_normalize(c("  Short---Stature ", "SEIZURES")), c("short stature", "seizures"))
expect_error(ducksemantics_tables("bad-prefix"))

expect_true(s7contract::implements(ducksemantics_prompt_runner(function(prompt) "[]"), DucksemanticsPromptRunner))
expect_true(s7contract::implements(ducksemantics_json_judgment_parser(), DucksemanticsJudgmentParser))
expect_true(s7contract::implements(ducksemantics_lexical_annotator(), DucksemanticsAnnotator))
embedding_provider <- ducksemantics_embedding_provider(function(text) matrix(seq_along(text), ncol = 1L))
expect_equal(dim(ducksemantics_embed(embedding_provider, c("alpha", "beta"))), c(2L, 1L))
token_provider <- ducksemantics_token_embedding_provider(function(text) lapply(text, function(one) {
  tokens <- strsplit(one, " ", fixed = TRUE)[[1L]]
  list(embeddings = cbind(seq_along(tokens), rev(seq_along(tokens))),
    token_index = seq_along(tokens) - 1L, tokens = tokens,
    start_offset = seq_along(tokens) - 1L, end_offset = seq_along(tokens))
}))
token_batch <- ducksemantics_token_embedding_batch_from_provider(c("short stature", "seizure"), token_provider,
  subject_id = c("HP:0004322", "HP:0001250"))
expect_true(is.data.frame(token_batch))
expect_equal(nrow(token_batch), 3L)

source_documents <- data.frame(document_id = "case-1", source_text = "Patient has seizures.", stringsAsFactors = FALSE)
contexts <- c("present", "absent/negated", "family_history", "uncertain", "conflict", "unsupported")
hpo_observations <- data.frame(
  document_id = rep("case-1", length(contexts)), hpo_id = rep("HP:0001250", length(contexts)),
  start_offset = rep(12L, length(contexts)), end_offset = rep(20L, length(contexts)),
  source_text = rep("seizures", length(contexts)), context_status = contexts,
  method = rep("lexical_alias", length(contexts)), provider_id = rep("tiny", length(contexts)),
  provider_version = rep("1", length(contexts)), confidence = rep(.9, length(contexts)),
  status = rep("accepted", length(contexts)), stringsAsFactors = FALSE
)
accepted <- ducksemantics_hpo_observations(source_documents, hpo_observations)
expect_equal(accepted$context_status, contexts)
expect_equal(ducksemantics_hpo_observation_contract(), names(accepted))
hallucinated <- hpo_observations[1L, , drop = FALSE]; hallucinated$source_text <- "diabetes"
expect_error(ducksemantics_hpo_observations(source_documents, hallucinated), "exactly")
empty_span <- hpo_observations[1L, , drop = FALSE]; empty_span$end_offset <- empty_span$start_offset
expect_error(ducksemantics_hpo_observations(source_documents, empty_span), "non-empty")
out_of_bounds <- hpo_observations[1L, , drop = FALSE]; out_of_bounds$end_offset <- 99L
expect_error(ducksemantics_hpo_observations(source_documents, out_of_bounds), "out of bounds")
rejected <- hpo_observations[1L, , drop = FALSE]; rejected$status <- "candidate"
expect_error(ducksemantics_hpo_observations(source_documents, rejected), "accepted")

monarch_releases <- data.frame(
  release_id = c("r-2024-02-09", "r-2024-02-10", "r-2025", "r-2024-02-10"),
  provider_id = c("monarch", "monarch", "monarch", "other-monarch"),
  effective_date = as.Date(c("2024-2-9", "2024-02-10", "2025-01-01", "2024-02-10"), format = "%Y-%m-%d"),
  source_ordinal = c(1, 2, 3, 1), stringsAsFactors = FALSE
)
release_catalog <- ducksemantics_monarch_release_catalog(monarch_releases)
expect_true(inherits(release_catalog$effective_date, "Date"))
monarch_phenotype <- data.frame(
  gene_id = c("HGNC:1", "HGNC:1", "HGNC:1"), phenotype_id = c("HP:0001250", "HP:0004322", "HP:0000707"),
  release_id = c("r-2024-02-09", "r-2024-02-10", "r-2025"), provider_id = "monarch",
  source_version = c("1", "2", "3"), stringsAsFactors = FALSE
)
imported_phenotype <- ducksemantics_monarch_import(monarch_phenotype, release_catalog, "gene_phenotype")
expect_true(all(c("release_id", "provider_id", "source_version") %in% names(imported_phenotype)))
expect_equal(ducksemantics_monarch_query(imported_phenotype, release_catalog, "gene_phenotype", "monarch", as_of = as.Date("2024-02-10"))$phenotype_id, "HP:0004322")
expect_equal(ducksemantics_monarch_query(imported_phenotype, release_catalog, "gene_phenotype", "monarch", as_of_ordinal = 1)$phenotype_id, "HP:0001250")
expect_error(ducksemantics_monarch_query(imported_phenotype, release_catalog, "gene_phenotype", "monarch", release_id = "unknown"), "Unknown Monarch release")
expect_error(ducksemantics_monarch_query(imported_phenotype, release_catalog, "gene_phenotype", "monarch", as_of = "2024-02-10"), "Date or POSIXct")
expect_error(ducksemantics_monarch_release_catalog(transform(monarch_releases, effective_date = as.character(effective_date), source_ordinal = NA_real_)), "Date or POSIXct")
monarch_disease_phenotype <- data.frame(
  disease_id = "MONDO:1", phenotype_id = "HP:0001250",
  release_id = "r-2024-02-09", provider_id = "monarch", stringsAsFactors = FALSE
)
imported_disease_phenotype <- ducksemantics_monarch_import(
  monarch_disease_phenotype, release_catalog, "disease_phenotype"
)
expect_equal(imported_disease_phenotype$predicate, "biolink:has_phenotype")
expect_equal(ducksemantics_monarch_query(
  imported_disease_phenotype, release_catalog, "disease_phenotype", "monarch",
  release_id = "r-2024-02-09"
)$phenotype_id, "HP:0001250")
monarch_disease <- data.frame(
  gene_id = c("HGNC:1", "HGNC:1", "HGNC:2", "HGNC:1"), disease_id = c("MONDO:1", "MONDO:2", "MONDO:3", "MONDO:OTHER"),
  release_id = c("r-2024-02-09", "r-2025", "r-2025", "r-2024-02-10"),
  provider_id = c("monarch", "monarch", "monarch", "other-monarch"), stringsAsFactors = FALSE
)
holdout <- ducksemantics_monarch_gene_disease_holdout_audit(monarch_disease, release_catalog, "monarch", "r-2024-02-09", "r-2025")
expect_equal(sort(holdout$disease_id), c("MONDO:2", "MONDO:3"))
expect_true(all(holdout$provider_id == "monarch"))

literature_snapshots <- data.frame(
  provider_id = c("pubmed", "pubmed", "pubmed", "europepmc"),
  snapshot_id = c("snapshot-9", "snapshot-10", "snapshot-11", "snapshot-10"),
  high_water_ordinal = c(9, 10, 11, 9),
  effective_at = as.POSIXct(c("2025-01-09 00:00:00", "2025-01-10 00:00:00", "2025-01-11 00:00:00", "2025-01-09 00:00:00"), tz = "UTC"),
  stringsAsFactors = FALSE
)
literature_articles <- data.frame(
  provider_id = c("pubmed", "pubmed", "pubmed", "pubmed", "pubmed", "europepmc"),
  article_id = c("pmid:1", "pmid:2", "pmid:1", "pmid:1", "pmid:2", "pmid:1"),
  pmid = c("1", "2", "1", "1", "2", "other-1"),
  version_id = c("source-9", "source-9", "source-10", "source-11", "source-11", "source-9"),
  source_ordinal = c(9, 9, 10, 11, 11, 9),
  is_deleted = c(FALSE, FALSE, FALSE, TRUE, TRUE, FALSE),
  stringsAsFactors = FALSE
)
literature_sections <- data.frame(
  provider_id = c("pubmed", "pubmed", "pubmed", "europepmc"),
  article_id = c("pmid:1", "pmid:2", "pmid:1", "pmid:1"),
  pmid = c("1", "2", "1", "other-1"),
  version_id = c("source-9", "source-9", "source-10", "source-9"),
  source_ordinal = c(9, 9, 10, 9),
  section = c("BACKGROUND", "BACKGROUND", "RESULTS", "BACKGROUND"),
  text = c("Seizure before update", "Seizure retained until deletion", "Seizure after update", "Seizure provider collision"),
  stringsAsFactors = FALSE
)
literature_catalog <- ducksemantics_literature_snapshot_catalog(literature_snapshots)
expect_equal(literature_catalog$high_water_ordinal, c(9, 10, 11, 9))
expect_true(inherits(literature_catalog$effective_at, "POSIXct"))
expect_true(all(is.na(ducksemantics_literature_snapshot_catalog(literature_snapshots[, names(literature_snapshots) != "effective_at"])$effective_at)))
literature_hits_9 <- ducksemantics_literature_retrieve(literature_articles, literature_sections, literature_snapshots,
  "seizure", "pubmed", "snapshot-9")
expect_equal(sort(literature_hits_9$pmid), c("1", "2"))
expect_equal(literature_hits_9$version_id[literature_hits_9$pmid == "1"], "source-9")
expect_true(all(literature_hits_9$provider_id == "pubmed"))
expect_true(all(literature_hits_9$end_offset > literature_hits_9$start_offset))
literature_hits_10 <- ducksemantics_literature_retrieve(literature_articles, literature_sections, literature_snapshots,
  "seizure", "pubmed", "snapshot-10")
expect_equal(sort(literature_hits_10$pmid), c("1", "2"))
expect_equal(literature_hits_10$version_id[literature_hits_10$pmid == "1"], "source-10")
expect_equal(literature_hits_10$source_ordinal[literature_hits_10$pmid == "1"], 10)
expect_equal(literature_hits_10$source_text[literature_hits_10$pmid == "1"], "Seizure")
expect_equal(attr(literature_hits_10, "high_water_ordinal"), 10)
literature_hits_11 <- ducksemantics_literature_retrieve(literature_articles, literature_sections, literature_snapshots,
  "seizure", "pubmed", "snapshot-11")
expect_equal(nrow(literature_hits_11), 0L)
expect_error(ducksemantics_literature_retrieve(literature_articles, literature_sections, literature_snapshots,
  "seizure", "pubmed", "not-cataloged"), "Unknown literature snapshot")
expect_error(ducksemantics_literature_retrieve(literature_articles, literature_sections, literature_snapshots,
  "seizure", "other-provider", "snapshot-10"), "Unknown literature snapshot")
expect_error(ducksemantics_literature_snapshot_catalog(transform(literature_snapshots, high_water_ordinal = 9.5)), "finite integer")
expect_error(ducksemantics_literature_snapshot_catalog(transform(literature_snapshots, effective_at = as.character(effective_at))), "POSIXct")
bad_sections <- literature_sections; bad_sections$source_ordinal[[1L]] <- 10
expect_error(ducksemantics_literature_retrieve(literature_articles, bad_sections, literature_snapshots,
  "seizure", "pubmed", "snapshot-10"), "must bind")
expect_false("ducksemantics_literature_projection" %in% getNamespaceExports("ducksemantics"))

obo_path <- tempfile(fileext = ".obo")
writeLines(c("format-version: 1.2", "", "[Term]", "id: HP:0000001", "name: All", "def: \"Root \\\"term\\\"\" []", "alt_id: HP:OLD0001", "synonym: \"whole phenotype\" EXACT []", "is_a: HP:0000118 {xref=\"PMID:1\"} ! Phenotypic abnormality", "relationship: part_of HP:0000707 ! Nervous system", "", "[Term]", "id: HP:9999999", "name: Obsolete term", "is_obsolete: true"), obo_path)
obo_graph <- ducksemantics_read_obo(obo_path, family = "HPO", source = "tiny.obo")
expect_equal(obo_graph$nodes$node_id, "HP:0000001")
expect_equal(obo_graph$nodes$description, 'Root "term"')
expect_equal(sort(obo_graph$aliases$alias), c("All", "HP:OLD0001", "whole phenotype"))
expect_equal(sort(obo_graph$edges$predicate), c("is_a", "part_of"))

if (requireNamespace("duckdb", quietly = TRUE) && requireNamespace("jsonlite", quietly = TRUE)) {
  conn <- ducksemantics_connect()
  on.exit(DBI::dbDisconnect(conn, shutdown = TRUE), add = TRUE)
  ducksemantics_init(conn)
  DBI::dbExecute(conn, empty_closure)
  closure_indexes <- DBI::dbGetQuery(conn, "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'semantic_entailed_edges'")
  expect_equal(nrow(closure_indexes), 2L)
  ducksemantics_write_obo(conn, obo_path, family = "HPO", source = "tiny.obo", replace = TRUE)
  expect_equal(ducksemantics_annotate(conn, "whole phenotype", document_id = "tiny-obo")$node_id, "HP:0000001")
  DBI::dbWriteTable(conn, "source_documents", source_documents, overwrite = TRUE)
  DBI::dbWriteTable(conn, "accepted_hpo_observations", hpo_observations, overwrite = TRUE)
  lazy_observations <- ducksemantics_hpo_observations("source_documents", "accepted_hpo_observations", conn = conn)
  expect_equal(nrow(lazy_observations), length(contexts))
  DBI::dbWriteTable(conn, "monarch_releases", monarch_releases, overwrite = TRUE)
  ducksemantics_monarch_import(monarch_disease, "monarch_releases", "gene_disease", conn = conn, table = "monarch_gene_disease", replace = TRUE)
  expect_equal(ducksemantics_monarch_query("monarch_gene_disease", "monarch_releases", "gene_disease", "monarch", release_id = "r-2024-02-09", conn = conn)$disease_id, "MONDO:1")
  DBI::dbWriteTable(conn, "literature_articles", literature_articles, overwrite = TRUE)
  DBI::dbWriteTable(conn, "literature_sections", literature_sections, overwrite = TRUE)
  DBI::dbWriteTable(conn, "literature_snapshots", literature_snapshots, overwrite = TRUE)
  literature_tables_before <- sort(DBI::dbListTables(conn))
  expect_equal(sort(ducksemantics_literature_retrieve("literature_articles", "literature_sections", "literature_snapshots", "seizure", "pubmed", "snapshot-10", conn = conn)$pmid), c("1", "2"))
  expect_equal(sort(DBI::dbListTables(conn)), literature_tables_before)
  ducksemantics_write_graph(conn,
    nodes = data.frame(node_id = c("HP:0004322", "HP:0001250", "HP:0000001"), family = "HPO", label = c("Short stature", "Seizure", "Phenotypic abnormality")),
    aliases = data.frame(node_id = c("HP:0004322", "HP:0001250"), alias = c("short stature", "seizures")),
    edges = data.frame(from_id = c("HP:0004322", "HP:0001250"), predicate = "is_a", to_id = "HP:0000001"), replace = TRUE)
  hits <- ducksemantics_annotate(conn, "The patient has short stature and seizures.", document_id = "case-001")
  expect_equal(sort(hits$node_id), c("HP:0001250", "HP:0004322"))
  expect_true(all(hits$method == "lexical_alias"))
  expect_equal(ducksemantics_annotate(conn, "SHORT---stature")$node_id, "HP:0004322")
  expect_true(DBI::dbExistsTable(conn, "semantic_alias_index"))
  graph_counts_before <- DBI::dbGetQuery(conn, "SELECT (SELECT COUNT(*) FROM semantic_aliases) AS aliases, (SELECT COUNT(*) FROM semantic_edges) AS edges")
  ducksemantics_write_graph(conn,
    aliases = data.frame(node_id = "HP:0004322", alias = "short stature", alias_kind = "label"),
    edges = data.frame(from_id = "HP:0004322", predicate = "is_a", to_id = "HP:0000001")
  )
  graph_counts_after <- DBI::dbGetQuery(conn, "SELECT (SELECT COUNT(*) FROM semantic_aliases) AS aliases, (SELECT COUNT(*) FROM semantic_edges) AS edges")
  expect_equal(graph_counts_after, graph_counts_before)

  embeddings <- ducksemantics_embedding_batch(matrix(c(1, 0, 0, 1), ncol = 2L, byrow = TRUE), c("HP:0004322", "HP:0001250"), provider = "tiny")
  ducksemantics_write_embeddings(conn, embeddings, replace = TRUE)
  vector_hits <- ducksemantics_embedding_search(conn, c(1, 0), provider = "tiny", top_k = 1L)
  expect_equal(vector_hits$subject_id, "HP:0004322")
  vector_rows <- ducksemantics_embedding_batch(
    matrix(c(1, 0, 0, .9, .1, 0, 0, 1, 0, .45, .55, 0), ncol = 3L, byrow = TRUE),
    c("HP:0004322", "HP:0004322_synonym", "HP:0001250", "HP:0000001"),
    provider = "tiny", text = c("short stature", "short height", "seizure", "phenotypic abnormality")
  )
  ducksemantics_write_embeddings(conn, vector_rows, replace = TRUE)
  ducksemantics_write_embeddings(conn, ducksemantics_embedding_batch(matrix(c(1, 0, 0), ncol = 3L), "other-provider", provider = "other"), replace = TRUE)
  vector_top <- ducksemantics_embedding_search(conn, c(1, 0, 0), provider = "tiny", subject_kind = "node", top_k = 2L)
  expect_equal(vector_top$subject_id[[1L]], "HP:0004322")
  expect_equal(nrow(vector_top), 2L)
  materialized <- ducksemantics_materialize_embedding_index(conn, 3L, provider = "tiny", subject_kind = "node", hnsw = FALSE)
  indexed_hits <- ducksemantics_embedding_search(conn, c(0, 1, 0), table = materialized, provider = "tiny", subject_kind = "node", top_k = 1L)
  expect_equal(indexed_hits$subject_id, "HP:0001250")

  token_embeddings <- ducksemantics_token_embedding_batch(matrix(c(1, 0, .8, .2, 0, 1), ncol = 2L, byrow = TRUE), c("HP:0004322", "HP:0004322", "HP:0001250"), provider = "tiny-token", token = c("short", "stature", "seizure"))
  ducksemantics_write_token_embeddings(conn, token_embeddings, replace = TRUE)
  late_hits <- ducksemantics_late_interaction_search(conn, matrix(c(1, 0, 0, 1), ncol = 2L, byrow = TRUE), provider = "tiny-token", top_k = 2L)
  expect_equal(late_hits$subject_id[[1L]], "HP:0004322")
  expect_true(late_hits$score[[1L]] > late_hits$score[[2L]])
  candidate_hit <- ducksemantics_late_interaction_search(conn, matrix(c(0, 1), ncol = 2L),
    provider = "tiny-token", subject_kind = "node", candidate_subject_id = "HP:0001250", top_k = 5L
  )
  expect_equal(candidate_hit$subject_id, "HP:0001250")
  expect_error(ducksemantics_late_interaction_search(conn, matrix(c(0, 0), ncol = 2L)), "non-zero norm")
  collision_batch <- ducksemantics_token_embedding_batch(
    matrix(c(1, 0, 0, 1), ncol = 2L, byrow = TRUE),
    subject_id = c("collision-a", "collision-b"), provider = "collision-provider",
    block_id = c("same-block", "same-block")
  )
  ducksemantics_write_token_embeddings(conn, collision_batch, replace = TRUE)
  collision_hits <- ducksemantics_late_interaction_search(conn, matrix(c(1, 0), ncol = 2L), provider = "collision-provider", top_k = 5L)
  expect_equal(nrow(collision_hits), 2L)
  expect_equal(sort(collision_hits$subject_id), c("collision-a", "collision-b"))

  runner <- ducksemantics_prompt_runner(function(prompt) jsonlite::toJSON(data.frame(mention_id = hits$mention_id, decision = "keep", confidence = 1), dataframe = "rows", auto_unbox = TRUE))
  judgments <- ducksemantics_judge("The patient has short stature and seizures.", hits, runner = runner, instructions = "Return JSON.")
  expect_equal(nrow(judgments), nrow(hits))
  expect_true(all(judgments$decision == "keep"))
  parsed <- ducksemantics_parse(ducksemantics_json_judgment_parser(), '[{"mention_id":"m1","decision":"keep"}]')
  expect_equal(parsed$mention_id, "m1")
  expect_error(ducksemantics_judge("text", hits,
    runner = ducksemantics_prompt_runner(function(prompt) '[{"mention_id":"case-001:000017","decision":"keep"}]')
  ), "exactly one result")
  expect_error(ducksemantics_judge("text", hits,
    runner = ducksemantics_prompt_runner(function(prompt) jsonlite::toJSON(
      data.frame(mention_id = hits$mention_id, decision = c("replace", "keep"), replacement_node_id = c("HP:NOT_SUPPLIED", NA_character_)),
      dataframe = "rows", auto_unbox = TRUE, na = "null"
    ))
  ), "outside supplied candidates")
  empty_judgments <- ducksemantics_judge("text", hits[FALSE, , drop = FALSE],
    runner = ducksemantics_prompt_runner(function(prompt) "[]")
  )
  expect_equal(nrow(empty_judgments), 0L)
}
