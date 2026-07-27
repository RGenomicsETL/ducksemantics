if (requireNamespace("duckdb", quietly = TRUE)) {
  conn <- ducksemantics_connect()
  on.exit(DBI::dbDisconnect(conn, shutdown = TRUE), add = TRUE)

  pack_path <- tempfile(fileext = ".duckdb")
  on.exit(unlink(pack_path), add = TRUE)
  DBI::dbExecute(conn, glue::glue_sql(
    "ATTACH {pack_path} AS monarch_pack", .con = conn
  ))
  DBI::dbExecute(conn, "
    CREATE TABLE monarch_pack.denormalized_edges (
      id VARCHAR,
      predicate VARCHAR,
      category VARCHAR,
      agent_type VARCHAR,
      aggregator_knowledge_source VARCHAR[],
      knowledge_level VARCHAR,
      primary_knowledge_source VARCHAR,
      file_source VARCHAR,
      provided_by VARCHAR,
      has_evidence VARCHAR[],
      publications VARCHAR[],
      qualifiers VARCHAR[],
      disease_context_qualifier VARCHAR,
      original_predicate VARCHAR,
      frequency_qualifier VARCHAR,
      onset_qualifier VARCHAR,
      sex_qualifier VARCHAR,
      object_aspect_qualifier VARCHAR,
      species_context_qualifier VARCHAR,
      stage_qualifier VARCHAR,
      qualifier VARCHAR,
      object_specialization_qualifier VARCHAR,
      negated VARCHAR,
      subject VARCHAR,
      object VARCHAR,
      original_subject VARCHAR,
      original_object VARCHAR,
      evidence_count BIGINT,
      grouping_key VARCHAR,
      subject_label VARCHAR,
      subject_category VARCHAR,
      subject_namespace VARCHAR,
      subject_closure VARCHAR[],
      subject_taxon VARCHAR,
      object_label VARCHAR,
      object_category VARCHAR,
      object_namespace VARCHAR,
      object_closure VARCHAR[],
      object_taxon VARCHAR
    )
  ")
  DBI::dbExecute(conn, "
    CREATE TABLE monarch_pack.nodes (
      id VARCHAR, category VARCHAR, name VARCHAR, xref VARCHAR[],
      synonym VARCHAR[], namespace VARCHAR
    )
  ")
  DBI::dbExecute(conn, "CREATE TABLE monarch_pack.information_content (term VARCHAR, ic DOUBLE)")
  DBI::dbExecute(conn, "
    CREATE TABLE monarch_pack.node_has_phenotype (
      id VARCHAR, has_phenotype VARCHAR[], has_phenotype_label VARCHAR[],
      has_phenotype_count BIGINT, has_phenotype_closure VARCHAR[],
      has_phenotype_closure_label VARCHAR[]
    )
  ")
  DBI::dbExecute(conn, "
    INSERT INTO monarch_pack.denormalized_edges (
      id, predicate, category, agent_type, aggregator_knowledge_source,
      knowledge_level, primary_knowledge_source, file_source, provided_by,
      has_evidence, publications, qualifiers, disease_context_qualifier,
      original_predicate, frequency_qualifier, onset_qualifier, sex_qualifier,
      object_aspect_qualifier, species_context_qualifier, stage_qualifier,
      qualifier, object_specialization_qualifier, negated, subject, object, original_subject,
      original_object, evidence_count, grouping_key, subject_label,
      subject_category, subject_namespace, subject_closure, subject_taxon,
      object_label, object_category, object_namespace, object_closure, object_taxon
    ) VALUES (
      'edge-gd-forward', 'biolink:causes', 'biolink:GeneToDiseaseAssociation',
      'manual_agent', ['infores:monarchinitiative', 'infores:aggregator'],
      'knowledge_assertion', 'infores:source', 'fixture.tsv', 'fixture',
      ['ECO:0001'], ['PMID:1'], ['subject=human'], 'MONDO:1', 'causes',
      'HP:0040281', 'HP:0003577', 'female', 'activity', 'NCBITaxon:9606',
      'adult', 'one', 'variant', 'False', 'HGNC:1', 'MONDO:1', 'HGNC:1',
      'MONDO:1', 2, 'group-gd-forward', 'GENE1', 'biolink:Gene', 'HGNC',
      ['HGNC:1'], 'NCBITaxon:9606', 'Disease 1', 'biolink:Disease', 'MONDO',
      ['MONDO:1'], 'NCBITaxon:9606'
    )
  ")
  DBI::dbExecute(conn, "
    INSERT INTO monarch_pack.denormalized_edges
      (id, predicate, category, negated, subject, object, subject_category, object_category)
    VALUES
      ('edge-gd-reverse', 'biolink:associated_with', 'biolink:GeneToDiseaseAssociation', 'False', 'MONDO:2', 'HGNC:2', 'biolink:Disease', 'biolink:Gene'),
      ('edge-dp-forward', 'biolink:has_phenotype', 'biolink:DiseaseToPhenotypicFeatureAssociation', 'False', 'MONDO:3', 'HP:0001250', 'biolink:Disease', 'biolink:PhenotypicFeature'),
      ('edge-gp-reverse', 'biolink:has_phenotype', 'biolink:GeneToPhenotypicFeatureAssociation', 'False', 'HP:0004322', 'HGNC:3', 'biolink:PhenotypicFeature', 'biolink:Gene'),
      ('edge-missing-role', 'biolink:has_phenotype', 'biolink:GeneToPhenotypicFeatureAssociation', 'False', 'HGNC:4', 'HP:0000707', NULL, 'biolink:PhenotypicFeature'),
      ('edge-negated', 'biolink:causes', 'biolink:GeneToDiseaseAssociation', 'True', 'HGNC:5', 'MONDO:5', 'biolink:Gene', 'biolink:Disease'),
      ('edge-unsupported', 'biolink:related_to', 'biolink:Association', 'False', 'HGNC:6', 'HGNC:7', 'biolink:Gene', 'biolink:Gene'),
      ('edge-malformed', 'biolink:causes', 'biolink:GeneToDiseaseAssociation', 'False', 'HGNC:8', 'MONDO:8', 'Gene', 'biolink:Disease'),
      ('edge-missing-predicate', NULL, 'biolink:GeneToDiseaseAssociation', 'False', 'HGNC:9', 'MONDO:9', 'biolink:Gene', 'biolink:Disease'),
      ('edge-missing-negation', 'biolink:causes', 'biolink:GeneToDiseaseAssociation', NULL, 'HGNC:10', 'MONDO:10', 'biolink:Gene', 'biolink:Disease'),
      ('edge-malformed-negation', 'biolink:causes', 'biolink:GeneToDiseaseAssociation', 'unknown', 'HGNC:11', 'MONDO:11', 'biolink:Gene', 'biolink:Disease')
  ")
  DBI::dbExecute(conn, "
    INSERT INTO monarch_pack.nodes
    VALUES ('HGNC:1', 'biolink:Gene', 'GENE1', ['NCBIGene:1'], ['Gene one'], 'HGNC')
  ")
  DBI::dbExecute(conn, "INSERT INTO monarch_pack.information_content VALUES ('HP:0001250', 4.2)")
  DBI::dbExecute(conn, "
    INSERT INTO monarch_pack.node_has_phenotype
    VALUES ('HGNC:1', ['HP:0001250'], ['Seizure'], 1, ['HP:0000118', 'HP:0001250'], ['Phenotypic abnormality', 'Seizure'])
  ")
  DBI::dbExecute(conn, "ALTER TABLE monarch_pack.denormalized_edges RENAME TO denormalized_edges_source")
  DBI::dbExecute(conn, "
    CREATE VIEW monarch_pack.denormalized_edges AS
    SELECT * FROM monarch_pack.denormalized_edges_source
  ")
  DBI::dbExecute(conn, "DETACH monarch_pack")
  DBI::dbExecute(conn, glue::glue_sql(
    "ATTACH {pack_path} AS monarch_pack (READ_ONLY)", .con = conn
  ))

  releases <- data.frame(
    provider_id = c('infores:monarchinitiative', 'infores:monarchinitiative'),
    release_id = c('2026-07-14', '2026-08-14'),
    effective_date = as.Date(c('2026-07-14', '2026-08-14')),
    stringsAsFactors = FALSE
  )
  projected <- ducksemantics_monarch_project_pack(
    conn, 'monarch_pack', releases,
    provider_id = 'infores:monarchinitiative', release_id = '2026-07-14'
  )
  expect_equal(
    projected$relation,
    c('associations', 'gene_disease', 'disease_phenotype', 'gene_phenotype',
      'nodes', 'node_has_phenotype', 'information_content')
  )
  expect_equal(
    projected$view_name,
    c('semantic_monarch_associations', 'semantic_monarch_gene_disease',
      'semantic_monarch_disease_phenotype', 'semantic_monarch_gene_phenotype',
      'semantic_monarch_nodes', 'semantic_monarch_node_has_phenotype',
      'semantic_monarch_information_content')
  )
  expect_equal(names(projected), c('relation', 'view_name', 'provider_id', 'release_id'))
  expect_equal(nrow(projected), 7L)
  expect_true(all(projected$provider_id == 'infores:monarchinitiative'))
  expect_true(all(projected$release_id == '2026-07-14'))

  association_rows <- DBI::dbGetQuery(conn, "SELECT COUNT(*) AS n FROM semantic_monarch_associations")$n[[1L]]
  source_rows <- DBI::dbGetQuery(conn, "SELECT COUNT(*) AS n FROM monarch_pack.denormalized_edges")$n[[1L]]
  expect_equal(association_rows, source_rows)
  expect_equal(association_rows, 11)
  provenance <- DBI::dbGetQuery(conn, "
    SELECT id, predicate, category, agent_type, knowledge_level,
      primary_knowledge_source,
      to_json(aggregator_knowledge_source) AS aggregators,
      to_json(has_evidence) AS evidence,
      to_json(publications) AS publications,
      to_json(qualifiers) AS qualifiers,
      original_subject, original_object, frequency_qualifier, stage_qualifier,
      subject_taxon, object_taxon
    FROM semantic_monarch_associations
    WHERE id = 'edge-gd-forward'
  ")
  expect_equal(provenance$id, 'edge-gd-forward')
  expect_equal(provenance$predicate, 'biolink:causes')
  expect_equal(provenance$category, 'biolink:GeneToDiseaseAssociation')
  expect_equal(provenance$agent_type, 'manual_agent')
  expect_equal(provenance$knowledge_level, 'knowledge_assertion')
  expect_equal(provenance$primary_knowledge_source, 'infores:source')
  expect_equal(provenance$aggregators, '["infores:monarchinitiative","infores:aggregator"]')
  expect_equal(provenance$evidence, '["ECO:0001"]')
  expect_equal(provenance$publications, '["PMID:1"]')
  expect_equal(provenance$qualifiers, '["subject=human"]')
  expect_equal(provenance$original_subject, 'HGNC:1')
  expect_equal(provenance$original_object, 'MONDO:1')
  expect_equal(provenance$frequency_qualifier, 'HP:0040281')
  expect_equal(provenance$stage_qualifier, 'adult')
  expect_equal(provenance$subject_taxon, 'NCBITaxon:9606')
  expect_equal(provenance$object_taxon, 'NCBITaxon:9606')
  association_types <- DBI::dbGetQuery(conn, "
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_catalog = 'temp'
      AND table_name = 'semantic_monarch_associations'
      AND column_name IN ('aggregator_knowledge_source', 'has_evidence', 'publications', 'qualifiers')
    ORDER BY column_name
  ")
  expect_equal(association_types$data_type, rep('VARCHAR[]', 4L))

  gene_disease <- DBI::dbGetQuery(conn, "
    SELECT id, association_status, negation_status, source_direction, gene_id, disease_id
    FROM semantic_monarch_gene_disease
    ORDER BY id
  ")
  forward <- gene_disease[gene_disease$id == 'edge-gd-forward', , drop = FALSE]
  reverse <- gene_disease[gene_disease$id == 'edge-gd-reverse', , drop = FALSE]
  expect_equal(forward$association_status, 'supported_subject_to_object')
  expect_equal(forward$source_direction, 'subject_to_object')
  expect_equal(forward$gene_id, 'HGNC:1')
  expect_equal(forward$disease_id, 'MONDO:1')
  expect_equal(reverse$association_status, 'supported_object_to_subject')
  expect_equal(reverse$source_direction, 'object_to_subject')
  expect_equal(reverse$gene_id, 'HGNC:2')
  expect_equal(reverse$disease_id, 'MONDO:2')
  expect_equal(gene_disease$association_status[gene_disease$id == 'edge-negated'], 'negated')
  expect_equal(gene_disease$association_status[gene_disease$id == 'edge-unsupported'], 'unsupported_orientation')
  expect_equal(gene_disease$association_status[gene_disease$id == 'edge-malformed'], 'malformed_role_category')
  expect_equal(gene_disease$association_status[gene_disease$id == 'edge-missing-predicate'], 'missing_predicate')
  expect_equal(gene_disease$association_status[gene_disease$id == 'edge-missing-negation'], 'supported_subject_to_object')
  expect_equal(gene_disease$negation_status[gene_disease$id == 'edge-missing-negation'], 'not_provided')
  expect_equal(gene_disease$association_status[gene_disease$id == 'edge-malformed-negation'], 'malformed_negation')
  expect_equal(gene_disease$negation_status[gene_disease$id == 'edge-malformed-negation'], 'malformed')
  expect_true(is.na(gene_disease$gene_id[gene_disease$id == 'edge-negated']))
  expect_true(is.na(gene_disease$gene_id[gene_disease$id == 'edge-unsupported']))

  disease_phenotype <- DBI::dbGetQuery(conn, "
    SELECT association_status, disease_id, phenotype_id, source_subject, source_object
    FROM semantic_monarch_disease_phenotype
    WHERE id = 'edge-dp-forward'
  ")
  expect_equal(disease_phenotype$association_status, 'supported_subject_to_object')
  expect_equal(disease_phenotype$disease_id, 'MONDO:3')
  expect_equal(disease_phenotype$phenotype_id, 'HP:0001250')
  expect_equal(disease_phenotype$source_subject, 'MONDO:3')
  expect_equal(disease_phenotype$source_object, 'HP:0001250')
  gene_phenotype <- DBI::dbGetQuery(conn, "
    SELECT association_status, source_direction, gene_id, phenotype_id
    FROM semantic_monarch_gene_phenotype
    WHERE id = 'edge-gp-reverse'
  ")
  expect_equal(gene_phenotype$association_status, 'supported_object_to_subject')
  expect_equal(gene_phenotype$source_direction, 'object_to_subject')
  expect_equal(gene_phenotype$gene_id, 'HGNC:3')
  expect_equal(gene_phenotype$phenotype_id, 'HP:0004322')
  missing_role <- DBI::dbGetQuery(conn, "
    SELECT association_status, gene_id, phenotype_id
    FROM semantic_monarch_gene_phenotype
    WHERE id = 'edge-missing-role'
  ")
  expect_equal(missing_role$association_status, 'missing_role_category')
  expect_true(is.na(missing_role$gene_id))
  expect_true(is.na(missing_role$phenotype_id))

  expect_equal(DBI::dbGetQuery(conn, "SELECT provider_id, release_id FROM semantic_monarch_nodes")$provider_id, 'infores:monarchinitiative')
  expect_equal(DBI::dbGetQuery(conn, "SELECT provider_id, release_id FROM semantic_monarch_node_has_phenotype")$release_id, '2026-07-14')
  expect_equal(DBI::dbGetQuery(conn, "SELECT term, ic FROM semantic_monarch_information_content")$term, 'HP:0001250')
  expect_error(
    ducksemantics_monarch_project_pack(
      conn, 'monarch_pack', releases,
      provider_id = 'infores:monarchinitiative', release_id = '2026-08-13',
      prefix = 'unknown_release'
    ),
    'exactly one typed Monarch release catalog row'
  )
  expect_false(DBI::dbExistsTable(conn, 'unknown_release_associations'))
  expect_error(
    ducksemantics_monarch_project_pack(
      conn, 'monarch_pack', releases,
      provider_id = 'infores:monarchinitiative', release_id = 'latest'
    ),
    'latest'
  )
  expect_error(
    ducksemantics_monarch_project_pack(
      conn, 'monarch_pack; DROP TABLE monarch_pack.nodes', releases,
      provider_id = 'infores:monarchinitiative', release_id = '2026-07-14'
    ),
    'source_catalog'
  )
  expect_error(
    ducksemantics_monarch_project_pack(
      conn, 'monarch_pack', releases,
      provider_id = 'infores:monarchinitiative', release_id = '2026-07-14',
      prefix = 'not-safe!'
    ),
    'prefix'
  )

  projected_again <- ducksemantics_monarch_project_pack(
    conn, 'monarch_pack', releases,
    provider_id = 'infores:monarchinitiative', release_id = '2026-07-14'
  )
  expect_equal(projected_again, projected)
  temporary_views <- DBI::dbGetQuery(conn, "
    SELECT table_name, table_type
    FROM information_schema.tables
    WHERE table_catalog = 'temp'
      AND table_name LIKE 'semantic_monarch_%'
    ORDER BY table_name
  ")
  expect_equal(nrow(temporary_views), 7L)
  expect_true(all(temporary_views$table_type == 'VIEW'))
  expect_equal(DBI::dbGetQuery(conn, "SELECT COUNT(*) AS n FROM semantic_monarch_associations")$n[[1L]], source_rows)

  trimmed_path <- tempfile(fileext = ".duckdb")
  on.exit(unlink(trimmed_path), add = TRUE)
  DBI::dbExecute(conn, glue::glue_sql(
    "ATTACH {trimmed_path} AS trimmed_pack", .con = conn
  ))
  DBI::dbExecute(conn, "
    CREATE TABLE trimmed_pack.edge_source AS
    SELECT * EXCLUDE (stage_qualifier) FROM monarch_pack.denormalized_edges
  ")
  DBI::dbExecute(conn, "
    CREATE VIEW trimmed_pack.denormalized_edges AS
    SELECT * FROM trimmed_pack.edge_source
  ")
  for (table_name in c("nodes", "information_content", "node_has_phenotype")) {
    DBI::dbExecute(conn, glue::glue_sql(
      "CREATE TABLE trimmed_pack.{`table_name`} AS
       SELECT * FROM monarch_pack.main.{`table_name`}",
      .con = conn
    ))
  }
  DBI::dbExecute(conn, "DETACH trimmed_pack")
  DBI::dbExecute(conn, glue::glue_sql(
    "ATTACH {trimmed_path} AS trimmed_pack (READ_ONLY)", .con = conn
  ))
  expect_error(
    ducksemantics_monarch_project_pack(
      conn, 'trimmed_pack', releases,
      provider_id = 'infores:monarchinitiative', release_id = '2026-07-14',
      prefix = 'trimmed_projection'
    ),
    'stage_qualifier'
  )
  expect_false(DBI::dbExistsTable(conn, 'trimmed_projection_associations'))

  DBI::dbExecute(conn, "ATTACH ':memory:' AS mutable_pack")
  expect_error(
    ducksemantics_monarch_project_pack(
      conn, 'mutable_pack', releases,
      provider_id = 'infores:monarchinitiative', release_id = '2026-07-14'
    ),
    'read-only DuckDB catalog'
  )

  malformed_path <- tempfile(fileext = ".duckdb")
  on.exit(unlink(malformed_path), add = TRUE)
  DBI::dbExecute(conn, glue::glue_sql(
    "ATTACH {malformed_path} AS malformed_pack", .con = conn
  ))
  DBI::dbExecute(conn, "CREATE TABLE malformed_pack.denormalized_edges (id VARCHAR)")
  DBI::dbExecute(conn, "CREATE TABLE malformed_pack.nodes (id VARCHAR)")
  DBI::dbExecute(conn, "CREATE TABLE malformed_pack.information_content (id VARCHAR)")
  DBI::dbExecute(conn, "CREATE TABLE malformed_pack.node_has_phenotype (id VARCHAR)")
  DBI::dbExecute(conn, "DETACH malformed_pack")
  DBI::dbExecute(conn, glue::glue_sql(
    "ATTACH {malformed_path} AS malformed_pack (READ_ONLY)", .con = conn
  ))
  expect_error(
    ducksemantics_monarch_project_pack(
      conn, 'malformed_pack', releases,
      provider_id = 'infores:monarchinitiative', release_id = '2026-07-14',
      prefix = 'malformed_projection'
    ),
    'missing required official relations or relation types'
  )
  expect_false(DBI::dbExistsTable(conn, 'malformed_projection_associations'))
}
