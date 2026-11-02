local({
  field_text <- "Reduced fibrillin changes extracellular matrix signaling."
  source_fields <- data.frame(
    provider_id = "DisMech",
    release_id = "fixture-release",
    source_id = "dismech-fixture",
    source_row_id = "MONDO:0007947",
    field_path = "/pathophysiology/0/description",
    value = field_text,
    stringsAsFactors = FALSE
  )
  hypothesis <- data.frame(
    hypothesis_id = "hypothesis-1",
    run_id = "run-1",
    case_id = "case-1",
    target_type = "gene",
    candidate_id = "HGNC:3603",
    proposal_type = "candidate",
    claim_text = field_text,
    inheritance_id = "HP:0000006",
    mechanism_text = "Reduced extracellular matrix signaling",
    proposer_type = "model",
    proposer_id = "fixture-model",
    proposer_version = "1",
    confidence = 0.9,
    judgment = "supports",
    missing_evidence = NA_character_,
    contradiction = NA_character_,
    review_status = "accepted",
    reviewer_id = "reviewer-1",
    reviewed_at = as.POSIXct("2026-01-02", tz = "UTC"),
    review_note = "The cited span supports the proposed mechanism.",
    supersedes_id = NA_character_,
    recorded_at = as.POSIXct("2026-01-01", tz = "UTC"),
    stringsAsFactors = FALSE
  )
  source <- data.frame(
    hypothesis_id = "hypothesis-1",
    provider_id = "DisMech",
    release_id = "fixture-release",
    source_id = "dismech-fixture",
    source_relation = "dismech_fields",
    source_record_id = "MONDO:0007947",
    source_field = "/pathophysiology/0/description",
    start_offset = 0L,
    end_offset = nchar(field_text, type = "chars"),
    source_span = field_text,
    citation = "PMID:8180508",
    evidence_role = "support",
    stringsAsFactors = FALSE
  )

  bundle <- ducksemantics::ducksemantics_hypotheses(
    hypothesis,
    source,
    source_fields
  )
  expect_equal(names(bundle), c("hypotheses", "sources"))
  expect_equal(bundle$hypotheses$hypothesis_id, "hypothesis-1")
  expect_equal(bundle$sources$source_span, field_text)

  invalid <- source
  invalid$source_span <- paste0(invalid$source_span, " altered")
  expect_error(
    ducksemantics::ducksemantics_hypotheses(hypothesis, invalid, source_fields),
    "exactly match"
  )

  invalid <- source
  invalid$end_offset <- invalid$end_offset + 1L
  expect_error(
    ducksemantics::ducksemantics_hypotheses(hypothesis, invalid, source_fields),
    "exceeds"
  )

  invalid <- source
  invalid$release_id <- "unknown-release"
  expect_error(
    ducksemantics::ducksemantics_hypotheses(hypothesis, invalid, source_fields),
    "resolve"
  )

  invalid <- hypothesis
  invalid$judgment <- "contradicts"
  invalid$contradiction <- "The source reports the opposite direction."
  expect_error(
    ducksemantics::ducksemantics_hypotheses(invalid, source, source_fields),
    "evidence roles"
  )

  invalid <- hypothesis
  invalid$judgment <- "insufficient_evidence"
  expect_error(
    ducksemantics::ducksemantics_hypotheses(invalid, source, source_fields),
    "missing_evidence"
  )

  abstention <- hypothesis
  abstention$confidence <- NA_real_
  abstention$judgment <- "abstain"
  abstention$review_status <- "abstained"
  abstention$reviewer_id <- NA_character_
  abstention$reviewed_at <- as.POSIXct(NA, tz = "UTC")
  abstention$review_note <- NA_character_
  abstention$inheritance_id <- NA_character_
  abstention$mechanism_text <- NA_character_
  empty_sources <- source[FALSE, ]
  bundle <- ducksemantics::ducksemantics_hypotheses(
    abstention,
    empty_sources,
    source_fields
  )
  expect_equal(nrow(bundle$sources), 0L)

  expect_error(
    ducksemantics::ducksemantics_hypotheses(abstention, source, source_fields),
    "abstaining hypotheses"
  )

  invalid <- hypothesis
  invalid$review_status <- "revised"
  expect_error(
    ducksemantics::ducksemantics_hypotheses(invalid, source, source_fields),
    "supersedes_id"
  )

  invalid <- rbind(hypothesis, hypothesis)
  expect_error(
    ducksemantics::ducksemantics_hypotheses(invalid, source, source_fields),
    "unique"
  )

  mondo_payload <- list(
    contribution_type = "new_term",
    Label = "example fibrillin signaling disorder",
    description = paste(
      "A disorder caused by reduced extracellular matrix signaling.",
      "PMID:8180508"
    ),
    parent = "MONDO:0000001 disease",
    orcid = "0000-0002-1825-0097"
  )
  contribution <- ducksemantics::ducksemantics_contribution(
    hypothesis,
    source,
    source_fields,
    "hypothesis-1",
    "mondo",
    mondo_payload
  )
  expect_equal(contribution$schema$schema_revision,
    "bcff665c1db253380ef14f06fc390bc0b66420d0")

  dismech_payload <- list(
    name = "Marfan Syndrome",
    disease_term = list(id = "MONDO:0007947", label = "Marfan syndrome"),
    pathophysiology = list(list(
      name = "Reduced extracellular matrix signaling",
      biological_scale = "MOLECULAR",
      description = field_text,
      evidence = list(list(
        reference = "PMID:8180508",
        supports = "SUPPORT",
        evidence_source = "HUMAN_CLINICAL",
        quote_role = "PRIMARY_RESULT",
        snippet = field_text,
        explanation = "The exact source span supports the mechanism."
      ))
    ))
  )
  contribution <- ducksemantics::ducksemantics_contribution(
    hypothesis,
    source,
    source_fields,
    "hypothesis-1",
    "dismech",
    dismech_payload
  )
  expect_equal(contribution$schema$schema_revision,
    "3e6cb8631b383355d9b590e5930a38f1813f9094")

  monarch_payload <- data.frame(
    subject = "HGNC:3603",
    predicate = "biolink:causes",
    object = "MONDO:0007947",
    primary_knowledge_source = "infores:ducksemantics",
    knowledge_level = "knowledge_assertion",
    agent_type = "manual_validation_of_automated_agent",
    stringsAsFactors = FALSE
  )
  monarch_payload$publications <- I(list("PMID:8180508"))
  monarch_payload$has_evidence <- I(list("ECO:0000033"))
  contribution <- ducksemantics::ducksemantics_contribution(
    hypothesis,
    source,
    source_fields,
    "hypothesis-1",
    "monarch",
    monarch_payload
  )
  expect_equal(contribution$schema$schema_revision,
    "v4.4.6; monarch-ingest 289efde68e817beeced342a5c80ba91cf0d53116")

  invalid_payload <- mondo_payload
  invalid_payload$description <- "No source identifier is present."
  expect_error(
    ducksemantics::ducksemantics_contribution(
      hypothesis,
      source,
      source_fields,
      "hypothesis-1",
      "mondo",
      invalid_payload
    ),
    "supporting citation"
  )

  invalid_payload <- dismech_payload
  invalid_payload$pathophysiology[[1L]]$evidence[[1L]]$snippet <- "Paraphrase"
  expect_error(
    ducksemantics::ducksemantics_contribution(
      hypothesis,
      source,
      source_fields,
      "hypothesis-1",
      "dismech",
      invalid_payload
    ),
    "supporting source receipt"
  )

  pair_fields <- rbind(
    source_fields,
    transform(
      source_fields,
      field_path = "/pathophysiology/0/evidence/1/snippet",
      value = "A different supporting quotation."
    )
  )
  pair_sources <- rbind(
    source,
    transform(
      source,
      source_field = "/pathophysiology/0/evidence/1/snippet",
      end_offset = nchar("A different supporting quotation.", type = "chars"),
      source_span = "A different supporting quotation.",
      citation = "PMID:99999999"
    )
  )
  invalid_payload <- dismech_payload
  invalid_payload$pathophysiology[[1L]]$evidence[[1L]]$snippet <-
    "A different supporting quotation."
  expect_error(
    ducksemantics::ducksemantics_contribution(
      hypothesis,
      pair_sources,
      pair_fields,
      "hypothesis-1",
      "dismech",
      invalid_payload
    ),
    "one supporting source receipt"
  )

  invalid_payload <- monarch_payload
  invalid_payload$predicate <- "biolink:related_to"
  expect_error(
    ducksemantics::ducksemantics_contribution(
      hypothesis,
      source,
      source_fields,
      "hypothesis-1",
      "monarch",
      invalid_payload
    ),
    "predicate"
  )

  blocked <- hypothesis
  blocked$missing_evidence <- "A replication cohort is missing."
  expect_error(
    ducksemantics::ducksemantics_contribution(
      blocked,
      source,
      source_fields,
      "hypothesis-1",
      "mondo",
      mondo_payload
    ),
    "blocks contribution"
  )

  revised <- hypothesis
  revised$hypothesis_id <- "hypothesis-2"
  revised$proposal_type <- "refinement"
  revised$review_status <- "revised"
  revised$supersedes_id <- "hypothesis-1"
  revised_source <- source
  revised_source$hypothesis_id <- "hypothesis-2"
  contribution <- ducksemantics::ducksemantics_contribution(
    revised,
    revised_source,
    source_fields,
    "hypothesis-2",
    "mondo",
    mondo_payload
  )
  expect_equal(contribution$hypothesis$review_status, "revised")

  rejected <- hypothesis
  rejected$review_status <- "rejected"
  expect_equal(
    ducksemantics::ducksemantics_hypotheses(
      rejected,
      source,
      source_fields
    )$hypotheses$review_status,
    "rejected"
  )
  expect_error(
    ducksemantics::ducksemantics_contribution(
      rejected,
      source,
      source_fields,
      "hypothesis-1",
      "mondo",
      mondo_payload
    ),
    "accepted or revised"
  )

  proposed <- hypothesis
  proposed$review_status <- "proposed"
  proposed$reviewer_id <- NA_character_
  proposed$reviewed_at <- as.POSIXct(NA, tz = "UTC")
  proposed$review_note <- NA_character_
  expect_equal(
    ducksemantics::ducksemantics_hypotheses(
      proposed,
      source,
      source_fields
    )$hypotheses$review_status,
    "proposed"
  )
})
