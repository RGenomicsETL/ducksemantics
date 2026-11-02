local({
  args <- list(
    hypothesis_id = "h1", run_id = "run-1", case_id = "case-1",
    target_type = "gene", candidate_id = "HGNC:3603",
    proposal_type = "candidate", claim_text = "FBN1 defects can cause disease.",
    mechanism_text = "FBN1 defects", proposer_type = "human",
    proposer_id = "author", proposer_version = "1", confidence = 0.8,
    judgment = "supports", review_status = "proposed",
    recorded_at = as.POSIXct("2026-10-14", tz = "UTC")
  )
  hypothesis <- do.call(ducksemantics::ducksemantics_hypothesis, args)
  expect_equal(nrow(hypothesis), 1L)
  expect_equal(ncol(hypothesis), 22L)
  expect_true(is.character(hypothesis$inheritance_id))
  expect_true(is.na(hypothesis$inheritance_id))
  expect_true(inherits(hypothesis$reviewed_at, "POSIXct"))
  expect_true(is.na(hypothesis$reviewed_at))
  expect_error(
    do.call(ducksemantics::ducksemantics_hypothesis,
            modifyList(args, list(confidence = 1.1))),
    "between zero and one"
  )
  expect_error(
    do.call(ducksemantics::ducksemantics_hypothesis,
            modifyList(args, list(review_status = "accepted"))),
    "reviewer_id and reviewed_at"
  )
  expect_error(
    do.call(ducksemantics::ducksemantics_hypothesis,
            modifyList(args, list(recorded_at = "2026-10-14"))),
    "POSIXct"
  )

  fields <- data.frame(
    provider_id = "fixture", release_id = "release-1", source_id = "source-1",
    source_row_id = "record-1", field_path = "/description",
    value = "\u03b1 FBN1 \u03b2.", source_count = 1L,
    stringsAsFactors = FALSE
  )
  source <- ducksemantics::ducksemantics_hypothesis_source(
    "h1", fields, "fixture_fields", "/description", "FBN1", "fixture:record-1"
  )
  expect_equal(source$start_offset, 2L)
  expect_equal(source$end_offset, 6L)
  expect_true(is.integer(source$start_offset))
  expect_true(is.integer(source$end_offset))
  expect_equal(source$source_span, "FBN1")
  expect_equal(source$evidence_role, "support")
  bundle <- ducksemantics::ducksemantics_hypotheses(hypothesis, source, fields)
  expect_equal(bundle$sources, source)

  whole <- ducksemantics::ducksemantics_hypothesis_source(
    "h1", fields, "fixture_fields", "/description", citation = "fixture:record-1"
  )
  expect_equal(whole$start_offset, 0L)
  expect_equal(whole$end_offset, nchar(fields$value, type = "chars"))
  expect_equal(whole$source_span, fields$value)

  ambiguous <- rbind(fields, transform(fields, source_row_id = "record-2"))
  expect_error(
    ducksemantics::ducksemantics_hypothesis_source(
      "h1", ambiguous, "fixture_fields", "/description", "FBN1", "fixture:record-1"
    ),
    "exactly one versioned source field"
  )
  resolved <- ducksemantics::ducksemantics_hypothesis_source(
    "h1", ambiguous, "fixture_fields", "/description", "FBN1", "fixture:record-2",
    source_row_id = "record-2"
  )
  expect_equal(resolved$source_record_id, "record-2")
  expect_error(
    ducksemantics::ducksemantics_hypothesis_source(
      "h1", fields, "fixture_fields", "/missing", "FBN1", "fixture:record-1"
    ),
    "exactly one versioned source field"
  )
  expect_error(
    ducksemantics::ducksemantics_hypothesis_source(
      "h1", transform(fields, value = "aaa"), "fixture_fields", "/description",
      "aa", "fixture:record-1"
    ),
    "exactly once"
  )
  expect_error(
    ducksemantics::ducksemantics_hypothesis_source(
      "h1", transform(fields, value = NA_character_), "fixture_fields",
      "/description", citation = "fixture:record-1"
    ),
    "nonblank text and receipt keys"
  )
  expect_error(
    ducksemantics::ducksemantics_hypothesis_source(
      "h1", transform(fields, provider_id = NA_character_), "fixture_fields",
      "/description", citation = "fixture:record-1"
    ),
    "nonblank text and receipt keys"
  )
  expect_error(
    ducksemantics::ducksemantics_hypothesis_source(
      "h1", transform(fields, field_path = NA_character_), "fixture_fields",
      "/description", citation = "fixture:record-1"
    ),
    "exactly one versioned source field"
  )
  expect_error(
    ducksemantics::ducksemantics_hypothesis_source(
      "h1", transform(fields, value = 42L), "fixture_fields",
      "/description", citation = "fixture:record-1"
    ),
    "must be character"
  )
  duplicated <- cbind(fields, value = "other text")
  expect_error(
    ducksemantics::ducksemantics_hypothesis_source(
      "h1", duplicated, "fixture_fields", "/description",
      citation = "fixture:record-1"
    ),
    "column names must be unique"
  )
})
