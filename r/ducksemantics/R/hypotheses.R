#' Validate reviewed semantic hypothesis relations
#'
#' Hypotheses remain separate from authoritative source relations. Each
#' non-abstaining hypothesis must cite an exact span in a versioned source field.
#'
#' @param hypotheses A data frame with one row per hypothesis observation.
#' @param sources A data frame linking hypotheses to exact source spans.
#' @param source_fields A versioned source-field relation emitted by a source
#'   import.
#'
#' @return A named list containing normalized `hypotheses` and `sources` data
#'   frames.
#' @export
#'
#' @examples
#' fields <- data.frame(
#'   provider_id = "fixture", release_id = "fixture-1",
#'   source_id = "fixture-records", source_row_id = "record-1",
#'   field_path = "mechanism",
#'   value = "Reduced fibrillin changes extracellular matrix signaling.",
#'   stringsAsFactors = FALSE
#' )
#' hypothesis <- ducksemantics_hypothesis(
#'   hypothesis_id = "hypothesis-1", run_id = "run-1", case_id = "case-1",
#'   target_type = "gene", candidate_id = "HGNC:3603",
#'   proposal_type = "candidate",
#'   claim_text = "FBN1 disruption alters extracellular matrix signaling.",
#'   mechanism_text = "Reduced extracellular matrix signaling",
#'   proposer_type = "model", proposer_id = "fixture-model",
#'   proposer_version = "1", confidence = 0.9, judgment = "supports",
#'   review_status = "accepted", reviewer_id = "reviewer-1",
#'   reviewed_at = as.POSIXct("2026-01-02", tz = "UTC"),
#'   review_note = "The exact source span supports the proposed mechanism.",
#'   recorded_at = as.POSIXct("2026-01-01", tz = "UTC")
#' )
#' source <- ducksemantics_hypothesis_source(
#'   "hypothesis-1", fields, "fixture_mechanisms", "mechanism",
#'   text = "fibrillin changes extracellular matrix",
#'   citation = "fixture:record-1"
#' )
#' ducksemantics_hypotheses(hypothesis, source, fields)
ducksemantics_hypotheses <- function(hypotheses, sources, source_fields) {
  hypotheses <- .ducksemantics_validate_hypothesis_rows(hypotheses)
  sources <- .ducksemantics_validate_hypothesis_sources(
    sources,
    hypotheses,
    source_fields
  )

  abstained <- hypotheses$judgment == "abstain"
  source_counts <- tabulate(
    match(sources$hypothesis_id, hypotheses$hypothesis_id),
    nbins = nrow(hypotheses)
  )
  if (any(source_counts[abstained] != 0L)) {
    stop("abstaining hypotheses cannot cite source spans", call. = FALSE)
  }
  if (any(source_counts[!abstained] == 0L)) {
    stop("each non-abstaining hypothesis must cite a source span", call. = FALSE)
  }

  roles <- split(sources$evidence_role, sources$hypothesis_id)
  required_role <- c(
    supports = "support",
    contradicts = "contradiction",
    insufficient_evidence = "context"
  )
  observed <- hypotheses$judgment != "abstain"
  role_ok <- vapply(
    hypotheses$hypothesis_id[observed],
    function(id) required_role[[hypotheses$judgment[match(id, hypotheses$hypothesis_id)]]] %in%
      roles[[id]],
    logical(1)
  )
  if (!all(role_ok)) {
    stop("hypothesis evidence roles do not match their judgments", call. = FALSE)
  }

  list(hypotheses = hypotheses, sources = sources)
}

.ducksemantics_hypothesis_columns <- c(
  "hypothesis_id", "run_id", "case_id", "target_type", "candidate_id",
  "proposal_type", "claim_text", "inheritance_id", "mechanism_text",
  "proposer_type", "proposer_id", "proposer_version", "confidence",
  "judgment", "missing_evidence", "contradiction", "review_status",
  "reviewer_id", "reviewed_at", "review_note", "supersedes_id",
  "recorded_at"
)

.ducksemantics_hypothesis_source_columns <- c(
  "hypothesis_id", "provider_id", "release_id", "source_id",
  "source_relation", "source_record_id", "source_field", "start_offset",
  "end_offset", "source_span", "citation", "evidence_role"
)

#' Construct hypothesis observations
#'
#' Constructs the exact typed relation consumed by
#' [ducksemantics_hypotheses()]. Semantic fields stay explicit; only fields
#' whose absence is meaningful default to typed missing values.
#'
#' @param hypothesis_id,run_id,case_id,target_type,candidate_id Character
#'   identity columns.
#' @param proposal_type,claim_text,proposer_type,proposer_id,proposer_version
#'   Character proposal columns.
#' @param judgment,review_status Character judgment and review states.
#' @param recorded_at POSIXct observation timestamp.
#' @param confidence Numeric confidence, or `NA` for abstention.
#' @param inheritance_id,mechanism_text,missing_evidence,contradiction Optional
#'   evidence descriptions.
#' @param reviewer_id,review_note,supersedes_id Optional review metadata.
#' @param reviewed_at POSIXct review timestamp, or missing when unreviewed.
#' @return A validated 22-column hypothesis relation.
#' @export
#'
#' @examples
#' ducksemantics_hypothesis(
#'   hypothesis_id = "h1", run_id = "run-1", case_id = "case-1",
#'   target_type = "gene", candidate_id = "HGNC:3603",
#'   proposal_type = "candidate", claim_text = "FBN1 causes Marfan syndrome.",
#'   mechanism_text = "Defective fibrillin-1", proposer_type = "human",
#'   proposer_id = "author", proposer_version = "1", confidence = 0.9,
#'   judgment = "supports", review_status = "accepted",
#'   reviewer_id = "reviewer",
#'   reviewed_at = as.POSIXct("2026-01-02", tz = "UTC"),
#'   recorded_at = as.POSIXct("2026-01-01", tz = "UTC")
#' )
ducksemantics_hypothesis <- function(
  hypothesis_id, run_id, case_id, target_type, candidate_id, proposal_type,
  claim_text, proposer_type, proposer_id, proposer_version, judgment,
  review_status, recorded_at, confidence = NA_real_,
  inheritance_id = NA_character_, mechanism_text = NA_character_,
  missing_evidence = NA_character_, contradiction = NA_character_,
  reviewer_id = NA_character_,
  reviewed_at = .POSIXct(NA_real_, tz = "UTC"),
  review_note = NA_character_, supersedes_id = NA_character_
) {
  .ducksemantics_validate_hypothesis_rows(data.frame(
    hypothesis_id = hypothesis_id,
    run_id = run_id,
    case_id = case_id,
    target_type = target_type,
    candidate_id = candidate_id,
    proposal_type = proposal_type,
    claim_text = claim_text,
    inheritance_id = inheritance_id,
    mechanism_text = mechanism_text,
    proposer_type = proposer_type,
    proposer_id = proposer_id,
    proposer_version = proposer_version,
    confidence = confidence,
    judgment = judgment,
    missing_evidence = missing_evidence,
    contradiction = contradiction,
    review_status = review_status,
    reviewer_id = reviewer_id,
    reviewed_at = reviewed_at,
    review_note = review_note,
    supersedes_id = supersedes_id,
    recorded_at = recorded_at,
    stringsAsFactors = FALSE
  ))
}

#' Construct an exact hypothesis source span
#'
#' Selects one versioned source field and locates a unique fixed substring.
#' Character offsets are zero-based and half-open. The returned span is copied from the
#' source relation; caller text cannot replace source bytes.
#'
#' @param hypothesis_id Hypothesis identity.
#' @param source_fields Versioned source-field relation, scoped to one source
#'   release when field paths repeat across releases.
#' @param source_relation Name of the source relation.
#' @param field_path Field path to cite.
#' @param text Unique fixed substring to cite. `NULL` cites the complete field.
#' @param citation Citation attached to the source receipt.
#' @param evidence_role One of `support`, `contradiction`, or `context`.
#' @param source_row_id Optional source-row identity used to disambiguate a
#'   repeated field path.
#' @return One row with the exact hypothesis-source schema.
#' @export
#'
#' @examples
#' fields <- data.frame(
#'   provider_id = "fixture", release_id = "fixture-1",
#'   source_id = "fixture-records", source_row_id = "record-1",
#'   field_path = "mechanism", value = "FBN1 causes Marfan syndrome.",
#'   stringsAsFactors = FALSE
#' )
#' ducksemantics_hypothesis_source(
#'   "h1", fields, "fixture_mechanisms", "mechanism",
#'   text = "causes Marfan", citation = "fixture:record-1"
#' )
ducksemantics_hypothesis_source <- function(
  hypothesis_id, source_fields, source_relation, field_path, text = NULL,
  citation, evidence_role = c("support", "contradiction", "context"),
  source_row_id = NULL
) {
  hypothesis_id <- ducksemantics_require_text(hypothesis_id, "hypothesis_id")
  source_relation <- ducksemantics_require_text(source_relation, "source_relation")
  field_path <- ducksemantics_require_text(field_path, "field_path")
  citation <- ducksemantics_require_text(citation, "citation")
  evidence_role <- match.arg(evidence_role)
  if (!is.null(source_row_id)) {
    source_row_id <- ducksemantics_require_text(source_row_id, "source_row_id")
  }

  columns <- c(
    "provider_id", "release_id", "source_id", "source_row_id",
    "field_path", "value"
  )
  fields <- .ducksemantics_require_relation_columns(
    source_fields, columns, "source_fields", allow_extra = TRUE
  )
  .ducksemantics_require_character_columns(fields, columns, "source_fields")
  selected <- !is.na(fields$field_path) & fields$field_path == field_path
  if (!is.null(source_row_id)) {
    selected <- selected & !is.na(fields$source_row_id) &
      fields$source_row_id == source_row_id
  }
  if (sum(selected) != 1L) {
    stop("field_path must resolve to exactly one versioned source field",
         call. = FALSE)
  }
  field <- fields[selected, , drop = FALSE]
  if (!all(vapply(field, .ducksemantics_nonblank, logical(1)))) {
    stop("the selected source field must contain nonblank text and receipt keys",
         call. = FALSE)
  }
  value <- field$value[[1L]]

  if (is.null(text)) {
    start_offset <- 0L
    end_offset <- nchar(value, type = "chars")
    source_span <- value
  } else {
    text <- ducksemantics_require_text(text, "text")
    start <- regexpr(text, value, fixed = TRUE)[[1L]]
    repeated <- start > 0L &&
      grepl(text, substring(value, start + 1L), fixed = TRUE)
    if (start < 1L || repeated) {
      stop("text must occur exactly once in the selected source field",
           call. = FALSE)
    }
    start_offset <- as.integer(start - 1L)
    end_offset <- as.integer(start_offset + nchar(text, type = "chars"))
    source_span <- substring(value, start_offset + 1L, end_offset)
  }

  data.frame(
    hypothesis_id = hypothesis_id,
    provider_id = field$provider_id,
    release_id = field$release_id,
    source_id = field$source_id,
    source_relation = source_relation,
    source_record_id = field$source_row_id,
    source_field = field$field_path,
    start_offset = start_offset,
    end_offset = end_offset,
    source_span = source_span,
    citation = citation,
    evidence_role = evidence_role,
    stringsAsFactors = FALSE
  )
}

.ducksemantics_nonblank <- function(x) {
  !is.na(x) & nzchar(trimws(x))
}

.ducksemantics_require_relation_columns <- function(
  x, columns, name, allow_extra = FALSE
) {
  if (!is.data.frame(x)) {
    stop(name, " must be a data frame", call. = FALSE)
  }
  if (anyDuplicated(names(x))) {
    stop(name, " column names must be unique", call. = FALSE)
  }
  missing <- setdiff(columns, names(x))
  extra <- if (isTRUE(allow_extra)) character() else setdiff(names(x), columns)
  if (length(missing) > 0L || length(extra) > 0L) {
    details <- c(
      if (length(missing) > 0L) paste0("missing: ", paste(missing, collapse = ", ")),
      if (length(extra) > 0L) paste0("unexpected: ", paste(extra, collapse = ", "))
    )
    stop(name, " columns are invalid (", paste(details, collapse = "; "), ")",
         call. = FALSE)
  }
  x[columns]
}

.ducksemantics_require_character_columns <- function(x, columns, name) {
  invalid <- columns[!vapply(x[columns], is.character, logical(1))]
  if (length(invalid) > 0L) {
    stop(name, " columns must be character: ", paste(invalid, collapse = ", "),
         call. = FALSE)
  }
}

.ducksemantics_validate_hypothesis_rows <- function(hypotheses) {
  hypotheses <- .ducksemantics_require_relation_columns(
    hypotheses,
    .ducksemantics_hypothesis_columns,
    "hypotheses"
  )
  character_columns <- setdiff(
    .ducksemantics_hypothesis_columns,
    c("confidence", "reviewed_at", "recorded_at")
  )
  .ducksemantics_require_character_columns(
    hypotheses,
    character_columns,
    "hypothesis"
  )
  if (!is.numeric(hypotheses$confidence)) {
    stop("hypothesis confidence must be numeric", call. = FALSE)
  }
  if (!inherits(hypotheses$reviewed_at, "POSIXct") ||
      !inherits(hypotheses$recorded_at, "POSIXct")) {
    stop("hypothesis timestamps must be POSIXct", call. = FALSE)
  }

  .ducksemantics_validate_hypothesis_identity(hypotheses)
  .ducksemantics_validate_hypothesis_evidence(hypotheses)
  .ducksemantics_validate_hypothesis_review(hypotheses)

  rownames(hypotheses) <- NULL
  hypotheses
}

.ducksemantics_validate_hypothesis_identity <- function(hypotheses) {
  required <- c(
    "hypothesis_id", "run_id", "case_id", "target_type", "candidate_id",
    "proposal_type", "claim_text", "proposer_type", "proposer_id",
    "proposer_version", "judgment", "review_status"
  )
  invalid_required <- required[vapply(
    hypotheses[required],
    function(x) !all(.ducksemantics_nonblank(x)),
    logical(1)
  )]
  if (length(invalid_required) > 0L) {
    stop("hypothesis fields must be nonblank: ",
         paste(invalid_required, collapse = ", "), call. = FALSE)
  }
  if (anyDuplicated(hypotheses$hypothesis_id) > 0L) {
    stop("hypothesis_id must be unique", call. = FALSE)
  }
  if (anyNA(hypotheses$recorded_at)) {
    stop("recorded_at is required", call. = FALSE)
  }

  allowed <- list(
    proposal_type = c("candidate", "refinement"),
    proposer_type = c("model", "human"),
    judgment = c("supports", "contradicts", "insufficient_evidence", "abstain"),
    review_status = c("proposed", "accepted", "revised", "rejected", "abstained")
  )
  for (field in names(allowed)) {
    if (!all(hypotheses[[field]] %in% allowed[[field]])) {
      stop("unsupported ", field, call. = FALSE)
    }
  }

  optional_text <- c(
    "inheritance_id", "mechanism_text", "missing_evidence", "contradiction",
    "reviewer_id", "review_note", "supersedes_id"
  )
  invalid_optional <- optional_text[vapply(
    hypotheses[optional_text],
    function(x) any(!is.na(x) & !.ducksemantics_nonblank(x)),
    logical(1)
  )]
  if (length(invalid_optional) > 0L) {
    stop("optional hypothesis fields cannot be blank: ",
         paste(invalid_optional, collapse = ", "), call. = FALSE)
  }
  invisible(TRUE)
}

.ducksemantics_validate_hypothesis_evidence <- function(hypotheses) {
  abstained <- hypotheses$judgment == "abstain"
  if (any(abstained != (hypotheses$review_status == "abstained"))) {
    stop("abstention requires judgment='abstain' and review_status='abstained'",
         call. = FALSE)
  }
  if (!all(is.na(hypotheses$confidence[abstained]))) {
    stop("abstaining hypotheses must not report confidence", call. = FALSE)
  }
  confidence <- hypotheses$confidence[!abstained]
  if (any(!is.finite(confidence) | confidence < 0 | confidence > 1)) {
    stop("non-abstaining confidence must be finite and between zero and one",
         call. = FALSE)
  }
  described <- .ducksemantics_nonblank(hypotheses$inheritance_id) |
    .ducksemantics_nonblank(hypotheses$mechanism_text)
  if (any(!abstained & !described)) {
    stop("non-abstaining hypotheses require inheritance_id or mechanism_text",
         call. = FALSE)
  }
  if (any(hypotheses$judgment == "insufficient_evidence" &
          !.ducksemantics_nonblank(hypotheses$missing_evidence))) {
    stop("insufficient-evidence hypotheses require missing_evidence",
         call. = FALSE)
  }
  if (any(hypotheses$judgment == "contradicts" &
          !.ducksemantics_nonblank(hypotheses$contradiction))) {
    stop("contradicting hypotheses require contradiction", call. = FALSE)
  }
  invisible(TRUE)
}

.ducksemantics_validate_hypothesis_review <- function(hypotheses) {
  reviewed <- hypotheses$review_status %in% c("accepted", "revised", "rejected")
  reviewer_present <- .ducksemantics_nonblank(hypotheses$reviewer_id)
  reviewed_at_present <- !is.na(hypotheses$reviewed_at)
  if (any(reviewed & !(reviewer_present & reviewed_at_present))) {
    stop("reviewed hypotheses require reviewer_id and reviewed_at", call. = FALSE)
  }
  if (any(!reviewed & (reviewer_present | reviewed_at_present))) {
    stop("unreviewed hypotheses cannot contain reviewer metadata", call. = FALSE)
  }
  revised <- hypotheses$review_status == "revised"
  if (any(revised & !.ducksemantics_nonblank(hypotheses$supersedes_id))) {
    stop("revised hypotheses require supersedes_id", call. = FALSE)
  }
  if (any(!revised & !is.na(hypotheses$supersedes_id))) {
    stop("supersedes_id is only valid for revised hypotheses", call. = FALSE)
  }
  if (any(revised & hypotheses$supersedes_id == hypotheses$hypothesis_id)) {
    stop("a revised hypothesis cannot supersede itself", call. = FALSE)
  }
  invisible(TRUE)
}

.ducksemantics_validate_hypothesis_sources <- function(sources, hypotheses,
                                                        source_fields) {
  sources <- .ducksemantics_require_relation_columns(
    sources,
    .ducksemantics_hypothesis_source_columns,
    "sources"
  )
  character_columns <- setdiff(
    .ducksemantics_hypothesis_source_columns,
    c("start_offset", "end_offset")
  )
  .ducksemantics_require_character_columns(sources, character_columns, "source")
  if (!is.integer(sources$start_offset) || !is.integer(sources$end_offset)) {
    stop("source offsets must be integer", call. = FALSE)
  }

  required_text <- character_columns
  invalid_required <- required_text[vapply(
    sources[required_text],
    function(x) !all(.ducksemantics_nonblank(x)),
    logical(1)
  )]
  if (length(invalid_required) > 0L) {
    stop("source fields must be nonblank: ", paste(invalid_required, collapse = ", "),
         call. = FALSE)
  }
  if (!all(sources$evidence_role %in% c("support", "contradiction", "context"))) {
    stop("unsupported evidence_role", call. = FALSE)
  }
  if (!all(sources$hypothesis_id %in% hypotheses$hypothesis_id)) {
    stop("source links contain an unknown hypothesis_id", call. = FALSE)
  }
  if (any(sources$start_offset < 0L | sources$end_offset <= sources$start_offset)) {
    stop("source offsets must define a nonempty zero-based half-open span",
         call. = FALSE)
  }

  source_key <- do.call(
    paste,
    c(sources[c(
      "hypothesis_id", "provider_id", "release_id", "source_id",
      "source_record_id", "source_field", "start_offset", "end_offset"
    )], sep = "\r")
  )
  if (anyDuplicated(source_key) > 0L) {
    stop("hypothesis source links must be unique", call. = FALSE)
  }

  source_fields <- .ducksemantics_require_relation_columns(
    source_fields,
    c("provider_id", "release_id", "source_id", "source_row_id", "field_path", "value"),
    "source_fields",
    allow_extra = TRUE
  )
  .ducksemantics_require_character_columns(
    source_fields,
    names(source_fields),
    "source_fields"
  )
  field_key <- do.call(
    paste,
    c(source_fields[c(
      "provider_id", "release_id", "source_id", "source_row_id", "field_path"
    )], sep = "\r")
  )
  if (anyDuplicated(field_key) > 0L) {
    stop("source_fields keys must be unique", call. = FALSE)
  }
  link_key <- do.call(
    paste,
    c(sources[c(
      "provider_id", "release_id", "source_id", "source_record_id", "source_field"
    )], sep = "\r")
  )
  field_row <- match(link_key, field_key)
  if (anyNA(field_row)) {
    stop("hypothesis source links must resolve to source_fields", call. = FALSE)
  }

  value <- source_fields$value[field_row]
  if (any(sources$end_offset > nchar(value, type = "chars"))) {
    stop("source span exceeds the versioned source field", call. = FALSE)
  }
  expected <- substring(
    value,
    first = sources$start_offset + 1L,
    last = sources$end_offset
  )
  if (any(expected != sources$source_span)) {
    stop("source_span must exactly match the versioned source field",
         call. = FALSE)
  }

  rownames(sources) <- NULL
  sources
}
