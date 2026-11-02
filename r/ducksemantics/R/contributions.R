#' Validate a reviewed hypothesis contribution
#'
#' Builds a reviewable contribution artifact from one accepted or revised,
#' source-grounded hypothesis. The function validates a bounded target-specific
#' payload against pinned upstream field names and enums. It does not publish the
#' contribution or replace the upstream project's own validator.
#'
#' @param hypotheses DuckSemantics hypothesis observations.
#' @param sources DuckSemantics hypothesis source links.
#' @param source_fields Versioned source fields used to validate exact spans.
#' @param hypothesis_id The hypothesis to export.
#' @param target One of `"mondo"`, `"dismech"`, or `"monarch"`.
#' @param payload A target-specific contribution payload.
#'
#' @return A named list containing the reviewed hypothesis, its source receipts,
#'   the validated payload, and the pinned schema receipt.
#' @export
ducksemantics_contribution <- function(hypotheses, sources, source_fields,
                                       hypothesis_id,
                                       target = c("mondo", "dismech", "monarch"),
                                       payload) {
  target <- match.arg(target)
  if (!is.character(hypothesis_id) || length(hypothesis_id) != 1L ||
      !.ducksemantics_nonblank(hypothesis_id)) {
    stop("hypothesis_id must be one nonblank string", call. = FALSE)
  }
  bundle <- ducksemantics_hypotheses(hypotheses, sources, source_fields)
  row <- match(hypothesis_id, bundle$hypotheses$hypothesis_id)
  if (is.na(row)) {
    stop("hypothesis_id is not present", call. = FALSE)
  }
  hypothesis <- bundle$hypotheses[row, , drop = FALSE]
  receipts <- bundle$sources[
    bundle$sources$hypothesis_id == hypothesis_id,
    ,
    drop = FALSE
  ]
  if (!hypothesis$review_status %in% c("accepted", "revised")) {
    stop("only accepted or revised hypotheses can become contributions",
         call. = FALSE)
  }
  if (hypothesis$judgment != "supports") {
    stop("contributions require a supporting judgment", call. = FALSE)
  }
  if (!is.na(hypothesis$missing_evidence) || !is.na(hypothesis$contradiction)) {
    stop("unresolved missing evidence or contradiction blocks contribution",
         call. = FALSE)
  }
  if (!any(receipts$evidence_role == "support")) {
    stop("contributions require a supporting source span", call. = FALSE)
  }

  validated <- switch(
    target,
    mondo = .ducksemantics_mondo_contribution(payload, receipts),
    dismech = .ducksemantics_dismech_contribution(payload, receipts),
    monarch = .ducksemantics_monarch_contribution(payload, receipts)
  )

  list(
    target = target,
    hypothesis = hypothesis,
    sources = receipts,
    payload = validated$payload,
    schema = validated$schema
  )
}

.ducksemantics_contribution_schema <- function(target, uri, revision, path,
                                                scope) {
  data.frame(
    target = target,
    schema_uri = uri,
    schema_revision = revision,
    schema_path = path,
    validation_scope = scope,
    stringsAsFactors = FALSE
  )
}

.ducksemantics_scalar_text <- function(x, field, optional = FALSE) {
  valid <- is.character(x) && length(x) == 1L &&
    (if (optional) is.na(x) || .ducksemantics_nonblank(x) else .ducksemantics_nonblank(x))
  if (!valid) {
    stop(field, " must be one ", if (optional) "nonblank string or NA" else "nonblank string",
         call. = FALSE)
  }
  x
}

.ducksemantics_named_list <- function(x, required, allowed, name) {
  names_present <- !is.null(names(x))
  names_valid <- names_present && all(nzchar(names(x))) &&
    anyDuplicated(names(x)) == 0L
  if (!is.list(x) || !names_valid) {
    stop(name, " must be a uniquely named list", call. = FALSE)
  }
  missing <- setdiff(required, names(x))
  extra <- setdiff(names(x), allowed)
  if (length(missing) > 0L || length(extra) > 0L) {
    details <- c(
      if (length(missing) > 0L) paste0("missing: ", paste(missing, collapse = ", ")),
      if (length(extra) > 0L) paste0("unexpected: ", paste(extra, collapse = ", "))
    )
    stop(name, " fields are invalid (", paste(details, collapse = "; "), ")",
         call. = FALSE)
  }
  x
}

.ducksemantics_contains_citation <- function(text, citations) {
  any(vapply(citations, function(citation) grepl(citation, text, fixed = TRUE), logical(1)))
}

.ducksemantics_mondo_contribution <- function(payload, sources) {
  payload <- .ducksemantics_named_list(
    payload,
    required = "contribution_type",
    allowed = c(
      "contribution_type", "Label", "description", "parent", "synonyms",
      "synonym-type", "children", "orcid", "website_url",
      "additional_information", "term", "suggested_revision"
    ),
    name = "Mondo payload"
  )
  type <- .ducksemantics_scalar_text(payload$contribution_type, "contribution_type")
  if (!(type %in% c("new_term", "revise_definition"))) {
    stop("unsupported Mondo contribution_type", call. = FALSE)
  }
  if (type == "new_term") {
    payload <- .ducksemantics_named_list(
      payload,
      required = c("contribution_type", "Label", "description", "parent"),
      allowed = c(
        "contribution_type", "Label", "description", "parent", "synonyms",
        "synonym-type", "children", "orcid", "website_url",
        "additional_information"
      ),
      name = "Mondo new-term payload"
    )
    .ducksemantics_scalar_text(payload$Label, "Label")
    evidence_text <- .ducksemantics_scalar_text(payload$description, "description")
    parent <- .ducksemantics_scalar_text(payload$parent, "parent")
    if (!grepl("MONDO:[0-9]+", parent)) {
      stop("Mondo parent must contain a MONDO identifier", call. = FALSE)
    }
    path <- ".github/ISSUE_TEMPLATE/03add-term.yml"
  } else {
    payload <- .ducksemantics_named_list(
      payload,
      required = c("contribution_type", "term", "suggested_revision", "orcid"),
      allowed = c("contribution_type", "term", "suggested_revision", "orcid"),
      name = "Mondo definition-revision payload"
    )
    term <- .ducksemantics_scalar_text(payload$term, "term")
    if (!grepl("MONDO:[0-9]+", term)) {
      stop("Mondo term must contain a MONDO identifier", call. = FALSE)
    }
    evidence_text <- .ducksemantics_scalar_text(
      payload$suggested_revision,
      "suggested_revision"
    )
    path <- ".github/ISSUE_TEMPLATE/06revise-textual-definition.md"
  }
  citations <- unique(sources$citation[sources$evidence_role == "support"])
  if (!.ducksemantics_contains_citation(evidence_text, citations)) {
    stop("Mondo contribution text must include a supporting citation",
         call. = FALSE)
  }
  if (!is.null(payload$orcid) && !is.na(payload$orcid)) {
    orcid <- .ducksemantics_scalar_text(payload$orcid, "orcid", optional = TRUE)
    if (!grepl("^[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9X]{4}$", orcid)) {
      stop("orcid must use the four-group ORCID format", call. = FALSE)
    }
  }

  list(
    payload = payload,
    schema = .ducksemantics_contribution_schema(
      "mondo",
      "https://github.com/monarch-initiative/mondo",
      "bcff665c1db253380ef14f06fc390bc0b66420d0",
      path,
      "Pinned issue-template fields and supporting citation; upstream template review remains external and submission is not performed."
    )
  )
}

.ducksemantics_dismech_evidence <- function(evidence, sources) {
  required <- c(
    "reference", "supports", "evidence_source", "quote_role", "snippet",
    "explanation"
  )
  allowed <- c(required, "reference_title", "directness")
  evidence <- .ducksemantics_named_list(evidence, required, allowed, "DisMech evidence")
  reference <- .ducksemantics_scalar_text(evidence$reference, "reference")
  if (!identical(.ducksemantics_scalar_text(evidence$supports, "supports"), "SUPPORT")) {
    stop("accepted DisMech contributions require supports='SUPPORT'", call. = FALSE)
  }
  source_values <- c("HUMAN_CLINICAL", "MODEL_ORGANISM", "IN_VITRO", "COMPUTATIONAL", "OTHER")
  if (!(.ducksemantics_scalar_text(evidence$evidence_source, "evidence_source") %in%
        source_values)) {
    stop("unsupported DisMech evidence_source", call. = FALSE)
  }
  quote_roles <- c("PRIMARY_RESULT", "BACKGROUND", "REVIEW_SYNTHESIS")
  if (!(.ducksemantics_scalar_text(evidence$quote_role, "quote_role") %in%
        quote_roles)) {
    stop("unsupported DisMech quote_role", call. = FALSE)
  }
  snippet <- .ducksemantics_scalar_text(evidence$snippet, "snippet")
  supporting <- sources$evidence_role == "support"
  pair_matches <- supporting & sources$citation == reference &
    sources$source_span == snippet
  if (!any(pair_matches)) {
    stop("DisMech evidence reference and snippet must match one supporting source receipt",
         call. = FALSE)
  }
  .ducksemantics_scalar_text(evidence$explanation, "explanation")
  evidence
}

.ducksemantics_dismech_pathophysiology <- function(entry, sources) {
  required <- c("name", "biological_scale", "description", "evidence")
  entry <- .ducksemantics_named_list(
    entry,
    required,
    c(required, "downstream", "mechanism", "phenotypes"),
    "DisMech pathophysiology entry"
  )
  .ducksemantics_scalar_text(entry$name, "pathophysiology name")
  scales <- c("MOLECULAR", "CELLULAR", "TISSUE", "ORGANISM")
  if (!(.ducksemantics_scalar_text(entry$biological_scale, "biological_scale") %in%
        scales)) {
    stop("unsupported DisMech biological_scale", call. = FALSE)
  }
  .ducksemantics_scalar_text(entry$description, "pathophysiology description")
  if (!is.list(entry$evidence) || length(entry$evidence) == 0L) {
    stop("DisMech pathophysiology evidence must be a nonempty list",
         call. = FALSE)
  }
  entry$evidence <- lapply(
    entry$evidence,
    .ducksemantics_dismech_evidence,
    sources = sources
  )
  entry
}

.ducksemantics_dismech_contribution <- function(payload, sources) {
  payload <- .ducksemantics_named_list(
    payload,
    c("name", "disease_term", "pathophysiology"),
    c("name", "disease_term", "pathophysiology"),
    "DisMech payload"
  )
  .ducksemantics_scalar_text(payload$name, "name")
  disease <- .ducksemantics_named_list(
    payload$disease_term,
    c("id", "label"),
    c("id", "label"),
    "DisMech disease_term"
  )
  if (!grepl("^MONDO:[0-9]+$", .ducksemantics_scalar_text(disease$id, "disease_term id"))) {
    stop("DisMech disease_term id must be a MONDO identifier", call. = FALSE)
  }
  .ducksemantics_scalar_text(disease$label, "disease_term label")
  if (!is.list(payload$pathophysiology) ||
      length(payload$pathophysiology) == 0L) {
    stop("DisMech pathophysiology must be a nonempty list", call. = FALSE)
  }
  payload$pathophysiology <- lapply(
    payload$pathophysiology,
    .ducksemantics_dismech_pathophysiology,
    sources = sources
  )

  list(
    payload = payload,
    schema = .ducksemantics_contribution_schema(
      "dismech",
      "https://github.com/monarch-initiative/dismech",
      "3e6cb8631b383355d9b590e5930a38f1813f9094",
      "src/dismech/schema/dismech.yaml",
      "Pinned disease, pathophysiology, and evidence fields/enums; run the upstream LinkML validator before submission."
    )
  )
}

.ducksemantics_monarch_contribution <- function(payload, sources) {
  columns <- c(
    "subject", "predicate", "object", "primary_knowledge_source",
    "publications", "has_evidence", "knowledge_level", "agent_type"
  )
  payload <- .ducksemantics_require_relation_columns(payload, columns, "Monarch payload")
  character_columns <- setdiff(columns, c("publications", "has_evidence"))
  .ducksemantics_require_character_columns(payload, character_columns, "Monarch payload")
  if (!is.list(payload$publications) || !is.list(payload$has_evidence)) {
    stop("Monarch publications and has_evidence must be list columns",
         call. = FALSE)
  }
  required_text <- character_columns
  scalar_fields_valid <- vapply(
    payload[required_text],
    function(x) all(.ducksemantics_nonblank(x)),
    logical(1)
  )
  if (!all(scalar_fields_valid)) {
    stop("Monarch scalar fields must be nonblank", call. = FALSE)
  }
  curie_pattern <- "^[A-Za-z][A-Za-z0-9_.-]*:[^[:space:]]+$"
  if (!all(grepl(curie_pattern, payload$subject)) ||
      !all(grepl(curie_pattern, payload$object))) {
    stop("Monarch subject and object must be CURIEs", call. = FALSE)
  }
  predicates <- c(
    "biolink:causes", "biolink:contributes_to",
    "biolink:gene_associated_with_condition", "biolink:has_phenotype"
  )
  if (!all(payload$predicate %in% predicates)) {
    stop("unsupported Monarch predicate", call. = FALSE)
  }
  if (!all(grepl(
    "^infores:[A-Za-z0-9_.-]+$",
    payload$primary_knowledge_source
  ))) {
    stop("Monarch primary_knowledge_source must be an infores CURIE",
         call. = FALSE)
  }
  knowledge_levels <- c(
    "knowledge_assertion", "logical_entailment", "prediction",
    "statistical_association"
  )
  agent_types <- c(
    "manual_agent", "automated_agent", "data_analysis_pipeline",
    "computational_model", "text_mining_agent", "image_processing_agent",
    "manual_validation_of_automated_agent"
  )
  if (!all(payload$knowledge_level %in% knowledge_levels)) {
    stop("unsupported Monarch knowledge_level", call. = FALSE)
  }
  if (!all(payload$agent_type %in% agent_types)) {
    stop("unsupported Monarch agent_type", call. = FALSE)
  }
  citations <- unique(sources$citation[sources$evidence_role == "support"])
  publication_ok <- vapply(
    payload$publications,
    function(x) is.character(x) && length(x) > 0L && any(x %in% citations),
    logical(1)
  )
  if (!all(publication_ok)) {
    stop("Monarch publications must include a supporting citation",
         call. = FALSE)
  }
  evidence_ok <- vapply(
    payload$has_evidence,
    function(x) is.character(x) && length(x) > 0L &&
      all(grepl(curie_pattern, x)),
    logical(1)
  )
  if (!all(evidence_ok)) {
    stop("Monarch has_evidence entries must be nonempty CURIE vectors",
         call. = FALSE)
  }

  list(
    payload = payload,
    schema = .ducksemantics_contribution_schema(
      "monarch",
      "https://github.com/biolink/biolink-model",
      "v4.4.6; monarch-ingest 289efde68e817beeced342a5c80ba91cf0d53116",
      "biolink-model.yaml; KGX edge TSV",
      "Pinned supported association predicates and provenance enums; run upstream Biolink/KGX validation before submission."
    )
  )
}
