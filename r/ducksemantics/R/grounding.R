#' HPO observation contract
#'
#' The final observation relation has exactly these required semantic fields:
#' `document_id`, `hpo_id`, `start_offset`, `end_offset`, `source_text`,
#' `context_status`, `method`, `provider_id`, `provider_version`,
#' `confidence`, and `status`. Offsets are zero-based, half-open R character
#' offsets. Lexical candidates returned by [ducksemantics_annotate()] are not
#' observations and cannot be substituted for this relation.
#'
#' @return Required HPO observation column names.
#' @export
ducksemantics_hpo_observation_contract <- function() {
  c(
    "document_id",
    "hpo_id",
    "start_offset",
    "end_offset",
    "source_text",
    "context_status",
    "method",
    "provider_id",
    "provider_version",
    "confidence",
    "status"
  )
}

#' Validate accepted HPO observations against source documents
#'
#' @param documents Data frame (or a caller-owned DuckDB table name with
#'   `conn`) containing `document_id` and `source_text`.
#' @param observations Data frame (or caller-owned DuckDB table name) following
#'   [ducksemantics_hpo_observation_contract()].
#' @param conn Optional DBI connection for a lazy caller-owned table name.
#' @return Validated accepted-observation data frame.
#' @export
ducksemantics_hpo_observations <- function(
  documents,
  observations,
  conn = NULL
) {
  documents <- ducksemantics_collect_relation(documents, conn, "documents")
  observations <- ducksemantics_collect_relation(
    observations,
    conn,
    "observations"
  )
  documents <- ducksemantics_require_data_frame(
    documents,
    "documents",
    c("document_id", "source_text")
  )
  observations <- ducksemantics_require_data_frame(
    observations,
    "observations",
    ducksemantics_hpo_observation_contract()
  )
  ducksemantics_validate_required_text(
    documents,
    c("document_id", "source_text"),
    "documents"
  )
  if (anyDuplicated(documents$document_id)) {
    stop(
      "`documents$document_id` must uniquely identify source documents.",
      call. = FALSE
    )
  }
  if (!nrow(observations)) {
    return(observations[,
      ducksemantics_hpo_observation_contract(),
      drop = FALSE
    ])
  }
  ducksemantics_validate_required_text(
    observations,
    c(
      "document_id",
      "hpo_id",
      "source_text",
      "context_status",
      "method",
      "provider_id",
      "provider_version",
      "status"
    ),
    "observations"
  )
  if (!all(grepl("^HP:[0-9]{7}$", observations$hpo_id))) {
    stop(
      "`observations$hpo_id` must contain HPO identifiers such as HP:0001250.",
      call. = FALSE
    )
  }
  if (
    !all(observations$context_status %in% ducksemantics_hpo_context_statuses())
  ) {
    stop(
      "`observations$context_status` must be one of: ",
      paste(ducksemantics_hpo_context_statuses(), collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  if (any(observations$status != "accepted")) {
    stop(
      "Final HPO observations must have explicit `status = \"accepted\"`; keep lexical candidates separately.",
      call. = FALSE
    )
  }
  ducksemantics_validate_half_open_offsets(
    observations$start_offset,
    observations$end_offset,
    "observations"
  )
  confidence <- suppressWarnings(as.numeric(observations$confidence))
  if (
    any(
      is.na(confidence) |
        !is.finite(confidence) |
        confidence < 0 |
        confidence > 1
    )
  ) {
    stop(
      "`observations$confidence` must contain finite values from 0 to 1.",
      call. = FALSE
    )
  }
  source <- documents$source_text[match(
    observations$document_id,
    documents$document_id
  )]
  if (anyNA(source)) {
    stop(
      "Every `observations$document_id` must occur in `documents`.",
      call. = FALSE
    )
  }
  lengths <- nchar(source, type = "chars")
  if (any(observations$end_offset > lengths)) {
    stop(
      "HPO observation span is out of bounds for its source document.",
      call. = FALSE
    )
  }
  exact <- mapply(
    function(text, start, end) substring(text, start + 1L, end),
    source,
    observations$start_offset,
    observations$end_offset,
    USE.NAMES = FALSE
  )
  if (any(exact != observations$source_text)) {
    stop(
      "HPO observation `source_text` must exactly equal its source-document span.",
      call. = FALSE
    )
  }
  observations$start_offset <- as.integer(observations$start_offset)
  observations$end_offset <- as.integer(observations$end_offset)
  observations$confidence <- confidence
  observations[, ducksemantics_hpo_observation_contract(), drop = FALSE]
}

ducksemantics_hpo_context_statuses <- function() {
  c(
    "present",
    "absent/negated",
    "family_history",
    "uncertain",
    "conflict",
    "unsupported"
  )
}

ducksemantics_collect_relation <- function(
  relation,
  conn = NULL,
  arg = "relation"
) {
  if (is.data.frame(relation)) {
    return(relation)
  }
  if (is.character(relation) && length(relation) == 1L && !is.na(relation)) {
    ducksemantics_require_connection(conn, "conn")
    table <- ducksemantics_require_identifier(relation, arg, qualified = TRUE)
    return(DBI::dbGetQuery(
      conn,
      paste0("SELECT * FROM ", ducksemantics_quote_ident(table))
    ))
  }
  stop(
    "`",
    arg,
    "` must be a data frame or a caller-owned DuckDB table name with `conn`.",
    call. = FALSE
  )
}

ducksemantics_validate_half_open_offsets <- function(start, end, arg) {
  valid <- is.numeric(start) &&
    is.numeric(end) &&
    length(start) == length(end) &&
    all(
      !is.na(start) &
        !is.na(end) &
        is.finite(start) &
        is.finite(end) &
        start >= 0 &
        end > start &
        start == floor(start) &
        end == floor(end)
    )
  if (!valid) {
    stop(
      "`",
      arg,
      "` spans must be non-empty zero-based half-open integer offsets.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' Default source-grounded judgment instructions
#' @return A character scalar.
#' @export
ducksemantics_default_judgment_instructions <- function() {
  paste(
    "You adjudicate deterministic semantic grounding candidates.",
    "Return one JSON object per supplied candidate in order with mention_id, decision, and optional confidence.",
    "Use drop for negated, uncertain, family-history-only, not-about-the-subject, duplicate, or unsupported candidates.",
    "Never invent identifiers; replacements must come from supplied candidates or graph context.",
    sep = "\n"
  )
}

#' Build a source-grounded judgment prompt
#' @param text Source text.
#' @param mentions Candidate data frame.
#' @param graph_context Optional graph context relation.
#' @param instructions Explicit judgment policy.
#' @return Prompt text.
#' @export
ducksemantics_judgment_prompt <- function(
  text,
  mentions,
  graph_context = NULL,
  instructions = ducksemantics_default_judgment_instructions()
) {
  ducksemantics_require_jsonlite()
  text <- ducksemantics_require_text(text, "text")
  mentions <- ducksemantics_require_data_frame(mentions, "mentions")
  instructions <- ducksemantics_prompt_text(instructions, "instructions")
  payload <- mentions[,
    intersect(
      c(
        "mention_id",
        "node_id",
        "span",
        "start_offset",
        "end_offset",
        "score",
        "alias",
        "alias_kind",
        "source"
      ),
      names(mentions)
    ),
    drop = FALSE
  ]
  paste(
    instructions,
    "",
    "TEXT:",
    text,
    "",
    "CANDIDATES_JSON:",
    jsonlite::toJSON(
      payload,
      auto_unbox = TRUE,
      null = "null",
      dataframe = "rows"
    ),
    "",
    "GRAPH_CONTEXT_JSON:",
    jsonlite::toJSON(
      graph_context %||% list(),
      auto_unbox = TRUE,
      null = "null",
      dataframe = "rows"
    ),
    sep = "\n"
  )
}

#' Create a BebeLM prompt provider
#' @param agent An `Rbebelm` agent.
#' @param on_event Optional event handler.
#' @return An object implementing [DucksemanticsPromptRunner].
#' @export
ducksemantics_bebel_runner <- function(agent, on_event = NULL) {
  if (!requireNamespace("Rbebelm", quietly = TRUE)) {
    stop("Rbebelm is required for BebeLM judgment.", call. = FALSE)
  }
  if (!is.null(on_event) && !is.function(on_event)) {
    stop("`on_event` must be NULL or a function.", call. = FALSE)
  }
  ducksemantics_bebel_runner_class(agent = agent, on_event = on_event)
}

#' Judge lexical candidates with a provider
#' @param text Source text.
#' @param mentions Candidate relation.
#' @param runner Prompt provider.
#' @param conn Optional DBI connection to record judgments.
#' @param prefix Table prefix.
#' @param graph_context Optional graph context.
#' @param instructions Explicit judgment instructions.
#' @param prompt_builder Function building the prompt.
#' @param parser Judgment parser provider.
#' @param record Record result when `conn` is supplied.
#' @param model Provider/model identity.
#' @param ... Arguments passed to `prompt_builder`.
#' @return Source-grounded judgment relation.
#' @export
ducksemantics_judge <- function(
  text,
  mentions,
  runner,
  conn = NULL,
  prefix = "semantic",
  graph_context = NULL,
  instructions = ducksemantics_default_judgment_instructions(),
  prompt_builder = ducksemantics_judgment_prompt,
  parser = ducksemantics_json_judgment_parser(),
  record = !is.null(conn),
  model = "semantic-runner",
  ...
) {
  s7contract::assert_implements(
    runner,
    DucksemanticsPromptRunner,
    arg = "runner"
  )
  s7contract::assert_implements(
    parser,
    DucksemanticsJudgmentParser,
    arg = "parser"
  )
  if (!is.function(prompt_builder)) {
    stop("`prompt_builder` must be a function.", call. = FALSE)
  }
  ducksemantics_require_flag(record, "record")
  model <- ducksemantics_require_text(model, "model")
  prompt <- prompt_builder(
    text = text,
    mentions = mentions,
    graph_context = graph_context,
    instructions = instructions,
    ...
  )
  response <- ducksemantics_run(runner, prompt)
  judgments <- ducksemantics_judgments_from_model(
    ducksemantics_parse(parser, response),
    mentions,
    model,
    ducksemantics_judgment_node_ids(mentions, graph_context)
  )
  attr(judgments, "prompt") <- prompt
  attr(judgments, "response") <- response
  if (record) {
    if (is.null(conn)) {
      stop("`conn` is required when `record = TRUE`.", call. = FALSE)
    }
    ducksemantics_record_judgments(conn, judgments, prefix)
  }
  judgments
}

#' Record source-grounded judgments
#' @param conn DBI connection.
#' @param judgments Judgment relation.
#' @param prefix Table prefix.
#' @return Invisibly, validated judgment rows.
#' @export
ducksemantics_record_judgments <- function(
  conn,
  judgments,
  prefix = "semantic"
) {
  ducksemantics_init(conn, prefix)
  judgments <- ducksemantics_prepare_judgments(judgments)
  if (nrow(judgments)) {
    DBI::dbAppendTable(
      conn,
      ducksemantics_tables(prefix)[["judgments"]],
      judgments
    )
  }
  invisible(judgments)
}

ducksemantics_prepare_judgments <- function(judgments) {
  judgments <- ducksemantics_require_data_frame(
    judgments,
    "judgments",
    c("judgment_id", "subject_id", "predicate", "decision")
  )
  ducksemantics_add_missing(
    judgments,
    c(
      object_id = NA_character_,
      value_json = NA_character_,
      confidence = NA_real_,
      evidence = NA_character_,
      model = NA_character_,
      recorded_at = format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"),
      attrs = NA_character_
    )
  )[, c(
    "judgment_id",
    "subject_id",
    "predicate",
    "object_id",
    "value_json",
    "decision",
    "confidence",
    "evidence",
    "model",
    "recorded_at",
    "attrs"
  )]
}
ducksemantics_empty_judgments <- function() {
  data.frame(
    judgment_id = character(),
    subject_id = character(),
    predicate = character(),
    object_id = character(),
    value_json = character(),
    decision = character(),
    confidence = numeric(),
    evidence = character(),
    model = character(),
    recorded_at = character(),
    attrs = character()
  )
}
ducksemantics_prompt_text <- function(x, arg) {
  if (
    !is.character(x) ||
      !length(x) ||
      anyNA(x) ||
      !nzchar(trimws(paste(x, collapse = "\n")))
  ) {
    stop("`", arg, "` must be non-blank character text.", call. = FALSE)
  }
  paste(x, collapse = "\n")
}
ducksemantics_response_text <- function(x) {
  if (is.character(x) && length(x) == 1L && !is.na(x)) {
    return(x)
  }
  if (is.list(x) && "text" %in% names(x)) {
    return(ducksemantics_response_text(x[["text"]]))
  }
  out <- as.character(x)
  if (!length(out) || is.na(out[[1L]])) {
    stop("Provider response could not be converted to text.", call. = FALSE)
  }
  out[[1L]]
}
ducksemantics_require_jsonlite <- function() {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop(
      "jsonlite is required for semantic judgment JSON handling.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}
ducksemantics_parse_json_response <- function(response) {
  ducksemantics_require_jsonlite()
  jsonlite::fromJSON(
    ducksemantics_extract_json(ducksemantics_require_text(
      response,
      "response"
    )),
    simplifyDataFrame = TRUE
  )
}
ducksemantics_extract_json <- function(response) {
  response <- trimws(response)
  if (startsWith(response, "[") || startsWith(response, "{")) {
    return(response)
  }
  starts <- gregexpr("[\\[{]", response, perl = TRUE)[[1L]]
  if (identical(starts, -1L)) {
    stop("Provider response did not contain JSON.", call. = FALSE)
  }
  for (start in starts) {
    candidate <- substring(response, start)
    if (isTRUE(try(jsonlite::validate(candidate), silent = TRUE))) {
      return(candidate)
    }
  }
  stop("Provider response did not contain valid JSON.", call. = FALSE)
}
ducksemantics_normalize_judgment_payload <- function(parsed) {
  wrappers <- c("array", "judgments", "results", "items", "arguments", "args")
  if (is.data.frame(parsed)) {
    return(parsed)
  }
  if (is.list(parsed) && !length(parsed)) {
    return(data.frame(mention_id = character(), decision = character()))
  }
  if (is.list(parsed)) {
    for (wrapper in wrappers) {
      if (!is.null(parsed[[wrapper]])) {
        return(ducksemantics_normalize_judgment_payload(parsed[[wrapper]]))
      }
    }
    if (all(c("mention_id", "decision") %in% names(parsed))) {
      return(ducksemantics_lists_to_data_frame(list(parsed)))
    }
  }
  stop(
    "JSON judgment payload must be an array or objects with mention_id and decision.",
    call. = FALSE
  )
}
ducksemantics_judgments_from_model <- function(
  parsed,
  mentions,
  model,
  allowed_node_id = NULL
) {
  ducksemantics_require_jsonlite()
  mentions <- ducksemantics_require_data_frame(
    mentions,
    "mentions",
    c("mention_id", "node_id")
  )
  ducksemantics_validate_required_text(
    mentions,
    c("mention_id", "node_id"),
    "mentions"
  )
  parsed <- ducksemantics_require_data_frame(
    if (is.list(parsed) && !is.data.frame(parsed)) {
      ducksemantics_lists_to_data_frame(parsed)
    } else {
      parsed
    },
    "parsed",
    c("mention_id", "decision")
  )
  if (!nrow(parsed) && !nrow(mentions)) {
    return(ducksemantics_empty_judgments())
  }
  ducksemantics_validate_required_text(
    parsed,
    c("mention_id", "decision"),
    "parsed"
  )
  if (
    anyDuplicated(parsed$mention_id) ||
      !identical(
        as.character(parsed$mention_id),
        as.character(mentions$mention_id)
      )
  ) {
    stop(
      "Model response must contain exactly one result per candidate in candidate order.",
      call. = FALSE
    )
  }
  if (!all(parsed$decision %in% c("keep", "drop", "replace", "enrich"))) {
    stop("Model returned unsupported decision value.", call. = FALSE)
  }
  confidence <- suppressWarnings(as.numeric(
    parsed$confidence %||% rep(NA_real_, nrow(parsed))
  ))
  if (
    any(
      !is.na(confidence) &
        (!is.finite(confidence) | confidence < 0 | confidence > 1)
    )
  ) {
    stop(
      "Model confidence values must be finite numbers from 0 to 1 or null.",
      call. = FALSE
    )
  }
  replacement <- if ("replacement_node_id" %in% names(parsed)) {
    as.character(parsed$replacement_node_id)
  } else {
    rep(NA_character_, nrow(parsed))
  }
  has <- !is.na(replacement) & nzchar(replacement)
  if (any(parsed$decision == "replace" & !has)) {
    stop(
      "Every replace decision must supply replacement_node_id.",
      call. = FALSE
    )
  }
  allowed <- unique(c(mentions$node_id, allowed_node_id))
  if (length(setdiff(replacement[has], allowed))) {
    stop(
      "Model returned replacement_node_id values outside supplied candidates or graph context.",
      call. = FALSE
    )
  }
  object_id <- as.character(mentions$node_id)
  object_id[has] <- replacement[has]
  object_id[parsed$decision == "drop"] <- NA_character_
  data.frame(
    judgment_id = paste0("judgment:", parsed$mention_id),
    subject_id = parsed$mention_id,
    predicate = "semantic:grounding_decision",
    object_id = object_id,
    value_json = vapply(
      seq_len(nrow(parsed)),
      function(i) {
        jsonlite::toJSON(
          as.list(parsed[i, , drop = FALSE]),
          auto_unbox = TRUE,
          null = "null"
        )
      },
      character(1)
    ),
    decision = parsed$decision,
    confidence = confidence,
    evidence = NA_character_,
    model = model,
    recorded_at = format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"),
    attrs = NA_character_,
    stringsAsFactors = FALSE
  )
}
ducksemantics_judgment_node_ids <- function(mentions, graph_context = NULL) {
  ids <- if (is.data.frame(mentions) && "node_id" %in% names(mentions)) {
    as.character(mentions$node_id)
  } else {
    character()
  }
  collect <- function(x) {
    if (is.data.frame(x)) {
      unlist(
        x[
          grep("(^|_)(node_id|from_id|to_id|object_id)$", names(x)),
          drop = FALSE
        ],
        use.names = FALSE
      )
    } else if (is.list(x)) {
      unlist(lapply(x, collect), use.names = FALSE)
    } else {
      character()
    }
  }
  unique(c(ids, collect(graph_context)))
}
