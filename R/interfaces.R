#' Provider protocol generics
#'
#' These S7 generics are reserved for real pluggable providers. All bulk input
#' and output remains an ordinary data frame or matrix.
#'
#' @param provider Prompt or embedding provider.
#' @param parser Judgment parser.
#' @param annotator Grounding provider.
#' @param prompt Prompt text.
#' @param response Raw response text.
#' @param text Source text.
#' @param conn DBI connection.
#' @param document_id Optional document identifier.
#' @param prefix Semantic-table prefix.
#' @param longest_match Drop nested lexical candidates.
#' @param record Persist candidates.
#' @param ... Provider-specific arguments.
#' @return Provider-specific text, matrix, or data frame.
#' @name ducksemantics_provider_generics
NULL

#' @rdname ducksemantics_provider_generics
#' @export
ducksemantics_run <- S7::new_generic("ducksemantics_run", "provider",
  function(provider, prompt, ...) S7::S7_dispatch())
#' @rdname ducksemantics_provider_generics
#' @export
ducksemantics_embed <- S7::new_generic("ducksemantics_embed", "provider",
  function(provider, text, ...) S7::S7_dispatch())
#' @rdname ducksemantics_provider_generics
#' @export
ducksemantics_token_embed <- S7::new_generic("ducksemantics_token_embed", "provider",
  function(provider, text, ...) S7::S7_dispatch())
#' @rdname ducksemantics_provider_generics
#' @export
ducksemantics_parse <- S7::new_generic("ducksemantics_parse", "parser",
  function(parser, response, ...) S7::S7_dispatch())
#' @rdname ducksemantics_provider_generics
#' @export
ducksemantics_ground <- S7::new_generic("ducksemantics_ground", "annotator",
  function(annotator, conn, text, document_id = NULL, prefix = "semantic",
           longest_match = TRUE, record = FALSE, ...) S7::S7_dispatch())

#' Prompt-runner provider protocol
#' @export
DucksemanticsPromptRunner <- s7contract::new_interface(
  "DucksemanticsPromptRunner", package = "ducksemantics",
  generics = list(run = s7contract::interface_requirement(
    ducksemantics_run, args = list(prompt = S7::class_character), returns = S7::class_character
  ))
)
#' Embedding-provider protocol
#' @export
DucksemanticsEmbeddingProvider <- s7contract::new_interface(
  "DucksemanticsEmbeddingProvider", package = "ducksemantics",
  generics = list(embed = s7contract::interface_requirement(
    ducksemantics_embed, args = list(text = S7::class_character), returns = S7::class_any
  ))
)
#' Token-embedding-provider protocol
#' @export
DucksemanticsTokenEmbeddingProvider <- s7contract::new_interface(
  "DucksemanticsTokenEmbeddingProvider", package = "ducksemantics",
  generics = list(token_embed = s7contract::interface_requirement(
    ducksemantics_token_embed, args = list(text = S7::class_character), returns = S7::class_any
  ))
)
#' Judgment-parser provider protocol
#' @export
DucksemanticsJudgmentParser <- s7contract::new_interface(
  "DucksemanticsJudgmentParser", package = "ducksemantics",
  generics = list(parse = s7contract::interface_requirement(
    ducksemantics_parse, args = list(response = S7::class_character), returns = S7::class_data.frame
  ))
)
#' Grounding-provider protocol
#' @export
DucksemanticsAnnotator <- s7contract::new_interface(
  "DucksemanticsAnnotator", package = "ducksemantics",
  generics = list(ground = s7contract::interface_requirement(
    ducksemantics_ground,
    args = list(conn = S7::class_any, text = S7::class_character,
      document_id = S7::new_union(NULL, S7::class_character),
      prefix = S7::class_character, longest_match = S7::class_logical,
      record = S7::class_logical), returns = S7::class_data.frame
  ))
)

ducksemantics_function_prompt_runner_class <- S7::new_class(
  "ducksemantics_function_prompt_runner", package = "ducksemantics",
  properties = list(fun = S7::class_function, label = S7::class_character)
)
ducksemantics_bebel_runner_class <- S7::new_class(
  "ducksemantics_bebel_runner", package = "ducksemantics",
  properties = list(agent = S7::class_any, on_event = S7::new_union(NULL, S7::class_function))
)
ducksemantics_function_embedding_provider_class <- S7::new_class(
  "ducksemantics_function_embedding_provider", package = "ducksemantics",
  properties = list(fun = S7::class_function, label = S7::class_character)
)
ducksemantics_function_token_embedding_provider_class <- S7::new_class(
  "ducksemantics_function_token_embedding_provider", package = "ducksemantics",
  properties = list(fun = S7::class_function, label = S7::class_character)
)
ducksemantics_embeddinggemma_provider_class <- S7::new_class(
  "ducksemantics_embeddinggemma_provider", package = "ducksemantics",
  properties = list(model = S7::class_any, label = S7::class_character,
    task = S7::class_character, title = S7::new_union(NULL, S7::class_character),
    dimensions = S7::class_numeric, normalize = S7::class_logical,
    truncate = S7::class_logical, check_interrupt = S7::class_logical)
)
ducksemantics_colbert_provider_class <- S7::new_class(
  "ducksemantics_colbert_provider", package = "ducksemantics",
  properties = list(model = S7::class_any, label = S7::class_character, role = S7::class_character)
)
ducksemantics_json_judgment_parser_class <- S7::new_class(
  "ducksemantics_json_judgment_parser", package = "ducksemantics"
)
ducksemantics_bebel_tool_judgment_parser_class <- S7::new_class(
  "ducksemantics_bebel_tool_judgment_parser", package = "ducksemantics",
  properties = list(tool_name = S7::new_union(NULL, S7::class_character))
)
ducksemantics_lexical_annotator_class <- S7::new_class(
  "ducksemantics_lexical_annotator", package = "ducksemantics"
)

#' Wrap a prompt function as a prompt provider
#' @param fun Function accepting a prompt and returning response text.
#' @param label Provider identity.
#' @return An object implementing [DucksemanticsPromptRunner].
#' @export
ducksemantics_prompt_runner <- function(fun, label = "function") {
  if (!is.function(fun)) stop("`fun` must be a function.", call. = FALSE)
  ducksemantics_require_text(label, "label")
  ducksemantics_function_prompt_runner_class(fun = fun, label = label)
}

#' Wrap an embedding function as an embedding provider
#' @inheritParams ducksemantics_prompt_runner
#' @return An object implementing [DucksemanticsEmbeddingProvider].
#' @export
ducksemantics_embedding_provider <- function(fun, label = "function") {
  if (!is.function(fun)) stop("`fun` must be a function.", call. = FALSE)
  ducksemantics_require_text(label, "label")
  ducksemantics_function_embedding_provider_class(fun = fun, label = label)
}

#' Wrap a token embedding function as a token provider
#' @inheritParams ducksemantics_prompt_runner
#' @return An object implementing [DucksemanticsTokenEmbeddingProvider].
#' @export
ducksemantics_token_embedding_provider <- function(fun, label = "function-token") {
  if (!is.function(fun)) stop("`fun` must be a function.", call. = FALSE)
  ducksemantics_require_text(label, "label")
  ducksemantics_function_token_embedding_provider_class(fun = fun, label = label)
}

#' Create an EmbeddingGemma provider
#' @param model An `Rbebelm` `EmbeddingGemmaModel`.
#' @param label Provider identity.
#' @param task EmbeddingGemma task.
#' @param title Optional document title.
#' @param dimensions Matryoshka dimension.
#' @param normalize L2-normalize output rows.
#' @param truncate Truncate overly long input.
#' @param check_interrupt Poll for R interrupts.
#' @return An object implementing [DucksemanticsEmbeddingProvider].
#' @export
ducksemantics_embeddinggemma_provider <- function(model, label = "Rbebelm EmbeddingGemma",
                                                  task = "semantic_similarity", title = NULL,
                                                  dimensions = 768L, normalize = TRUE,
                                                  truncate = TRUE, check_interrupt = TRUE) {
  if (!requireNamespace("Rbebelm", quietly = TRUE)) stop("Rbebelm is required for the EmbeddingGemma provider.", call. = FALSE)
  ducksemantics_require_text(label, "label")
  if (!task %in% c("retrieval_query", "retrieval_document", "question_answering",
      "fact_verification", "classification", "clustering", "semantic_similarity",
      "code_retrieval", "summarization", "raw")) stop("`task` must be an EmbeddingGemma task.", call. = FALSE)
  if (!is.null(title) && !identical(task, "retrieval_document")) stop("`title` is valid only for retrieval_document.", call. = FALSE)
  if (!dimensions %in% c(768, 512, 256, 128)) stop("`dimensions` must be 768, 512, 256, or 128.", call. = FALSE)
  ducksemantics_embeddinggemma_provider_class(model = model, label = label, task = task,
    title = title, dimensions = as.numeric(dimensions), normalize = ducksemantics_require_flag(normalize, "normalize"),
    truncate = ducksemantics_require_flag(truncate, "truncate"), check_interrupt = ducksemantics_require_flag(check_interrupt, "check_interrupt"))
}

#' Create a native ColBERT provider
#' @param model An `Rbebelm` `ColbertModel`.
#' @param role Query or document encoding role.
#' @param label Provider identity.
#' @return An object implementing [DucksemanticsTokenEmbeddingProvider].
#' @export
ducksemantics_colbert_provider <- function(model, role = c("document", "query"),
                                           label = "Rbebelm ColBERT") {
  if (!requireNamespace("Rbebelm", quietly = TRUE)) stop("Rbebelm is required for the ColBERT provider.", call. = FALSE)
  ducksemantics_colbert_provider_class(model = model, role = match.arg(role),
    label = ducksemantics_require_text(label, "label"))
}

#' Create a JSON judgment parser
#' @return An object implementing [DucksemanticsJudgmentParser].
#' @export
ducksemantics_json_judgment_parser <- function() ducksemantics_json_judgment_parser_class()

#' Create a BebeLM tool-call judgment parser
#' @param tool_name Accepted tool-call names, or `NULL`.
#' @return An object implementing [DucksemanticsJudgmentParser].
#' @export
ducksemantics_bebel_tool_judgment_parser <- function(tool_name = NULL) {
  if (!is.null(tool_name) && (!is.character(tool_name) || anyNA(tool_name) || any(!nzchar(tool_name)))) {
    stop("`tool_name` must be NULL or non-empty character names.", call. = FALSE)
  }
  ducksemantics_bebel_tool_judgment_parser_class(tool_name = tool_name)
}

#' Create the default lexical grounding provider
#' @return An object implementing [DucksemanticsAnnotator].
#' @export
ducksemantics_lexical_annotator <- function() ducksemantics_lexical_annotator_class()

S7::method(ducksemantics_run, ducksemantics_function_prompt_runner_class) <- function(provider, prompt, ...) {
  ducksemantics_response_text(provider@fun(ducksemantics_require_text(prompt, "prompt"), ...))
}
S7::method(ducksemantics_run, ducksemantics_bebel_runner_class) <- function(provider, prompt, ...) {
  if (!requireNamespace("Rbebelm", quietly = TRUE)) stop("Rbebelm is required for BebeLM judgment.", call. = FALSE)
  Rbebelm::bebel_append_user(provider@agent, ducksemantics_require_text(prompt, "prompt"))
  ducksemantics_response_text(Rbebelm::bebel_assistant_turn(provider@agent, on_event = provider@on_event))
}
S7::method(ducksemantics_embed, ducksemantics_function_embedding_provider_class) <- function(provider, text, ...) {
  if (!is.character(text) || anyNA(text)) stop("`text` must be a character vector without NA.", call. = FALSE)
  ducksemantics_require_matrix(provider@fun(text, ...), "provider result", length(text))
}
S7::method(ducksemantics_embed, ducksemantics_embeddinggemma_provider_class) <- function(provider, text, ...) {
  if (!is.character(text) || anyNA(text)) stop("`text` must be a character vector without NA.", call. = FALSE)
  ducksemantics_require_matrix(Rbebelm::embeddinggemma_embed(provider@model, text,
    task = provider@task, title = provider@title, dimensions = provider@dimensions,
    normalize = provider@normalize, truncate = provider@truncate,
    check_interrupt = provider@check_interrupt), "provider result", length(text))
}
S7::method(ducksemantics_token_embed, ducksemantics_function_token_embedding_provider_class) <- function(provider, text, ...) {
  if (!is.character(text) || anyNA(text)) stop("`text` must be a character vector without NA.", call. = FALSE)
  out <- provider@fun(text, ...)
  if (!is.list(out) || length(out) != length(text)) stop("Token embedding providers must return one object per input text.", call. = FALSE)
  out
}
S7::method(ducksemantics_token_embed, ducksemantics_colbert_provider_class) <- function(provider, text, ...) {
  if (!is.character(text) || anyNA(text) || any(!nzchar(text))) stop("`text` must be a non-empty character vector without NA.", call. = FALSE)
  lapply(text, function(one) {
    encoded <- if (identical(provider@role, "query")) Rbebelm::colbert_encode_query(provider@model, one) else Rbebelm::colbert_encode_document(provider@model, one)
    ids <- Rbebelm::colbert_embedding_ids(encoded)
    list(embeddings = Rbebelm::colbert_embedding_vectors(encoded), token_index = seq_along(ids) - 1L,
      tokens = paste0("token_id:", ids))
  })
}
S7::method(ducksemantics_parse, ducksemantics_json_judgment_parser_class) <- function(parser, response, ...) {
  parsed <- ducksemantics_normalize_judgment_payload(ducksemantics_parse_json_response(response))
  ducksemantics_require_data_frame(parsed, "parsed")
}
S7::method(ducksemantics_parse, ducksemantics_bebel_tool_judgment_parser_class) <- function(parser, response, ...) {
  if (!requireNamespace("Rbebelm", quietly = TRUE)) stop("Rbebelm is required to parse BebeLM tool calls.", call. = FALSE)
  blocks <- ducksemantics_bebel_tool_blocks(response)
  calls <- tryCatch(unlist(lapply(blocks, Rbebelm::bebel_parse_tool_calls), recursive = FALSE, use.names = FALSE), error = function(e) list())
  if (!is.null(parser@tool_name) && length(calls)) calls <- calls[vapply(calls, function(x) x$name %in% parser@tool_name, logical(1))]
  if (length(calls)) return(ducksemantics_lists_to_data_frame(lapply(calls, `[[`, "arguments")))
  ducksemantics_parse_json_candidates(c(blocks, response))
}
S7::method(ducksemantics_ground, ducksemantics_lexical_annotator_class) <- function(annotator, conn, text,
                                                                                    document_id = NULL, prefix = "semantic",
                                                                                    longest_match = TRUE, record = FALSE, ...) {
  ducksemantics_annotate(conn, text, document_id = document_id, prefix = prefix,
    longest_match = longest_match, record = record)
}

ducksemantics_bebel_tool_blocks <- function(response) {
  response <- ducksemantics_require_text(response, "response")
  pieces <- strsplit(response, "<\\|tool_call_start\\|>", perl = TRUE)[[1L]][-1L]
  pieces <- sub("<\\|tool_call_end\\|>.*$", "", pieces, perl = TRUE)
  pieces <- trimws(pieces)
  if (!length(pieces) || !any(nzchar(pieces))) trimws(response) else pieces[nzchar(pieces)]
}
ducksemantics_parse_json_candidates <- function(candidates) {
  for (candidate in candidates[nzchar(trimws(candidates))]) {
    parsed <- try(ducksemantics_normalize_judgment_payload(ducksemantics_parse_json_response(candidate)), silent = TRUE)
    if (!inherits(parsed, "try-error")) return(ducksemantics_require_data_frame(parsed, "parsed"))
  }
  stop("BebeLM response did not contain judgment tool calls or JSON.", call. = FALSE)
}
ducksemantics_lists_to_data_frame <- function(rows) {
  columns <- unique(unlist(lapply(rows, names), use.names = FALSE))
  if (!length(columns)) return(data.frame())
  out <- lapply(columns, function(column) vapply(rows, function(row) {
    value <- row[[column]]
    if (is.null(value) || !length(value) || is.na(value[[1L]])) NA_character_ else as.character(value[[1L]])
  }, character(1)))
  names(out) <- columns
  data.frame(out, stringsAsFactors = FALSE, check.names = FALSE)
}
