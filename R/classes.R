ducksemantics_require_text <- function(x, arg, allow_null = FALSE) {
  if (is.null(x) && isTRUE(allow_null)) return(x)
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    stop("`", arg, "` must be a non-empty character scalar.", call. = FALSE)
  }
  x
}

ducksemantics_require_flag <- function(x, arg) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop("`", arg, "` must be TRUE or FALSE.", call. = FALSE)
  }
  x
}

ducksemantics_require_positive_integer <- function(x, arg) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || !is.finite(x) ||
      x < 1 || x != floor(x)) {
    stop("`", arg, "` must be a positive integer scalar.", call. = FALSE)
  }
  as.integer(x)
}

ducksemantics_require_connection <- function(conn, arg = "conn") {
  if (!inherits(conn, "DBIConnection") || !DBI::dbIsValid(conn)) {
    stop("`", arg, "` must be a valid DBI connection.", call. = FALSE)
  }
  conn
}

ducksemantics_require_data_frame <- function(x, arg, required = character(),
                                             allow_empty = TRUE) {
  if (!is.data.frame(x)) stop("`", arg, "` must be a data frame.", call. = FALSE)
  missing <- setdiff(required, names(x))
  if (length(missing)) {
    stop("`", arg, "` is missing required column(s): ",
      paste(missing, collapse = ", "), ".", call. = FALSE)
  }
  if (!isTRUE(allow_empty) && !nrow(x)) {
    stop("`", arg, "` must contain at least one row.", call. = FALSE)
  }
  x
}

ducksemantics_require_identifier <- function(x, arg, qualified = FALSE) {
  x <- ducksemantics_require_text(x, arg)
  pattern <- if (isTRUE(qualified)) {
    "^[A-Za-z_][A-Za-z0-9_]*(\\.[A-Za-z_][A-Za-z0-9_]*){0,2}$"
  } else "^[A-Za-z_][A-Za-z0-9_]*$"
  if (!grepl(pattern, x, perl = TRUE)) {
    stop("`", arg, "` must be a valid SQL identifier.", call. = FALSE)
  }
  x
}

ducksemantics_require_matrix <- function(x, arg, rows = NULL) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (!nrow(x) || !ncol(x) || anyNA(x) || any(!is.finite(x))) {
    stop("`", arg, "` must be a non-empty finite numeric matrix.", call. = FALSE)
  }
  if (!is.null(rows) && nrow(x) != rows) {
    stop("`", arg, "` must have one row per input text.", call. = FALSE)
  }
  x
}

#' Construct embedding rows
#'
#' Bulk embedding values are an ordinary data frame with one `FLOAT[]`-ready
#' numeric vector per row; they are not an S7 value object.
#'
#' @param embeddings Finite numeric matrix with one row per subject.
#' @param subject_id Non-empty subject identifiers.
#' @param subject_kind Subject type.
#' @param provider Provider identity.
#' @param text Optional source text per row.
#' @param attrs Optional metadata text per row.
#' @return A data frame suitable for [ducksemantics_write_embeddings()].
#' @export
ducksemantics_embedding_batch <- function(embeddings, subject_id,
                                          subject_kind = "node",
                                          provider = "embedding",
                                          text = NULL, attrs = NULL) {
  subject_id <- as.character(subject_id)
  embeddings <- ducksemantics_require_matrix(embeddings, "embeddings", length(subject_id))
  ducksemantics_require_text(subject_kind, "subject_kind")
  ducksemantics_require_text(provider, "provider")
  if (anyNA(subject_id) || any(!nzchar(subject_id))) {
    stop("`subject_id` must contain one non-empty value per embedding row.", call. = FALSE)
  }
  n <- nrow(embeddings)
  if (!is.null(text) && !(length(text) %in% c(1L, n))) {
    stop("`text` must be NULL, length 1, or one value per embedding row.", call. = FALSE)
  }
  if (!is.null(attrs) && !(length(attrs) %in% c(1L, n))) {
    stop("`attrs` must be NULL, length 1, or one value per embedding row.", call. = FALSE)
  }
  out <- data.frame(
    subject_id = subject_id, subject_kind = rep(subject_kind, n),
    provider = rep(provider, n),
    text = if (is.null(text)) NA_character_ else rep(as.character(text), length.out = n),
    dim = rep(as.integer(ncol(embeddings)), n),
    attrs = if (is.null(attrs)) NA_character_ else rep(as.character(attrs), length.out = n),
    stringsAsFactors = FALSE
  )
  out$embedding <- I(lapply(seq_len(n), function(i) as.single(embeddings[i, ])))
  out[, c("subject_id", "subject_kind", "provider", "text", "dim", "embedding", "attrs")]
}

ducksemantics_default_block_id <- function(provider, subject_kind, subject_id) {
  paste0(provider, "|", subject_kind, "|", subject_id)
}

#' Construct token embedding rows
#'
#' @inheritParams ducksemantics_embedding_batch
#' @param token_index Zero-based token index within a document block.
#' @param block_id Optional block identifier.
#' @param token Optional token text.
#' @param start_offset,end_offset Optional zero-based half-open source offsets.
#' @return A data frame suitable for [ducksemantics_write_token_embeddings()].
#' @export
ducksemantics_token_embedding_batch <- function(embeddings, subject_id,
                                                subject_kind = "node",
                                                provider = "embedding",
                                                token_index = NULL,
                                                block_id = NULL,
                                                token = NULL,
                                                start_offset = NULL,
                                                end_offset = NULL,
                                                attrs = NULL) {
  subject_id <- as.character(subject_id)
  embeddings <- ducksemantics_require_matrix(embeddings, "embeddings", length(subject_id))
  ducksemantics_require_text(subject_kind, "subject_kind")
  ducksemantics_require_text(provider, "provider")
  n <- nrow(embeddings)
  if (anyNA(subject_id) || any(!nzchar(subject_id))) {
    stop("`subject_id` must contain one non-empty value per token row.", call. = FALSE)
  }
  if (is.null(block_id)) block_id <- ducksemantics_default_block_id(provider, subject_kind, subject_id)
  block_id <- as.character(block_id)
  if (length(block_id) != n || anyNA(block_id) || any(!nzchar(block_id))) {
    stop("`block_id` must contain one non-empty value per token row.", call. = FALSE)
  }
  if (is.null(token_index)) {
    groups <- interaction(subject_id, block_id, drop = TRUE, lex.order = TRUE)
    token_index <- stats::ave(seq_len(n), groups, FUN = seq_along) - 1L
  }
  if (!is.numeric(token_index) || length(token_index) != n || anyNA(token_index) ||
      any(!is.finite(token_index)) || any(token_index < 0) || any(token_index != floor(token_index))) {
    stop("`token_index` must contain one non-negative integer per token row.", call. = FALSE)
  }
  if (anyDuplicated(data.frame(block_id, subject_id, token_index))) {
    stop("`token_index` must be unique within each block and subject.", call. = FALSE)
  }
  recycle_optional <- function(x, arg) {
    if (is.null(x)) return(rep(NA, n))
    if (!(length(x) %in% c(1L, n))) stop("`", arg, "` must be NULL, length 1, or one value per token row.", call. = FALSE)
    rep(x, length.out = n)
  }
  token <- as.character(recycle_optional(token, "token"))
  attrs <- as.character(recycle_optional(attrs, "attrs"))
  start_offset <- recycle_optional(start_offset, "start_offset")
  end_offset <- recycle_optional(end_offset, "end_offset")
  if (any(!is.na(start_offset) & (!is.finite(start_offset) | start_offset < 0 | start_offset != floor(start_offset))) ||
      any(!is.na(end_offset) & (!is.finite(end_offset) | end_offset < 0 | end_offset != floor(end_offset))) ||
      any(xor(is.na(start_offset), is.na(end_offset))) ||
      any(!is.na(start_offset) & start_offset >= end_offset)) {
    stop("Token source offsets must be paired, non-negative, and non-empty half-open spans.", call. = FALSE)
  }
  out <- data.frame(
    block_id = block_id, subject_id = subject_id,
    subject_kind = rep(subject_kind, n), provider = rep(provider, n),
    token_index = as.integer(token_index), token = token,
    start_offset = as.integer(start_offset), end_offset = as.integer(end_offset),
    dim = rep(as.integer(ncol(embeddings)), n), attrs = attrs,
    stringsAsFactors = FALSE
  )
  out$embedding <- I(lapply(seq_len(n), function(i) as.single(embeddings[i, ])))
  out[, c("block_id", "subject_id", "subject_kind", "provider", "token_index",
    "token", "start_offset", "end_offset", "dim", "embedding", "attrs")]
}

#' Construct token embedding rows from a provider
#'
#' @param text Character vector to embed.
#' @param provider Object implementing [DucksemanticsTokenEmbeddingProvider].
#' @param subject_id Subject identifiers for input texts.
#' @param subject_kind Subject type for stored rows.
#' @param provider_label Stored provider identity.
#' @param block_id Optional block identifier per input text.
#' @param attrs Optional metadata per input text.
#' @param ... Arguments forwarded to [ducksemantics_token_embed()].
#' @return A token-embedding data frame.
#' @export
ducksemantics_token_embedding_batch_from_provider <- function(text, provider,
                                                              subject_id = text,
                                                              subject_kind = "node",
                                                              provider_label = NULL,
                                                              block_id = NULL,
                                                              attrs = NULL, ...) {
  if (!is.character(text) || !length(text) || anyNA(text)) {
    stop("`text` must be a non-empty character vector without NA.", call. = FALSE)
  }
  subject_id <- as.character(subject_id)
  if (length(subject_id) != length(text) || anyNA(subject_id) || any(!nzchar(subject_id))) {
    stop("`subject_id` must contain one non-empty value per input text.", call. = FALSE)
  }
  ducksemantics_require_text(subject_kind, "subject_kind")
  if (is.null(provider_label)) provider_label <- provider@label
  ducksemantics_require_text(provider_label, "provider_label")
  tokens <- ducksemantics_token_embed(provider, text, ...)
  if (!is.list(tokens) || length(tokens) != length(text)) {
    stop("Token embedding providers must return one object per input text.", call. = FALSE)
  }
  rows <- lapply(seq_along(tokens), function(i) {
    one <- tokens[[i]]
    if (is.null(one$embeddings)) stop("Token embedding object ", i, " must contain `embeddings`.", call. = FALSE)
    n <- nrow(as.matrix(one$embeddings))
    ducksemantics_token_embedding_batch(
      embeddings = one$embeddings, subject_id = rep(subject_id[[i]], n),
      subject_kind = subject_kind, provider = provider_label,
      token_index = one$token_index %||% (seq_len(n) - 1L),
      block_id = if (is.null(block_id)) rep(ducksemantics_default_block_id(provider_label, subject_kind, subject_id[[i]]), n) else rep(block_id[[i]], n),
      token = one$tokens %||% rep(NA_character_, n),
      start_offset = one$start_offset %||% rep(NA_integer_, n),
      end_offset = one$end_offset %||% rep(NA_integer_, n),
      attrs = if (is.null(attrs)) NA_character_ else attrs[[i]]
    )
  })
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out
}

`%||%` <- function(x, y) {
  if (is.null(x) || !length(x) || is.na(x[[1L]])) y else x
}
