#' Store embedding rows in DuckDB
#' @param conn DBI connection.
#' @param embeddings Data frame from [ducksemantics_embedding_batch()].
#' @param prefix Table prefix.
#' @param replace Replace existing rows with the same subject and provider.
#' @return Invisibly, written rows.
#' @export
ducksemantics_write_embeddings <- function(
  conn,
  embeddings,
  prefix = "semantic",
  replace = FALSE
) {
  ducksemantics_init(conn, prefix)
  rows <- ducksemantics_prepare_embedding_rows(embeddings)
  replace <- ducksemantics_require_flag(replace, "replace")
  table <- ducksemantics_tables(prefix)[["embeddings"]]
  DBI::dbWithTransaction(conn, {
    if (replace && nrow(rows)) {
      ducksemantics_delete_embedding_subjects(conn, table, rows)
    }
    if (nrow(rows)) DBI::dbAppendTable(conn, table, rows)
  })
  invisible(rows)
}

#' Store token embedding rows in DuckDB
#' @param conn DBI connection.
#' @param embeddings Data frame from [ducksemantics_token_embedding_batch()].
#' @param prefix Table prefix.
#' @param replace Replace existing rows with the same subject and provider.
#' @return Invisibly, written rows.
#' @export
ducksemantics_write_token_embeddings <- function(
  conn,
  embeddings,
  prefix = "semantic",
  replace = FALSE
) {
  ducksemantics_init(conn, prefix)
  rows <- ducksemantics_prepare_token_embedding_rows(embeddings)
  replace <- ducksemantics_require_flag(replace, "replace")
  table <- ducksemantics_tables(prefix)[["token_embeddings"]]
  DBI::dbWithTransaction(conn, {
    if (replace && nrow(rows)) {
      ducksemantics_delete_embedding_subjects(conn, table, rows)
    }
    if (nrow(rows)) DBI::dbAppendTable(conn, table, rows)
  })
  invisible(rows)
}

#' Search dense embeddings
#' @param conn DBI connection.
#' @param embedding Numeric query vector.
#' @param provider Optional provider identity.
#' @param subject_kind Optional subject-kind filter.
#' @param top_k Number of results.
#' @param metric `"cosine"`, `"cosine_distance"`, `"l2"`, or
#'   `"inner_product"`.
#' @param table Caller-selected embedding table.
#' @return Ranked embedding rows.
#' @export
ducksemantics_embedding_search <- function(
  conn,
  embedding,
  provider = NULL,
  subject_kind = NULL,
  top_k = 10L,
  metric = c("cosine", "cosine_distance", "l2", "inner_product"),
  table = NULL
) {
  ducksemantics_require_connection(conn)
  embedding <- as.numeric(embedding)
  if (!length(embedding) || anyNA(embedding) || !all(is.finite(embedding))) {
    stop("`embedding` must be a finite numeric vector.", call. = FALSE)
  }
  metric <- match.arg(metric)
  top_k <- ducksemantics_require_positive_integer(top_k, "top_k")
  if (
    metric %in%
      c("cosine", "cosine_distance") &&
      sum(embedding * embedding) == 0
  ) {
    stop("`embedding` must be non-zero for a cosine metric.", call. = FALSE)
  }
  if (!is.null(provider)) {
    ducksemantics_require_text(provider, "provider")
  }
  if (!is.null(subject_kind)) {
    ducksemantics_require_text(subject_kind, "subject_kind")
  }
  table <- ducksemantics_require_identifier(
    table %||% ducksemantics_tables(prefix = "semantic")[["embeddings"]],
    "table",
    TRUE
  )
  dim <- length(embedding)
  vector <- ducksemantics_float_array_literal(embedding)
  stored <- paste0("embedding::FLOAT[", dim, "]")
  score <- switch(
    metric,
    cosine = paste0("array_cosine_similarity(", stored, ", ", vector, ")"),
    cosine_distance = paste0(
      "array_cosine_distance(",
      stored,
      ", ",
      vector,
      ")"
    ),
    l2 = paste0("array_distance(", stored, ", ", vector, ")"),
    inner_product = paste0("array_inner_product(", stored, ", ", vector, ")")
  )
  filters <- c(paste0("dim = ", dim), "embedding IS NOT NULL")
  if (!is.null(provider)) {
    filters <- c(
      filters,
      paste0("provider = ", ducksemantics_quote_string(provider))
    )
  }
  if (!is.null(subject_kind)) {
    filters <- c(
      filters,
      paste0("subject_kind = ", ducksemantics_quote_string(subject_kind))
    )
  }
  DBI::dbGetQuery(
    conn,
    paste0(
      "SELECT subject_id, subject_kind, provider, text, dim, ",
      score,
      " AS score FROM ",
      ducksemantics_quote_ident(table),
      " WHERE ",
      paste(filters, collapse = " AND "),
      " ORDER BY score ",
      if (metric %in% c("cosine", "inner_product")) "DESC" else "ASC",
      " LIMIT ",
      top_k
    )
  )
}

#' Exact token-level late-interaction search
#' @param conn DBI connection.
#' @param embeddings Numeric query-token matrix.
#' @param provider Optional provider identity.
#' @param subject_kind Optional subject-kind filter.
#' @param top_k Number of blocks to return.
#' @param table Caller-selected token embedding table.
#' @param candidate_subject_id Optional candidate restriction.
#' @return Ranked exact MaxSim rows.
#' @export
ducksemantics_late_interaction_search <- function(
  conn,
  embeddings,
  provider = NULL,
  subject_kind = NULL,
  top_k = 10L,
  table = NULL,
  candidate_subject_id = NULL
) {
  ducksemantics_require_connection(conn)
  query <- ducksemantics_require_matrix(embeddings, "embeddings")
  query <- ducksemantics_l2_normalize_rows(query)
  top_k <- ducksemantics_require_positive_integer(top_k, "top_k")
  if (!is.null(provider)) {
    ducksemantics_require_text(provider, "provider")
  }
  if (!is.null(subject_kind)) {
    ducksemantics_require_text(subject_kind, "subject_kind")
  }
  table <- ducksemantics_require_identifier(
    table %||% ducksemantics_tables()[["token_embeddings"]],
    "table",
    TRUE
  )
  filters <- c(paste0("dim = ", ncol(query)), "embedding IS NOT NULL")
  if (!is.null(provider)) {
    filters <- c(
      filters,
      paste0("provider = ", ducksemantics_quote_string(provider))
    )
  }
  if (!is.null(subject_kind)) {
    filters <- c(
      filters,
      paste0("subject_kind = ", ducksemantics_quote_string(subject_kind))
    )
  }
  if (!is.null(candidate_subject_id)) {
    if (
      !is.character(candidate_subject_id) ||
        anyNA(candidate_subject_id) ||
        !all(nzchar(candidate_subject_id))
    ) {
      stop("`candidate_subject_id` must be non-empty text.", call. = FALSE)
    }
    filters <- c(
      filters,
      paste0(
        "subject_id IN (",
        paste(
          vapply(
            unique(candidate_subject_id),
            ducksemantics_quote_string,
            character(1)
          ),
          collapse = ", "
        ),
        ")"
      )
    )
  }
  rows <- DBI::dbGetQuery(
    conn,
    paste0(
      "SELECT block_id, subject_id, subject_kind, provider, token_index, token, dim, embedding FROM ",
      ducksemantics_quote_ident(table),
      " WHERE ",
      paste(filters, collapse = " AND "),
      " ORDER BY provider, subject_kind, subject_id, block_id, token_index"
    )
  )
  empty <- data.frame(
    block_id = character(),
    subject_id = character(),
    subject_kind = character(),
    provider = character(),
    dim = integer(),
    score = numeric(),
    score_mean = numeric(),
    score_sum = numeric(),
    query_token_count = integer(),
    candidate_token_count = integer(),
    best_token_index = character(),
    best_token = character()
  )
  if (!nrow(rows)) {
    return(empty)
  }
  stored <- ducksemantics_l2_normalize_rows(ducksemantics_embedding_column_matrix(
    rows$embedding,
    ncol(query)
  ))
  starts <- c(
    TRUE,
    rows$provider[-1L] != rows$provider[-nrow(rows)] |
      rows$subject_kind[-1L] != rows$subject_kind[-nrow(rows)] |
      rows$subject_id[-1L] != rows$subject_id[-nrow(rows)] |
      rows$block_id[-1L] != rows$block_id[-nrow(rows)]
  )
  groups <- split(seq_len(nrow(rows)), cumsum(starts))
  out <- do.call(
    rbind,
    lapply(groups, function(i) {
      sim <- query %*% t(stored[i, , drop = FALSE])
      best <- max.col(sim, ties.method = "first")
      scores <- sim[cbind(seq_len(nrow(sim)), best)]
      picks <- i[best]
      data.frame(
        block_id = rows$block_id[[i[[1L]]]],
        subject_id = rows$subject_id[[i[[1L]]]],
        subject_kind = rows$subject_kind[[i[[1L]]]],
        provider = rows$provider[[i[[1L]]]],
        dim = ncol(query),
        score = sum(scores),
        score_mean = mean(scores),
        score_sum = sum(scores),
        query_token_count = nrow(query),
        candidate_token_count = length(i),
        best_token_index = paste(rows$token_index[picks], collapse = ","),
        best_token = paste(rows$token[picks], collapse = ""),
        stringsAsFactors = FALSE
      )
    })
  )
  out <- out[order(out$score, decreasing = TRUE), , drop = FALSE]
  row.names(out) <- NULL
  utils::head(out, top_k)
}

#' Materialize a fixed-dimension embedding table
#' @param conn DBI connection.
#' @param dimensions Embedding width.
#' @param provider Optional provider filter.
#' @param subject_kind Optional subject kind filter.
#' @param table Target table name.
#' @param source_table Source embedding table.
#' @param hnsw Create an optional DuckDB HNSW index.
#' @param metric HNSW metric.
#' @return Target table name.
#' @export
ducksemantics_materialize_embedding_index <- function(
  conn,
  dimensions,
  provider = NULL,
  subject_kind = NULL,
  table = NULL,
  source_table = "semantic_embeddings",
  hnsw = FALSE,
  metric = "cosine"
) {
  ducksemantics_require_connection(conn)
  dimensions <- ducksemantics_require_positive_integer(dimensions, "dimensions")
  table <- ducksemantics_require_identifier(
    table %||% paste0("semantic_embedding_index_", dimensions),
    "table",
    TRUE
  )
  source_table <- ducksemantics_require_identifier(
    source_table,
    "source_table",
    TRUE
  )
  ducksemantics_require_flag(hnsw, "hnsw")
  filters <- paste0("dim = ", dimensions)
  if (!is.null(provider)) {
    filters <- c(
      filters,
      paste0(
        "provider = ",
        ducksemantics_quote_string(ducksemantics_require_text(
          provider,
          "provider"
        ))
      )
    )
  }
  if (!is.null(subject_kind)) {
    filters <- c(
      filters,
      paste0(
        "subject_kind = ",
        ducksemantics_quote_string(ducksemantics_require_text(
          subject_kind,
          "subject_kind"
        ))
      )
    )
  }
  DBI::dbExecute(
    conn,
    paste0(
      "CREATE OR REPLACE TABLE ",
      ducksemantics_quote_ident(table),
      " AS SELECT subject_id, subject_kind, provider, text, dim, embedding::FLOAT[",
      dimensions,
      "] AS embedding, attrs FROM ",
      ducksemantics_quote_ident(source_table),
      " WHERE ",
      paste(filters, collapse = " AND ")
    )
  )
  if (hnsw) {
    DBI::dbExecute(conn, "LOAD vss")
    DBI::dbExecute(
      conn,
      paste0(
        "CREATE INDEX ",
        ducksemantics_quote_ident(paste0(
          gsub("[^A-Za-z0-9_]", "_", table),
          "_hnsw_idx"
        )),
        " ON ",
        ducksemantics_quote_ident(table),
        " USING HNSW (embedding) WITH (metric = ",
        ducksemantics_quote_string(metric),
        ")"
      )
    )
  }
  table
}

ducksemantics_prepare_embedding_rows <- function(rows) {
  rows <- ducksemantics_require_data_frame(
    rows,
    "embeddings",
    c("subject_id", "subject_kind", "provider", "dim", "embedding")
  )
  rows <- ducksemantics_add_missing(
    rows,
    c(text = NA_character_, attrs = NA_character_)
  )[, c(
    "subject_id",
    "subject_kind",
    "provider",
    "text",
    "dim",
    "embedding",
    "attrs"
  )]
  ducksemantics_validate_required_text(
    rows,
    c("subject_id", "subject_kind", "provider"),
    "embeddings"
  )
  if (
    !is.numeric(rows$dim) ||
      anyNA(rows$dim) ||
      any(rows$dim < 1 | rows$dim != floor(rows$dim))
  ) {
    stop("`embeddings$dim` must contain positive integers.", call. = FALSE)
  }
  ducksemantics_embedding_column_matrix(rows$embedding, unique(rows$dim))
  rows$dim <- as.integer(rows$dim)
  rows
}
ducksemantics_prepare_token_embedding_rows <- function(rows) {
  rows <- ducksemantics_require_data_frame(
    rows,
    "embeddings",
    c(
      "block_id",
      "subject_id",
      "subject_kind",
      "provider",
      "token_index",
      "dim",
      "embedding"
    )
  )
  rows <- ducksemantics_add_missing(
    rows,
    c(
      token = NA_character_,
      start_offset = NA_integer_,
      end_offset = NA_integer_,
      attrs = NA_character_
    )
  )[, c(
    "block_id",
    "subject_id",
    "subject_kind",
    "provider",
    "token_index",
    "token",
    "start_offset",
    "end_offset",
    "dim",
    "embedding",
    "attrs"
  )]
  ducksemantics_validate_required_text(
    rows,
    c("block_id", "subject_id", "subject_kind", "provider"),
    "embeddings"
  )
  ducksemantics_validate_half_open_offsets(
    rows$start_offset[!is.na(rows$start_offset)],
    rows$end_offset[!is.na(rows$end_offset)],
    "embeddings"
  )
  ducksemantics_embedding_column_matrix(rows$embedding, unique(rows$dim))
  rows$token_index <- as.integer(rows$token_index)
  rows$dim <- as.integer(rows$dim)
  rows
}
ducksemantics_embedding_column_matrix <- function(embedding, dimensions) {
  if (length(dimensions) != 1L) {
    stop("Embedding rows must have one shared dimension.", call. = FALSE)
  }
  rows <- if (is.matrix(embedding)) {
    split(embedding, row(embedding))
  } else {
    lapply(embedding, function(x) as.numeric(unlist(x, use.names = FALSE)))
  }
  if (any(lengths(rows) != dimensions)) {
    stop("Embedding row does not match declared dimension.", call. = FALSE)
  }
  out <- do.call(rbind, rows)
  storage.mode(out) <- "double"
  if (anyNA(out) || !all(is.finite(out))) {
    stop("Embedding rows must be finite.", call. = FALSE)
  }
  out
}
ducksemantics_l2_normalize_rows <- function(x) {
  norms <- sqrt(rowSums(x * x))
  if (any(!is.finite(norms) | norms <= 0)) {
    stop(
      "Late-interaction embedding rows must have a finite non-zero norm.",
      call. = FALSE
    )
  }
  x / norms
}
ducksemantics_float_array_literal <- function(x) {
  paste0(
    "[",
    paste(sprintf("%.9g::FLOAT", as.numeric(x)), collapse = ","),
    "]::FLOAT[",
    length(x),
    "]"
  )
}

ducksemantics_delete_embedding_subjects <- function(conn, table, rows) {
  groups <- interaction(
    rows$subject_kind,
    rows$provider,
    drop = TRUE,
    lex.order = TRUE
  )
  for (group in split(seq_len(nrow(rows)), groups)) {
    ids <- paste(
      vapply(
        unique(rows$subject_id[group]),
        ducksemantics_quote_string,
        character(1)
      ),
      collapse = ", "
    )
    DBI::dbExecute(
      conn,
      paste0(
        "DELETE FROM ",
        ducksemantics_quote_ident(table),
        " WHERE subject_kind = ",
        ducksemantics_quote_string(rows$subject_kind[[group[[1L]]]]),
        " AND provider = ",
        ducksemantics_quote_string(rows$provider[[group[[1L]]]]),
        " AND subject_id IN (",
        ids,
        ")"
      )
    )
  }
  invisible(NULL)
}
