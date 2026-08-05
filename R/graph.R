DUCKSEMANTICS_SCHEMA_VERSION <- "ducksemantics.schema.v2"

#' DuckDB semantic graph schema
#'
#' @param prefix Prefix for semantic tables.
#' @return Character DDL statements.
#' @export
ducksemantics_schema_sql <- function(prefix = "semantic") {
  prefix <- ducksemantics_require_identifier(prefix, "prefix")
  tables <- ducksemantics_tables(prefix)
  ddl <- c(
    paste0(
      "CREATE TABLE IF NOT EXISTS ",
      ducksemantics_quote_ident(tables[["nodes"]]),
      " (node_id TEXT PRIMARY KEY, family TEXT NOT NULL, label TEXT, description TEXT, attrs TEXT, trust TEXT)"
    ),
    paste0(
      "CREATE TABLE IF NOT EXISTS ",
      ducksemantics_quote_ident(tables[["aliases"]]),
      " (node_id TEXT NOT NULL, alias TEXT NOT NULL, alias_kind TEXT NOT NULL, source TEXT, weight DOUBLE, attrs TEXT)"
    ),
    paste0(
      "CREATE TABLE IF NOT EXISTS ",
      ducksemantics_quote_ident(tables[["edges"]]),
      " (from_id TEXT NOT NULL, predicate TEXT NOT NULL, to_id TEXT NOT NULL, attrs TEXT, trust TEXT)"
    ),
    paste0(
      "CREATE TABLE IF NOT EXISTS ",
      ducksemantics_quote_ident(tables[["entailed_edges"]]),
      " (from_id TEXT NOT NULL, predicate TEXT NOT NULL, to_id TEXT NOT NULL)"
    ),
    paste0(
      "CREATE TABLE IF NOT EXISTS ",
      ducksemantics_quote_ident(tables[["mentions"]]),
      " (document_id TEXT, mention_id TEXT NOT NULL, node_id TEXT NOT NULL, span TEXT NOT NULL, start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL, score DOUBLE, method TEXT NOT NULL, attrs TEXT, trust TEXT)"
    ),
    paste0(
      "CREATE TABLE IF NOT EXISTS ",
      ducksemantics_quote_ident(tables[["judgments"]]),
      " (judgment_id TEXT NOT NULL, subject_id TEXT NOT NULL, predicate TEXT NOT NULL, object_id TEXT, value_json TEXT, decision TEXT NOT NULL, confidence DOUBLE, evidence TEXT, model TEXT, recorded_at TIMESTAMP, attrs TEXT)"
    ),
    paste0(
      "CREATE TABLE IF NOT EXISTS ",
      ducksemantics_quote_ident(tables[["embeddings"]]),
      " (subject_id TEXT NOT NULL, subject_kind TEXT NOT NULL, provider TEXT NOT NULL, text TEXT, dim INTEGER NOT NULL, embedding FLOAT[], attrs TEXT)"
    ),
    paste0(
      "CREATE TABLE IF NOT EXISTS ",
      ducksemantics_quote_ident(tables[["token_embeddings"]]),
      " (block_id TEXT NOT NULL, subject_id TEXT NOT NULL, subject_kind TEXT NOT NULL, provider TEXT NOT NULL, token_index INTEGER NOT NULL, token TEXT, start_offset INTEGER, end_offset INTEGER, dim INTEGER NOT NULL, embedding FLOAT[], attrs TEXT)"
    )
  )
  indexes <- c(
    ducksemantics_index_ddl(tables[["aliases"]], "alias_idx", "alias"),
    ducksemantics_index_ddl(tables[["aliases"]], "node_idx", "node_id"),
    ducksemantics_index_ddl(
      tables[["edges"]],
      "subj_idx",
      "from_id, predicate"
    ),
    ducksemantics_index_ddl(tables[["edges"]], "obj_idx", "to_id, predicate"),
    ducksemantics_index_ddl(
      tables[["entailed_edges"]],
      "subj_idx",
      "from_id, predicate"
    ),
    ducksemantics_index_ddl(
      tables[["entailed_edges"]],
      "obj_idx",
      "to_id, predicate"
    ),
    ducksemantics_index_ddl(
      tables[["embeddings"]],
      "subject_idx",
      "subject_kind, subject_id, provider"
    ),
    ducksemantics_index_ddl(tables[["embeddings"]], "dim_idx", "provider, dim"),
    ducksemantics_index_ddl(
      tables[["token_embeddings"]],
      "subject_idx",
      "subject_kind, subject_id, provider, block_id"
    ),
    ducksemantics_index_ddl(
      tables[["token_embeddings"]],
      "dim_idx",
      "provider, dim"
    )
  )
  c(ddl, indexes)
}

ducksemantics_index_ddl <- function(table, suffix, columns) {
  paste0(
    "CREATE INDEX IF NOT EXISTS ",
    ducksemantics_quote_ident(paste0(table, "_", suffix)),
    " ON ",
    ducksemantics_quote_ident(table),
    " (",
    columns,
    ")"
  )
}

#' Semantic table names
#' @param prefix Table prefix.
#' @return A named character vector.
#' @export
ducksemantics_tables <- function(prefix = "semantic") {
  prefix <- ducksemantics_require_identifier(prefix, "prefix")
  c(
    nodes = paste0(prefix, "_nodes"),
    aliases = paste0(prefix, "_aliases"),
    alias_index = paste0(prefix, "_alias_index"),
    edges = paste0(prefix, "_edges"),
    entailed_edges = paste0(prefix, "_entailed_edges"),
    mentions = paste0(prefix, "_mentions"),
    judgments = paste0(prefix, "_judgments"),
    embeddings = paste0(prefix, "_embeddings"),
    token_embeddings = paste0(prefix, "_token_embeddings")
  )
}

#' Connect to a DuckDB semantic store
#' @param dbdir DuckDB path or `":memory:"`.
#' @param read_only Open read-only.
#' @param array DuckDB array conversion mode.
#' @return A DBI connection.
#' @export
ducksemantics_connect <- function(
  dbdir = ":memory:",
  read_only = FALSE,
  array = "matrix"
) {
  if (!requireNamespace("duckdb", quietly = TRUE)) {
    stop(
      "The duckdb R package is required to create a DuckDB connection.",
      call. = FALSE
    )
  }
  DBI::dbConnect(
    duckdb::duckdb(),
    dbdir = ducksemantics_require_text(dbdir, "dbdir"),
    read_only = ducksemantics_require_flag(read_only, "read_only"),
    array = ducksemantics_require_text(array, "array")
  )
}

#' Initialize semantic graph tables
#' @param conn DBI connection.
#' @param prefix Table prefix.
#' @return Invisibly, `conn`.
#' @export
ducksemantics_init <- function(conn, prefix = "semantic") {
  ducksemantics_require_connection(conn)
  DBI::dbWithTransaction(
    conn,
    for (sql in ducksemantics_schema_sql(prefix)) {
      DBI::dbExecute(conn, sql)
    }
  )
  invisible(conn)
}

#' Write ontology graph rows
#' @param conn DBI connection.
#' @param nodes Data frame with `node_id` and `family`.
#' @param aliases Data frame with `node_id` and `alias`.
#' @param edges Data frame with `from_id`, `predicate`, and `to_id`.
#' @param prefix Table prefix.
#' @param replace Replace supplied target relations.
#' @param index Rebuild lexical alias index.
#' @return Invisibly, semantic table names.
#' @export
ducksemantics_write_graph <- function(
  conn,
  nodes = NULL,
  aliases = NULL,
  edges = NULL,
  prefix = "semantic",
  replace = FALSE,
  index = TRUE
) {
  ducksemantics_init(conn, prefix)
  replace <- ducksemantics_require_flag(replace, "replace")
  index <- ducksemantics_require_flag(index, "index")
  tables <- ducksemantics_tables(prefix)
  if (!is.null(nodes)) {
    nodes <- ducksemantics_prepare_nodes(nodes)
  }
  if (!is.null(aliases)) {
    aliases <- ducksemantics_prepare_aliases(aliases)
  }
  if (!is.null(edges)) {
    edges <- ducksemantics_prepare_edges(edges)
  }
  DBI::dbWithTransaction(conn, {
    items <- list(nodes = nodes, aliases = aliases, edges = edges)
    for (name in names(items)) {
      item <- items[[name]]
      if (is.null(item)) {
        next
      }
      table <- tables[[name]]
      if (replace) {
        DBI::dbExecute(
          conn,
          paste0("DELETE FROM ", ducksemantics_quote_ident(table))
        )
      }
      ducksemantics_append_unique_rows(conn, table, item)
    }
    if (index && !is.null(aliases)) ducksemantics_index_aliases(conn, prefix)
  })
  invisible(tables)
}

#' Read an OBO ontology into graph relations
#' @param path OBO file path.
#' @param family Ontology family.
#' @param source Alias source identity.
#' @param include_obsolete Include obsolete terms.
#' @return A list of `nodes`, `aliases`, and `edges` data frames.
#' @export
ducksemantics_read_obo <- function(
  path,
  family,
  source = basename(path),
  include_obsolete = FALSE
) {
  path <- ducksemantics_require_text(path, "path")
  family <- ducksemantics_require_text(family, "family")
  source <- ducksemantics_require_text(source, "source")
  ducksemantics_require_flag(include_obsolete, "include_obsolete")
  if (!file.exists(path)) {
    stop("OBO file does not exist: ", path, call. = FALSE)
  }
  stanzas <- ducksemantics_obo_term_stanzas(readLines(
    path,
    warn = FALSE,
    encoding = "UTF-8"
  ))
  nodes <- aliases <- edges <- list()
  for (stanza in stanzas) {
    id <- ducksemantics_obo_first(stanza, "id")
    if (
      is.na(id) ||
        !nzchar(id) ||
        (!include_obsolete &&
          identical(
            tolower(ducksemantics_obo_first(stanza, "is_obsolete")),
            "true"
          ))
    ) {
      next
    }
    label <- ducksemantics_obo_first(stanza, "name")
    nodes[[length(nodes) + 1L]] <- data.frame(
      node_id = id,
      family = family,
      label = label,
      description = ducksemantics_obo_quoted(ducksemantics_obo_first_line(
        stanza,
        "def"
      )),
      attrs = NA_character_,
      trust = NA_character_,
      stringsAsFactors = FALSE
    )
    add_alias <- function(value, kind, weight) {
      if (!is.na(value) && nzchar(value)) {
        aliases[[length(aliases) + 1L]] <<- data.frame(
          node_id = id,
          alias = value,
          alias_kind = kind,
          source = source,
          weight = weight,
          attrs = NA_character_,
          stringsAsFactors = FALSE
        )
      }
    }
    add_alias(label, "label", 1)
    for (value in ducksemantics_obo_values(stanza, "alt_id")) {
      add_alias(value, "alt_id", 1)
    }
    for (line in ducksemantics_obo_lines(stanza, "synonym")) {
      add_alias(
        ducksemantics_obo_quoted(line),
        ducksemantics_obo_synonym_kind(line),
        0.95
      )
    }
    add_edge <- function(predicate, value) {
      target <- ducksemantics_obo_object_id(value)
      if (!is.na(target) && nzchar(target)) {
        edges[[length(edges) + 1L]] <<- data.frame(
          from_id = id,
          predicate = predicate,
          to_id = target,
          attrs = NA_character_,
          trust = NA_character_,
          stringsAsFactors = FALSE
        )
      }
    }
    for (value in ducksemantics_obo_values(stanza, "is_a")) {
      add_edge("is_a", value)
    }
    for (value in ducksemantics_obo_values(stanza, "relationship")) {
      bits <- strsplit(
        sub("[[:space:]]*!.*$", "", value, perl = TRUE),
        "[[:space:]]+",
        perl = TRUE
      )[[1L]]
      if (length(bits) >= 2L) add_edge(bits[[1L]], bits[[2L]])
    }
  }
  list(
    nodes = ducksemantics_bind_or_empty(nodes, ducksemantics_empty_nodes()),
    aliases = ducksemantics_bind_or_empty(
      aliases,
      ducksemantics_empty_aliases()
    ),
    edges = ducksemantics_bind_or_empty(edges, ducksemantics_empty_edges())
  )
}

#' Write an OBO ontology into the graph
#' @inheritParams ducksemantics_read_obo
#' @param conn DBI connection.
#' @param prefix Table prefix.
#' @param replace Replace graph data.
#' @param index Rebuild alias index.
#' @return Parsed graph relations, invisibly.
#' @export
ducksemantics_write_obo <- function(
  conn,
  path,
  family,
  source = basename(path),
  prefix = "semantic",
  replace = FALSE,
  index = TRUE,
  include_obsolete = FALSE
) {
  graph <- ducksemantics_read_obo(path, family, source, include_obsolete)
  ducksemantics_write_graph(
    conn,
    graph$nodes,
    graph$aliases,
    graph$edges,
    prefix,
    replace,
    index
  )
  invisible(graph)
}

#' Normalize semantic text
#' @param text Character vector.
#' @return Normalized character vector.
#' @export
ducksemantics_normalize <- function(text) {
  if (!is.character(text)) {
    stop("`text` must be character.", call. = FALSE)
  }
  trimws(gsub(
    "[[:space:]]+",
    " ",
    gsub("[^[:alnum:]]+", " ", tolower(text), perl = TRUE),
    perl = TRUE
  ))
}

#' Tokenize source text
#' @param text Character scalar.
#' @return Token data frame with zero-based half-open character offsets.
#' @export
ducksemantics_tokens <- function(text) {
  text <- ducksemantics_require_text(text, "text")
  hits <- gregexpr("[[:alnum:]]+", text, perl = TRUE)[[1L]]
  if (identical(hits, -1L)) {
    return(data.frame(
      token_index = integer(),
      token = character(),
      normalized = character(),
      start_offset = integer(),
      end_offset = integer()
    ))
  }
  lengths <- attr(hits, "match.length")
  data.frame(
    token_index = seq_along(hits),
    token = substring(text, hits, hits + lengths - 1L),
    normalized = ducksemantics_normalize(substring(
      text,
      hits,
      hits + lengths - 1L
    )),
    start_offset = as.integer(hits - 1L),
    end_offset = as.integer(hits + lengths - 1L),
    stringsAsFactors = FALSE
  )
}

#' Build the lexical alias index
#' @param conn DBI connection.
#' @param prefix Table prefix.
#' @return Invisibly, index table name.
#' @export
ducksemantics_index_aliases <- function(conn, prefix = "semantic") {
  ducksemantics_require_connection(conn)
  tables <- ducksemantics_tables(prefix)
  aliases <- DBI::dbGetQuery(
    conn,
    paste0(
      "SELECT node_id, alias, alias_kind, source, COALESCE(weight, 1.0) AS weight, attrs FROM ",
      ducksemantics_quote_ident(tables[["aliases"]])
    )
  )
  if (nrow(aliases)) {
    aliases$normalized_alias <- ducksemantics_normalize(aliases$alias)
    aliases <- aliases[nzchar(aliases$normalized_alias), , drop = FALSE]
    aliases$token_count <- lengths(strsplit(
      aliases$normalized_alias,
      " ",
      fixed = TRUE
    ))
  } else {
    aliases <- data.frame(
      node_id = character(),
      alias = character(),
      alias_kind = character(),
      source = character(),
      weight = numeric(),
      attrs = character(),
      normalized_alias = character(),
      token_count = integer()
    )
  }
  DBI::dbWriteTable(conn, tables[["alias_index"]], aliases, overwrite = TRUE)
  DBI::dbExecute(
    conn,
    ducksemantics_index_ddl(
      tables[["alias_index"]],
      "norm_idx",
      "normalized_alias, token_count"
    )
  )
  invisible(tables[["alias_index"]])
}

#' Generate lexical HPO/ontology candidates
#'
#' These candidates are not accepted observations. Use
#' [ducksemantics_hpo_observations()] to validate a separately accepted HPO
#' observation relation against source documents.
#'
#' @param conn DBI connection.
#' @param text Source text.
#' @param document_id Optional source-document identity.
#' @param prefix Table prefix.
#' @param longest_match Drop spans contained by a longer span.
#' @param record Append candidate rows to the semantic mention table.
#' @return Candidate data frame.
#' @export
ducksemantics_annotate <- function(
  conn,
  text,
  document_id = NULL,
  prefix = "semantic",
  longest_match = TRUE,
  record = FALSE
) {
  ducksemantics_require_connection(conn)
  text <- ducksemantics_require_text(text, "text")
  if (!is.null(document_id)) {
    ducksemantics_require_text(document_id, "document_id")
  }
  longest_match <- ducksemantics_require_flag(longest_match, "longest_match")
  record <- ducksemantics_require_flag(record, "record")
  tables <- ducksemantics_tables(prefix)
  if (!DBI::dbExistsTable(conn, tables[["alias_index"]])) {
    ducksemantics_index_aliases(conn, prefix)
  }
  tokens <- ducksemantics_tokens(text)
  if (!nrow(tokens)) {
    return(ducksemantics_empty_mentions())
  }
  max_n <- as.integer(DBI::dbGetQuery(
    conn,
    paste0(
      "SELECT COALESCE(MAX(token_count), 0) AS n FROM ",
      ducksemantics_quote_ident(tables[["alias_index"]])
    )
  )$n[[1L]])
  if (is.na(max_n) || max_n < 1L) {
    return(ducksemantics_empty_mentions())
  }
  candidates <- ducksemantics_ngrams(text, tokens, max_n, document_id)
  if (!nrow(candidates)) {
    return(ducksemantics_empty_mentions())
  }
  candidate_table <- ducksemantics_temp_table_name("ducksemantics_candidates")
  DBI::dbWriteTable(
    conn,
    candidate_table,
    candidates,
    temporary = TRUE,
    overwrite = TRUE
  )
  on.exit(
    try(
      DBI::dbExecute(
        conn,
        paste0(
          "DROP TABLE IF EXISTS ",
          ducksemantics_quote_ident(candidate_table)
        )
      ),
      silent = TRUE
    ),
    add = TRUE
  )
  sql <- paste0(
    "WITH matched AS (SELECT c.document_id, c.mention_id AS span_id, a.node_id, c.span, c.start_offset, c.end_offset, a.weight AS score, 'lexical_alias' AS method, CAST(NULL AS TEXT) AS attrs, CAST(NULL AS TEXT) AS trust, a.alias, a.alias_kind, a.source, ROW_NUMBER() OVER (PARTITION BY c.mention_id, a.node_id ORDER BY a.weight DESC NULLS LAST, a.alias_kind, a.alias, a.source) AS alias_rank FROM ",
    ducksemantics_quote_ident(candidate_table),
    " c JOIN ",
    ducksemantics_quote_ident(tables[["alias_index"]]),
    " a ON c.normalized_span = a.normalized_alias AND c.token_count = a.token_count), deduplicated AS (SELECT * FROM matched WHERE alias_rank = 1), identified AS (SELECT document_id, CASE WHEN COUNT(*) OVER (PARTITION BY span_id) > 1 THEN span_id || ':' || node_id ELSE span_id END AS mention_id, node_id, span, start_offset, end_offset, score, method, attrs, trust, alias, alias_kind, source FROM deduplicated) SELECT * FROM identified ORDER BY start_offset, end_offset DESC, node_id"
  )
  out <- DBI::dbGetQuery(conn, sql)
  if (longest_match && nrow(out)) {
    out <- ducksemantics_longest_matches(out)
  }
  row.names(out) <- NULL
  if (record && nrow(out)) {
    DBI::dbAppendTable(
      conn,
      tables[["mentions"]],
      out[, c(
        "document_id",
        "mention_id",
        "node_id",
        "span",
        "start_offset",
        "end_offset",
        "score",
        "method",
        "attrs",
        "trust"
      )]
    )
  }
  out
}

#' Project edge-shaped data to graph SQL
#' @param source_table Source relation name.
#' @param from,predicate,to Source columns.
#' @param target_table Target table.
#' @param attrs,trust Optional source columns.
#' @return SQL script.
#' @export
ducksemantics_projection_sql <- function(
  source_table,
  from,
  predicate,
  to,
  target_table = "semantic_edges",
  attrs = NULL,
  trust = NULL
) {
  source_table <- ducksemantics_require_identifier(
    source_table,
    "source_table",
    TRUE
  )
  target_table <- ducksemantics_require_identifier(
    target_table,
    "target_table",
    TRUE
  )
  columns <- vapply(
    list(from = from, predicate = predicate, to = to),
    ducksemantics_require_identifier,
    character(1),
    arg = "column"
  )
  optional <- function(x) {
    if (is.null(x)) {
      "NULL"
    } else {
      ducksemantics_quote_ident(ducksemantics_require_identifier(x, "column"))
    }
  }
  paste0(
    "CREATE OR REPLACE TABLE ",
    ducksemantics_quote_ident(target_table),
    " AS SELECT ",
    ducksemantics_quote_ident(columns[["from"]]),
    " AS from_id, ",
    ducksemantics_quote_ident(columns[["predicate"]]),
    " AS predicate, ",
    ducksemantics_quote_ident(columns[["to"]]),
    " AS to_id, ",
    optional(attrs),
    " AS attrs, ",
    optional(trust),
    " AS trust FROM ",
    ducksemantics_quote_ident(source_table),
    "; ",
    ducksemantics_index_ddl(target_table, "subj_idx", "from_id, predicate"),
    "; ",
    ducksemantics_index_ddl(target_table, "obj_idx", "to_id, predicate")
  )
}

#' Materialize transitive closure SQL
#' @param transitive_predicates Non-empty predicate names, or `character()`.
#' @param source_table Source edge relation.
#' @param target_table Target closure relation.
#' @return SQL script.
#' @export
ducksemantics_closure_sql <- function(
  transitive_predicates,
  source_table = "semantic_edges",
  target_table = "semantic_entailed_edges"
) {
  source_table <- ducksemantics_require_identifier(
    source_table,
    "source_table",
    TRUE
  )
  target_table <- ducksemantics_require_identifier(
    target_table,
    "target_table",
    TRUE
  )
  if (
    !is.character(transitive_predicates) ||
      anyNA(transitive_predicates) ||
      !all(nzchar(transitive_predicates))
  ) {
    stop(
      "`transitive_predicates` must be a character vector of non-empty strings.",
      call. = FALSE
    )
  }
  create <- if (!length(transitive_predicates)) {
    paste0(
      "CREATE OR REPLACE TABLE ",
      ducksemantics_quote_ident(target_table),
      " (from_id TEXT, predicate TEXT, to_id TEXT)"
    )
  } else {
    predicates <- paste(
      vapply(transitive_predicates, ducksemantics_quote_string, character(1)),
      collapse = ", "
    )
    paste0(
      "CREATE OR REPLACE TABLE ",
      ducksemantics_quote_ident(target_table),
      " AS WITH RECURSIVE closure(from_id, predicate, to_id) AS (SELECT from_id, predicate, to_id FROM ",
      ducksemantics_quote_ident(source_table),
      " WHERE predicate IN (",
      predicates,
      ") UNION SELECT c.from_id, c.predicate, e.to_id FROM closure c JOIN ",
      ducksemantics_quote_ident(source_table),
      " e ON e.from_id = c.to_id AND e.predicate = c.predicate) SELECT DISTINCT from_id, predicate, to_id FROM closure"
    )
  }
  paste0(
    create,
    "; ",
    ducksemantics_index_ddl(target_table, "subj_idx", "from_id, predicate"),
    "; ",
    ducksemantics_index_ddl(target_table, "obj_idx", "to_id, predicate")
  )
}

ducksemantics_quote_ident <- function(x) {
  paste(
    sprintf(
      '"%s"',
      gsub('"', '""', strsplit(x, ".", fixed = TRUE)[[1L]], fixed = TRUE)
    ),
    collapse = "."
  )
}
ducksemantics_quote_string <- function(x) {
  paste0("'", gsub("'", "''", x, fixed = TRUE), "'")
}
ducksemantics_temp_table_name <- function(prefix) {
  gsub("[^A-Za-z0-9_]", "_", basename(tempfile(paste0(prefix, "_"))))
}
ducksemantics_validate_required_text <- function(x, columns, arg) {
  for (column in columns) {
    if (
      !is.character(x[[column]]) ||
        anyNA(x[[column]]) ||
        !all(nzchar(x[[column]]))
    ) {
      stop(
        "`",
        arg,
        "$",
        column,
        "` must contain non-empty strings without NA.",
        call. = FALSE
      )
    }
  }
}
ducksemantics_validate_optional_text <- function(x, columns, arg) {
  for (column in columns) {
    if (
      !is.character(x[[column]]) ||
        any(!is.na(x[[column]]) & !nzchar(x[[column]]))
    ) {
      stop(
        "`",
        arg,
        "$",
        column,
        "` must contain non-empty strings or NA.",
        call. = FALSE
      )
    }
  }
}
ducksemantics_add_missing <- function(x, defaults) {
  for (nm in names(defaults)) {
    if (!nm %in% names(x)) x[[nm]] <- rep(defaults[[nm]], nrow(x))
  }
  x
}
ducksemantics_prepare_nodes <- function(nodes) {
  nodes <- ducksemantics_require_data_frame(
    nodes,
    "nodes",
    c("node_id", "family")
  )
  nodes <- ducksemantics_add_missing(
    nodes,
    c(
      label = NA_character_,
      description = NA_character_,
      attrs = NA_character_,
      trust = NA_character_
    )
  )[, c("node_id", "family", "label", "description", "attrs", "trust")]
  ducksemantics_validate_required_text(nodes, c("node_id", "family"), "nodes")
  ducksemantics_validate_optional_text(
    nodes,
    c("label", "description", "attrs", "trust"),
    "nodes"
  )
  if (anyDuplicated(nodes$node_id)) {
    stop("`nodes$node_id` must uniquely identify one node row.", call. = FALSE)
  }
  unique(nodes)
}
ducksemantics_prepare_aliases <- function(aliases) {
  aliases <- ducksemantics_require_data_frame(
    aliases,
    "aliases",
    c("node_id", "alias")
  )
  aliases <- ducksemantics_add_missing(
    aliases,
    c(
      alias_kind = "label",
      source = NA_character_,
      weight = 1,
      attrs = NA_character_
    )
  )[, c("node_id", "alias", "alias_kind", "source", "weight", "attrs")]
  ducksemantics_validate_required_text(
    aliases,
    c("node_id", "alias", "alias_kind"),
    "aliases"
  )
  ducksemantics_validate_optional_text(aliases, c("source", "attrs"), "aliases")
  aliases$weight <- as.numeric(aliases$weight)
  if (any(!is.na(aliases$weight) & !is.finite(aliases$weight))) {
    stop("`aliases$weight` must be finite or NA.", call. = FALSE)
  }
  unique(aliases)
}
ducksemantics_prepare_edges <- function(edges) {
  edges <- ducksemantics_require_data_frame(
    edges,
    "edges",
    c("from_id", "predicate", "to_id")
  )
  edges <- ducksemantics_add_missing(
    edges,
    c(attrs = NA_character_, trust = NA_character_)
  )[, c("from_id", "predicate", "to_id", "attrs", "trust")]
  ducksemantics_validate_required_text(
    edges,
    c("from_id", "predicate", "to_id"),
    "edges"
  )
  ducksemantics_validate_optional_text(edges, c("attrs", "trust"), "edges")
  unique(edges)
}
ducksemantics_append_unique_rows <- function(conn, table, rows) {
  if (!nrow(rows)) {
    return(invisible(0L))
  }
  incoming <- ducksemantics_temp_table_name("ducksemantics_incoming")
  DBI::dbWriteTable(conn, incoming, rows, temporary = TRUE)
  on.exit(
    try(
      DBI::dbExecute(
        conn,
        paste0("DROP TABLE IF EXISTS ", ducksemantics_quote_ident(incoming))
      ),
      silent = TRUE
    ),
    add = TRUE
  )
  columns <- names(rows)
  q <- vapply(columns, ducksemantics_quote_ident, character(1))
  equal <- paste0("existing.", q, " IS NOT DISTINCT FROM incoming.", q)
  DBI::dbExecute(
    conn,
    paste0(
      "INSERT INTO ",
      ducksemantics_quote_ident(table),
      " (",
      paste(q, collapse = ", "),
      ") SELECT ",
      paste0("incoming.", q, collapse = ", "),
      " FROM ",
      ducksemantics_quote_ident(incoming),
      " incoming WHERE NOT EXISTS (SELECT 1 FROM ",
      ducksemantics_quote_ident(table),
      " existing WHERE ",
      paste(equal, collapse = " AND "),
      ")"
    )
  )
  invisible(nrow(rows))
}
ducksemantics_empty_nodes <- function() {
  data.frame(
    node_id = character(),
    family = character(),
    label = character(),
    description = character(),
    attrs = character(),
    trust = character()
  )
}
ducksemantics_empty_aliases <- function() {
  data.frame(
    node_id = character(),
    alias = character(),
    alias_kind = character(),
    source = character(),
    weight = numeric(),
    attrs = character()
  )
}
ducksemantics_empty_edges <- function() {
  data.frame(
    from_id = character(),
    predicate = character(),
    to_id = character(),
    attrs = character(),
    trust = character()
  )
}
ducksemantics_empty_mentions <- function() {
  data.frame(
    document_id = character(),
    mention_id = character(),
    node_id = character(),
    span = character(),
    start_offset = integer(),
    end_offset = integer(),
    score = numeric(),
    method = character(),
    attrs = character(),
    trust = character(),
    alias = character(),
    alias_kind = character(),
    source = character()
  )
}
ducksemantics_bind_or_empty <- function(rows, empty) {
  if (!length(rows)) {
    empty
  } else {
    out <- do.call(rbind, rows)
    row.names(out) <- NULL
    out
  }
}
ducksemantics_obo_term_stanzas <- function(lines) {
  out <- list()
  current <- character()
  active <- FALSE
  for (line in c(lines, "")) {
    line <- trimws(line)
    if (!nzchar(line)) {
      if (active && length(current)) {
        out[[length(out) + 1L]] <- current
      }
      current <- character()
      active <- FALSE
    } else if (startsWith(line, "[")) {
      if (active && length(current)) {
        out[[length(out) + 1L]] <- current
      }
      current <- character()
      active <- identical(line, "[Term]")
    } else if (active) {
      current <- c(current, line)
    }
  }
  out
}
ducksemantics_obo_lines <- function(stanza, tag) {
  stanza[startsWith(stanza, paste0(tag, ":"))]
}
ducksemantics_obo_values <- function(stanza, tag) {
  trimws(sub(
    paste0("^", tag, ":[[:space:]]*"),
    "",
    ducksemantics_obo_lines(stanza, tag),
    perl = TRUE
  ))
}
ducksemantics_obo_first_line <- function(stanza, tag) {
  x <- ducksemantics_obo_lines(stanza, tag)
  if (length(x)) x[[1L]] else NA_character_
}
ducksemantics_obo_first <- function(stanza, tag) {
  x <- ducksemantics_obo_values(stanza, tag)
  if (length(x)) x[[1L]] else NA_character_
}
ducksemantics_obo_quoted <- function(line) {
  if (is.na(line) || !grepl('"', line, fixed = TRUE)) {
    return(NA_character_)
  }
  gsub(
    paste0("\\", '"'),
    '"',
    sub('^[^"]*"(([^"\\\\]|\\\\.)*)".*$', "\\1", line, perl = TRUE),
    fixed = TRUE
  )
}
ducksemantics_obo_object_id <- function(value) {
  strsplit(
    trimws(sub("[[:space:]]*[!{].*$", "", value, perl = TRUE)),
    "[[:space:]]+",
    perl = TRUE
  )[[1L]][[1L]]
}
ducksemantics_obo_synonym_kind <- function(line) {
  rest <- sub('^[^"]*"([^"\\\\]|\\\\.)*"[[:space:]]*', "", line, perl = TRUE)
  kind <- tolower(strsplit(rest, "[[:space:]]+", perl = TRUE)[[1L]][[1L]])
  if (!nzchar(kind) || kind == "[]") "synonym" else paste0("synonym:", kind)
}
ducksemantics_ngrams <- function(text, tokens, max_n, document_id = NULL) {
  rows <- list()
  n <- nrow(tokens)
  for (i in seq_len(n)) {
    for (j in i:min(n, i + max_n - 1L)) {
      start <- tokens$start_offset[[i]]
      end <- tokens$end_offset[[j]]
      span <- substring(text, start + 1L, end)
      if (nzchar(ducksemantics_normalize(span))) {
        rows[[length(rows) + 1L]] <- data.frame(
          document_id = document_id %||% NA_character_,
          mention_id = sprintf(
            "%s%06d",
            if (is.null(document_id)) "mention:" else paste0(document_id, ":"),
            length(rows) + 1L
          ),
          span = span,
          normalized_span = ducksemantics_normalize(span),
          start_offset = start,
          end_offset = end,
          token_count = j - i + 1L
        )
      }
    }
  }
  ducksemantics_bind_or_empty(
    rows,
    data.frame(
      document_id = character(),
      mention_id = character(),
      span = character(),
      normalized_span = character(),
      start_offset = integer(),
      end_offset = integer(),
      token_count = integer()
    )
  )
}
ducksemantics_longest_matches <- function(x) {
  widths <- x$end_offset - x$start_offset
  keep <- logical(nrow(x))
  retained <- integer()
  for (i in order(-widths, x$start_offset, x$end_offset, x$node_id)) {
    contained <- any(vapply(
      retained,
      function(j) {
        (x$start_offset[[j]] <= x$start_offset[[i]]) &&
          (x$end_offset[[j]] >= x$end_offset[[i]]) &&
          (x$start_offset[[j]] < x$start_offset[[i]] ||
            x$end_offset[[j]] > x$end_offset[[i]])
      },
      logical(1)
    ))
    if (!contained) {
      keep[[i]] <- TRUE
      retained <- c(retained, i)
    }
  }
  x[keep, , drop = FALSE][
    order(x$start_offset[keep], x$end_offset[keep], x$node_id[keep]),
    ,
    drop = FALSE
  ]
}
