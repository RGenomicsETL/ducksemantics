#' Validate a typed Monarch release catalog
#'
#' A catalog row identifies one provider release. It must carry either a typed
#' `effective_date` (`Date` or `POSIXct`) or a finite numeric
#' `source_ordinal`; both may be supplied when the provider declares both.
#'
#' @param releases Data frame or caller-owned DuckDB table name containing
#'   `release_id`, `provider_id`, and one or both ordering columns.
#' @param conn Optional DBI connection for a table relation.
#' @return Validated release catalog.
#' @export
ducksemantics_monarch_release_catalog <- function(releases, conn = NULL) {
  releases <- ducksemantics_collect_relation(releases, conn, "releases")
  releases <- ducksemantics_require_data_frame(releases, "releases", c("release_id", "provider_id"))
  has_date <- "effective_date" %in% names(releases)
  has_ordinal <- "source_ordinal" %in% names(releases)
  if (!has_date && !has_ordinal) {
    stop("`releases` must contain typed `effective_date` or numeric `source_ordinal`.", call. = FALSE)
  }
  if (!has_date) releases$effective_date <- as.Date(rep(NA_real_, nrow(releases)), origin = "1970-01-01")
  if (!has_ordinal) releases$source_ordinal <- rep(NA_real_, nrow(releases))
  releases <- releases[, c("release_id", "provider_id", "effective_date", "source_ordinal"), drop = FALSE]
  ducksemantics_validate_required_text(releases, c("release_id", "provider_id"), "releases")
  effective_date <- ducksemantics_typed_time(releases$effective_date, "releases$effective_date", allow_na = TRUE)
  source_ordinal <- suppressWarnings(as.numeric(releases$source_ordinal))
  if (any(!is.na(releases$source_ordinal) & (is.na(source_ordinal) | !is.finite(source_ordinal)))) {
    stop("`releases$source_ordinal` must contain finite numbers or NA.", call. = FALSE)
  }
  if (any(is.na(effective_date) & is.na(source_ordinal))) {
    stop("Every release must have `effective_date` or `source_ordinal`.", call. = FALSE)
  }
  if (anyDuplicated(releases[c("provider_id", "release_id")])) {
    stop("`provider_id` and `release_id` must uniquely identify catalog rows.", call. = FALSE)
  }
  releases$source_ordinal <- source_ordinal
  releases
}

#' Import caller-supplied Monarch relations
#'
#' This function never downloads Monarch data. A typed release catalog is
#' required to prove that every fact has a known provider/release identity.
#'
#' @param facts Gene-phenotype or gene-disease data frame, or a caller-owned
#'   DuckDB table name with `conn`.
#' @param releases Typed catalog from [ducksemantics_monarch_release_catalog()].
#' @param relation Either `"gene_phenotype"` or `"gene_disease"`.
#' @param conn Optional DBI connection.
#' @param table Optional destination table when `conn` is supplied.
#' @param replace Replace destination rows.
#' @return Normalized Monarch relation.
#' @export
ducksemantics_monarch_import <- function(facts, releases,
                                         relation = c("gene_phenotype", "gene_disease"),
                                         conn = NULL, table = NULL, replace = FALSE) {
  relation <- match.arg(relation)
  catalog <- ducksemantics_monarch_release_catalog(releases, conn)
  rows <- ducksemantics_monarch_normalize(ducksemantics_collect_relation(facts, conn, "facts"), relation)
  ducksemantics_monarch_validate_fact_catalog(rows, catalog)
  if (!is.null(conn)) {
    ducksemantics_require_connection(conn)
    table <- ducksemantics_require_identifier(table %||% paste0("semantic_monarch_", relation), "table", TRUE)
    DBI::dbWriteTable(conn, table, rows, overwrite = ducksemantics_require_flag(replace, "replace"), append = !replace)
  } else if (!is.null(table)) {
    stop("`table` requires `conn`.", call. = FALSE)
  }
  rows
}

#' Select one cataloged Monarch release
#'
#' @param facts Monarch fact relation or a caller-owned DuckDB table name.
#' @param releases Typed Monarch release catalog.
#' @param relation Relation type.
#' @param provider_id Provider identity to bind.
#' @param release_id Exact catalog release identity.
#' @param as_of Typed `Date` or `POSIXct` cutoff.
#' @param as_of_ordinal Numeric source-order cutoff.
#' @param conn Optional DBI connection for table relations.
#' @return Facts from one selected provider/release. Selection metadata is held
#'   in `release_selected` and `provider_selected` attributes.
#' @export
ducksemantics_monarch_query <- function(facts, releases,
                                        relation = c("gene_phenotype", "gene_disease"),
                                        provider_id, release_id = NULL, as_of = NULL,
                                        as_of_ordinal = NULL, conn = NULL) {
  relation <- match.arg(relation)
  provider_id <- ducksemantics_require_text(provider_id, "provider_id")
  catalog <- ducksemantics_monarch_release_catalog(releases, conn)
  rows <- ducksemantics_monarch_normalize(ducksemantics_collect_relation(facts, conn, "facts"), relation)
  ducksemantics_monarch_validate_fact_catalog(rows, catalog)
  selected <- ducksemantics_monarch_select_release(
    catalog, provider_id, release_id = release_id, as_of = as_of,
    as_of_ordinal = as_of_ordinal
  )
  out <- rows[rows$provider_id == provider_id & rows$release_id == selected$release_id[[1L]], , drop = FALSE]
  row.names(out) <- NULL
  attr(out, "release_selected") <- selected$release_id[[1L]]
  attr(out, "provider_selected") <- provider_id
  attr(out, "effective_date") <- selected$effective_date[[1L]]
  attr(out, "source_ordinal") <- selected$source_ordinal[[1L]]
  out
}

#' Audit a historical Monarch gene-disease holdout
#'
#' The selected training and holdout releases must be exact rows of one typed
#' provider catalog. The result is a provenance-preserving anti-join.
#'
#' @param facts Caller-supplied gene-disease relation.
#' @param releases Typed Monarch release catalog.
#' @param provider_id Provider identity.
#' @param train_release_id Exact training release identity.
#' @param holdout_release_id Exact holdout release identity.
#' @param conn Optional DBI connection for table relations.
#' @return Holdout rows absent from the historical relation.
#' @export
ducksemantics_monarch_gene_disease_holdout_audit <- function(facts, releases,
                                                             provider_id,
                                                             train_release_id,
                                                             holdout_release_id,
                                                             conn = NULL) {
  train <- ducksemantics_monarch_query(facts, releases, "gene_disease", provider_id,
    release_id = train_release_id, conn = conn)
  holdout <- ducksemantics_monarch_query(facts, releases, "gene_disease", provider_id,
    release_id = holdout_release_id, conn = conn)
  train_key <- paste(train$gene_id, train$disease_id, train$predicate, sep = "\r")
  holdout_key <- paste(holdout$gene_id, holdout$disease_id, holdout$predicate, sep = "\r")
  out <- holdout[!holdout_key %in% train_key, , drop = FALSE]
  out$train_release_id <- train_release_id
  out$holdout_release_id <- holdout_release_id
  row.names(out) <- NULL
  out
}

ducksemantics_monarch_normalize <- function(facts, relation) {
  object <- if (identical(relation, "gene_phenotype")) "phenotype_id" else "disease_id"
  facts <- ducksemantics_require_data_frame(facts, "facts", c("gene_id", object, "release_id", "provider_id"))
  facts <- ducksemantics_add_missing(facts, c(
    predicate = if (identical(relation, "gene_phenotype")) "biolink:has_phenotype" else "biolink:associated_with",
    source_version = NA_character_, attrs = NA_character_
  ))
  facts <- facts[, c("gene_id", object, "predicate", "release_id", "provider_id", "source_version", "attrs"), drop = FALSE]
  ducksemantics_validate_required_text(facts, c("gene_id", object, "predicate", "release_id", "provider_id"), "facts")
  ducksemantics_validate_optional_text(facts, c("source_version", "attrs"), "facts")
  unique(facts)
}

ducksemantics_monarch_validate_fact_catalog <- function(facts, catalog) {
  fact_key <- paste(facts$provider_id, facts$release_id, sep = "\r")
  catalog_key <- paste(catalog$provider_id, catalog$release_id, sep = "\r")
  unknown <- unique(fact_key[!fact_key %in% catalog_key])
  if (length(unknown)) {
    stop("Monarch facts reference provider/release rows absent from `releases`.", call. = FALSE)
  }
  invisible(TRUE)
}

ducksemantics_monarch_select_release <- function(catalog, provider_id,
                                                  release_id = NULL, as_of = NULL,
                                                  as_of_ordinal = NULL) {
  provided <- c(!is.null(release_id), !is.null(as_of), !is.null(as_of_ordinal))
  if (sum(provided) != 1L) {
    stop("Supply exactly one of `release_id`, typed `as_of`, or `as_of_ordinal`.", call. = FALSE)
  }
  candidates <- catalog[catalog$provider_id == provider_id, , drop = FALSE]
  if (!nrow(candidates)) stop("Unknown Monarch provider: ", provider_id, call. = FALSE)
  if (!is.null(release_id)) {
    release_id <- ducksemantics_require_text(release_id, "release_id")
    selected <- candidates[candidates$release_id == release_id, , drop = FALSE]
    if (!nrow(selected)) stop("Unknown Monarch release for provider: ", release_id, call. = FALSE)
    return(selected)
  }
  if (!is.null(as_of)) {
    if (length(as_of) != 1L) stop("`as_of` must be one typed Date or POSIXct value.", call. = FALSE)
    cutoff <- ducksemantics_typed_time(as_of, "as_of")
    times <- ducksemantics_typed_time(candidates$effective_date, "releases$effective_date", allow_na = TRUE)
    selected <- candidates[!is.na(times) & times <= cutoff, , drop = FALSE]
    if (!nrow(selected)) stop("No Monarch release is visible at `as_of`.", call. = FALSE)
    times <- ducksemantics_typed_time(selected$effective_date, "releases$effective_date", allow_na = TRUE)
    return(selected[order(times, selected$release_id, decreasing = TRUE)[[1L]], , drop = FALSE])
  }
  if (!is.numeric(as_of_ordinal) || length(as_of_ordinal) != 1L || is.na(as_of_ordinal) || !is.finite(as_of_ordinal)) {
    stop("`as_of_ordinal` must be one finite numeric scalar.", call. = FALSE)
  }
  selected <- candidates[!is.na(candidates$source_ordinal) & candidates$source_ordinal <= as_of_ordinal, , drop = FALSE]
  if (!nrow(selected)) stop("No Monarch release is visible at `as_of_ordinal`.", call. = FALSE)
  selected[order(selected$source_ordinal, selected$release_id, decreasing = TRUE)[[1L]], , drop = FALSE]
}

#' Validate an append-only literature snapshot catalog
#'
#' `RClinVarbitration` owns article and section storage. This catalog identifies
#' a provider snapshot by its source-order cutoff; it creates no shadow
#' literature table.
#'
#' @param snapshots Data frame or caller-owned DuckDB table name containing
#'   `provider_id`, `snapshot_id`, and finite integer `high_water_ordinal`;
#'   `effective_at` is an optional typed `POSIXct` timestamp.
#' @param conn Optional DBI connection for a table relation.
#' @return Validated source snapshot catalog.
#' @export
ducksemantics_literature_snapshot_catalog <- function(snapshots, conn = NULL) {
  snapshots <- ducksemantics_collect_relation(snapshots, conn, "snapshots")
  snapshots <- ducksemantics_require_data_frame(
    snapshots, "snapshots", c("provider_id", "snapshot_id", "high_water_ordinal")
  )
  if (!"effective_at" %in% names(snapshots)) {
    snapshots$effective_at <- as.POSIXct(rep(NA_real_, nrow(snapshots)), origin = "1970-01-01", tz = "UTC")
  }
  snapshots <- snapshots[, c("provider_id", "snapshot_id", "high_water_ordinal", "effective_at"), drop = FALSE]
  ducksemantics_validate_required_text(snapshots, c("provider_id", "snapshot_id"), "snapshots")
  snapshots$high_water_ordinal <- ducksemantics_literature_ordinal(
    snapshots$high_water_ordinal, "snapshots$high_water_ordinal"
  )
  ducksemantics_literature_optional_timestamp(snapshots$effective_at, "snapshots$effective_at")
  if (anyDuplicated(snapshots[c("provider_id", "snapshot_id")])) {
    stop("`provider_id` and `snapshot_id` must uniquely identify snapshots.", call. = FALSE)
  }
  snapshots
}

#' Retrieve exact literature spans from append-only caller-owned relations
#'
#' `RClinVarbitration` owns the article and section relations. Retrieval binds a
#' provider and exact cataloged source-order cutoff, selects the maximum article
#' `source_ordinal` at or below that cutoff per provider/article, drops a latest
#' deletion event, and joins only sections from that exact version. It never
#' imports, writes, or fetches literature.
#'
#' @details A read-only RClinVarbitration projection joins `pubmed_sources` to
#' `pubmed_articles` and `pubmed_abstracts`: use source provider as
#' `provider_id`, source id as `version_id`/`snapshot_id`, source ordinal for
#' both ordinal fields, and `source_applied_at` as `effective_at`. A section
#' must retain the exact article event's version id and source ordinal.
#' @param articles Caller-owned relation with `provider_id`, `article_id`,
#'   `pmid`, `version_id`, finite integer `source_ordinal`, and logical
#'   `is_deleted`.
#' @param sections Caller-owned relation with `provider_id`, `article_id`,
#'   `pmid`, `version_id`, finite integer `source_ordinal`, `section`, and
#'   exact non-empty `text`.
#' @param snapshots Source snapshot catalog from
#'   [ducksemantics_literature_snapshot_catalog()].
#' @param query Non-empty lexical query text.
#' @param provider_id Provider identity.
#' @param snapshot_id Exact provider snapshot identity.
#' @param section_names Optional non-empty section names to search; `NULL`
#'   searches every version-bound section.
#' @param conn Optional DBI connection for table relations.
#' @return Exact non-empty source spans from non-deleted article versions visible
#'   at the cataloged source-order cutoff.
#' @export
ducksemantics_literature_retrieve <- function(articles, sections, snapshots,
                                             query, provider_id, snapshot_id,
                                             section_names = NULL,
                                             conn = NULL) {
  articles <- ducksemantics_literature_articles(ducksemantics_collect_relation(articles, conn, "articles"))
  sections <- ducksemantics_literature_sections(ducksemantics_collect_relation(sections, conn, "sections"))
  snapshots <- ducksemantics_literature_snapshot_catalog(snapshots, conn)
  ducksemantics_literature_validate_section_versions(articles, sections)
  query <- ducksemantics_require_text(query, "query")
  provider_id <- ducksemantics_require_text(provider_id, "provider_id")
  snapshot_id <- ducksemantics_require_text(snapshot_id, "snapshot_id")
  if (!is.null(section_names) && (!is.character(section_names) || !length(section_names) ||
    anyNA(section_names) || any(!nzchar(section_names)))) {
    stop("`section_names` must be NULL or non-empty character names.", call. = FALSE)
  }
  snapshot <- snapshots[snapshots$provider_id == provider_id & snapshots$snapshot_id == snapshot_id, , drop = FALSE]
  if (!nrow(snapshot)) stop("Unknown literature snapshot for provider: ", snapshot_id, call. = FALSE)
  high_water_ordinal <- snapshot$high_water_ordinal[[1L]]
  articles <- articles[articles$provider_id == provider_id, , drop = FALSE]
  visible <- ducksemantics_literature_visible_articles(articles, high_water_ordinal)
  if (!nrow(visible)) return(ducksemantics_empty_literature_hits(snapshot))
  section_key <- ducksemantics_literature_version_key(sections)
  visible_key <- ducksemantics_literature_version_key(visible)
  matched <- sections[section_key %in% visible_key & sections$provider_id == provider_id, , drop = FALSE]
  if (!is.null(section_names)) matched <- matched[matched$section %in% section_names, , drop = FALSE]
  if (!nrow(matched)) return(ducksemantics_empty_literature_hits(snapshot))
  matches <- lapply(seq_len(nrow(matched)), function(i) {
    starts <- gregexpr(tolower(query), tolower(matched$text[[i]]), fixed = TRUE)[[1L]]
    if (identical(starts, -1L)) return(NULL)
    widths <- attr(starts, "match.length")
    data.frame(
      provider_id = rep(provider_id, length(starts)), snapshot_id = rep(snapshot_id, length(starts)),
      high_water_ordinal = rep(high_water_ordinal, length(starts)),
      effective_at = snapshot$effective_at[rep.int(1L, length(starts))],
      article_id = rep(matched$article_id[[i]], length(starts)), pmid = rep(matched$pmid[[i]], length(starts)),
      version_id = rep(matched$version_id[[i]], length(starts)), source_ordinal = rep(matched$source_ordinal[[i]], length(starts)),
      section = rep(matched$section[[i]], length(starts)), start_offset = as.integer(starts - 1L),
      end_offset = as.integer(starts + widths - 1L),
      source_text = substring(matched$text[[i]], starts, starts + widths - 1L),
      section_text = rep(matched$text[[i]], length(starts)), stringsAsFactors = FALSE
    )
  })
  out <- ducksemantics_bind_or_empty(matches, ducksemantics_empty_literature_hits(snapshot))
  if (nrow(out)) {
    ducksemantics_validate_half_open_offsets(out$start_offset, out$end_offset, "literature result")
    if (any(!nzchar(out$source_text))) stop("Literature retrieval produced an empty source span.", call. = FALSE)
  }
  attr(out, "provider_id") <- provider_id
  attr(out, "snapshot_id") <- snapshot_id
  attr(out, "high_water_ordinal") <- high_water_ordinal
  attr(out, "effective_at") <- snapshot$effective_at[1L]
  out
}

ducksemantics_literature_articles <- function(articles) {
  articles <- ducksemantics_require_data_frame(
    articles, "articles", c("provider_id", "article_id", "pmid", "version_id", "source_ordinal", "is_deleted")
  )
  articles <- articles[, c("provider_id", "article_id", "pmid", "version_id", "source_ordinal", "is_deleted"), drop = FALSE]
  ducksemantics_validate_required_text(articles, c("provider_id", "article_id", "pmid", "version_id"), "articles")
  articles$source_ordinal <- ducksemantics_literature_ordinal(
    articles$source_ordinal, "articles$source_ordinal"
  )
  if (!is.logical(articles$is_deleted) || anyNA(articles$is_deleted)) {
    stop("`articles$is_deleted` must contain logical values without NA.", call. = FALSE)
  }
  if (anyDuplicated(articles[c("provider_id", "article_id", "version_id")])) {
    stop("Article provider/id/version rows must be unique.", call. = FALSE)
  }
  if (anyDuplicated(articles[c("provider_id", "article_id", "source_ordinal")])) {
    stop("Article provider/id/source-ordinal rows must be unique.", call. = FALSE)
  }
  articles
}

ducksemantics_literature_sections <- function(sections) {
  sections <- ducksemantics_require_data_frame(
    sections, "sections", c("provider_id", "article_id", "pmid", "version_id", "source_ordinal", "section", "text")
  )
  sections <- sections[, c("provider_id", "article_id", "pmid", "version_id", "source_ordinal", "section", "text"), drop = FALSE]
  ducksemantics_validate_required_text(
    sections, c("provider_id", "article_id", "pmid", "version_id", "section", "text"), "sections"
  )
  sections$source_ordinal <- ducksemantics_literature_ordinal(
    sections$source_ordinal, "sections$source_ordinal"
  )
  sections
}

ducksemantics_literature_validate_section_versions <- function(articles, sections) {
  article_key <- ducksemantics_literature_version_key(articles)
  section_key <- ducksemantics_literature_version_key(sections)
  if (length(setdiff(unique(section_key), article_key))) {
    stop("Every literature section must bind a supplied provider/article/PMID/version/source-ordinal row.", call. = FALSE)
  }
  invisible(TRUE)
}

ducksemantics_literature_version_key <- function(rows) {
  paste(rows$provider_id, rows$article_id, rows$pmid, rows$version_id, rows$source_ordinal, sep = "\r")
}

ducksemantics_literature_visible_articles <- function(articles, high_water_ordinal) {
  if (!nrow(articles)) return(articles)
  eligible <- articles[articles$source_ordinal <= high_water_ordinal, , drop = FALSE]
  if (!nrow(eligible)) return(eligible)
  eligible <- eligible[order(eligible$provider_id, eligible$article_id, -eligible$source_ordinal), , drop = FALSE]
  latest <- eligible[!duplicated(eligible[c("provider_id", "article_id")]), , drop = FALSE]
  latest[!latest$is_deleted, , drop = FALSE]
}

ducksemantics_empty_literature_hits <- function(snapshot) {
  data.frame(
    provider_id = character(), snapshot_id = character(), high_water_ordinal = numeric(),
    effective_at = snapshot$effective_at[FALSE], article_id = character(), pmid = character(),
    version_id = character(), source_ordinal = numeric(), section = character(),
    start_offset = integer(), end_offset = integer(), source_text = character(), section_text = character(),
    stringsAsFactors = FALSE
  )
}

ducksemantics_literature_ordinal <- function(x, arg) {
  if (!is.numeric(x) || anyNA(x) || any(!is.finite(x)) || any(x != trunc(x))) {
    stop("`", arg, "` must contain finite integer values without NA.", call. = FALSE)
  }
  as.numeric(x)
}

ducksemantics_literature_optional_timestamp <- function(x, arg) {
  if (!inherits(x, "POSIXct")) {
    stop("`", arg, "` must be a POSIXct timestamp or omitted.", call. = FALSE)
  }
  numeric <- as.numeric(x)
  if (any(!is.na(numeric) & !is.finite(numeric))) {
    stop("`", arg, "` must contain finite timestamps or NA.", call. = FALSE)
  }
  invisible(x)
}

ducksemantics_typed_time <- function(x, arg, allow_na = FALSE) {
  if (!(inherits(x, "Date") || inherits(x, "POSIXct"))) {
    stop("`", arg, "` must be a Date or POSIXct value.", call. = FALSE)
  }
  if (!allow_na && (length(x) != 1L && anyNA(x) || length(x) == 1L && is.na(x))) {
    stop("`", arg, "` must not contain NA.", call. = FALSE)
  }
  numeric <- as.numeric(x)
  if (any(!is.na(numeric) & !is.finite(numeric))) stop("`", arg, "` must contain finite times.", call. = FALSE)
  numeric
}
