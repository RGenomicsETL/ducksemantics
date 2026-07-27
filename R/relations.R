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
#' @param facts Gene-phenotype, gene-disease, or disease-phenotype data frame,
#'   or a caller-owned DuckDB table name with `conn`.
#' @param releases Typed catalog from [ducksemantics_monarch_release_catalog()].
#' @param relation One of `"gene_phenotype"`, `"gene_disease"`, or
#'   `"disease_phenotype"`.
#' @param conn Optional DBI connection.
#' @param table Optional destination table when `conn` is supplied.
#' @param replace Replace destination rows.
#' @return Normalized Monarch relation.
#' @export
ducksemantics_monarch_import <- function(facts, releases,
                                         relation = c("gene_phenotype", "gene_disease", "disease_phenotype"),
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
                                        relation = c("gene_phenotype", "gene_disease", "disease_phenotype"),
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
  roles <- ducksemantics_monarch_relation_roles(relation)
  facts <- ducksemantics_require_data_frame(facts, "facts", c(roles, "release_id", "provider_id"))
  facts <- ducksemantics_add_missing(facts, c(
    predicate = if (identical(roles[[2L]], "phenotype_id")) "biolink:has_phenotype" else "biolink:associated_with",
    source_version = NA_character_, attrs = NA_character_
  ))
  facts <- facts[, c(roles, "predicate", "release_id", "provider_id", "source_version", "attrs"), drop = FALSE]
  ducksemantics_validate_required_text(facts, c(roles, "predicate", "release_id", "provider_id"), "facts")
  ducksemantics_validate_optional_text(facts, c("source_version", "attrs"), "facts")
  unique(facts)
}

ducksemantics_monarch_relation_roles <- function(relation) {
  switch(relation,
    gene_phenotype = c("gene_id", "phenotype_id"),
    gene_disease = c("gene_id", "disease_id"),
    disease_phenotype = c("disease_id", "phenotype_id"),
    stop("Unsupported Monarch relation: ", relation, call. = FALSE)
  )
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

#' Project one attached, dated Monarch annotation pack
#'
#' Projects an already attached read-only official Monarch DuckDB pack into
#' connection-local temporary views. It never attaches, downloads, collects,
#' copies, rewrites, or merges pack rows. `source_catalog` must name an attached
#' DuckDB catalog whose `main`
#' schema contains the official `denormalized_edges`, `nodes`,
#' `information_content`, and `node_has_phenotype` tables. The edge table is
#' checked before any view is made, including its scalar role categories,
#' provider-serialized text negation, and array provenance fields. The official
#' required `nodes`, `node_has_phenotype`, and `information_content` projection
#' columns/types are also validated.
#'
#' `associations` retains every source edge and all source columns exactly, then
#' adds the selected provider and release. Each role view is likewise a total
#' projection: it retains raw source columns and adds normalized role columns,
#' `source_subject`, `source_object`, `source_direction`, and
#' `association_status`. Only `supported_subject_to_object` and
#' `supported_object_to_subject` rows have non-missing normalized roles.
#' `missing_endpoint`, `missing_predicate`, `malformed_negation`,
#' `missing_role_category`, `malformed_role_category`, `negated`, and
#' `unsupported_orientation` are not support. `negation_status` separately
#' distinguishes `not_provided`, `not_negated`, `negated`, and `malformed`:
#' Monarch omits the optional qualifier on many positive assertion rows, so
#' omission is retained but does not erase the assertion. In particular, a
#' missing predicate never becomes a causal predicate.
#'
#' @param conn Valid DBI connection with the pack already attached.
#' @param source_catalog Attached DuckDB catalog name, not the current catalog.
#' @param releases Typed Monarch release catalog or caller-owned catalog table.
#'   This binds provider/release semantics; the caller must verify the immutable
#'   file receipt before attaching the pack.
#' @param provider_id Exact provider identity in `releases`.
#' @param release_id Exact, dated release identity in `releases`; `"latest"` is
#'   rejected.
#' @param prefix Safe unqualified prefix for temporary view names.
#' @return A data frame naming the seven connection-local temporary views, with
#'   `relation`, `view_name`, `provider_id`, and `release_id` columns. It does
#'   not return pack contents.
#' @export
ducksemantics_monarch_project_pack <- function(conn, source_catalog, releases,
                                               provider_id, release_id,
                                               prefix = "semantic_monarch") {
  ducksemantics_require_connection(conn)
  source_catalog <- ducksemantics_require_identifier(source_catalog, "source_catalog")
  prefix <- ducksemantics_require_identifier(prefix, "prefix")
  provider_id <- ducksemantics_require_text(provider_id, "provider_id")
  release_id <- ducksemantics_require_text(release_id, "release_id")
  if (identical(tolower(release_id), "latest")) {
    stop("`release_id` must name one exact dated release; `latest` is not allowed.", call. = FALSE)
  }
  source_catalog <- ducksemantics_monarch_pack_attached_catalog(conn, source_catalog)
  catalog <- ducksemantics_monarch_release_catalog(releases, conn)
  selected <- catalog[catalog$provider_id == provider_id & catalog$release_id == release_id, , drop = FALSE]
  if (nrow(selected) != 1L) {
    stop("`provider_id` and `release_id` must identify exactly one typed Monarch release catalog row.", call. = FALSE)
  }
  ducksemantics_monarch_pack_validate_source(conn, source_catalog)
  views <- ducksemantics_monarch_pack_view_names(prefix)
  association_view <- views[["associations"]]
  node_view <- views[["nodes"]]
  node_has_phenotype_view <- views[["node_has_phenotype"]]
  information_content_view <- views[["information_content"]]
  DBI::dbWithTransaction(conn, {
    DBI::dbExecute(conn, glue::glue_sql(
      "CREATE OR REPLACE TEMP VIEW {`association_view`} AS
       SELECT e.*, {provider_id} AS provider_id, {release_id} AS release_id
       FROM {`source_catalog`}.main.denormalized_edges AS e",
      .con = conn
    ))
    relations <- list(
      gene_disease = c(first = "gene", first_category = "biolink:Gene", second = "disease", second_category = "biolink:Disease"),
      disease_phenotype = c(first = "disease", first_category = "biolink:Disease", second = "phenotype", second_category = "biolink:PhenotypicFeature"),
      gene_phenotype = c(first = "gene", first_category = "biolink:Gene", second = "phenotype", second_category = "biolink:PhenotypicFeature")
    )
    for (relation in names(relations)) {
      spec <- relations[[relation]]
      DBI::dbExecute(conn, ducksemantics_monarch_pack_role_view_sql(
        conn, views[["associations"]], views[[relation]], spec[["first"]],
        spec[["first_category"]], spec[["second"]], spec[["second_category"]]
      ))
    }
    DBI::dbExecute(conn, glue::glue_sql(
      "CREATE OR REPLACE TEMP VIEW {`node_view`} AS
       SELECT n.*, {provider_id} AS provider_id, {release_id} AS release_id
       FROM {`source_catalog`}.main.nodes AS n",
      .con = conn
    ))
    DBI::dbExecute(conn, glue::glue_sql(
      "CREATE OR REPLACE TEMP VIEW {`node_has_phenotype_view`} AS
       SELECT p.*, {provider_id} AS provider_id, {release_id} AS release_id
       FROM {`source_catalog`}.main.node_has_phenotype AS p",
      .con = conn
    ))
    DBI::dbExecute(conn, glue::glue_sql(
      "CREATE OR REPLACE TEMP VIEW {`information_content_view`} AS
       SELECT i.*, {provider_id} AS provider_id, {release_id} AS release_id
       FROM {`source_catalog`}.main.information_content AS i",
      .con = conn
    ))
  })
  data.frame(
    relation = names(views), view_name = unname(views),
    provider_id = rep(provider_id, length(views)),
    release_id = rep(release_id, length(views)),
    stringsAsFactors = FALSE
  )
}

ducksemantics_monarch_pack_view_names <- function(prefix) {
  c(
    associations = paste0(prefix, "_associations"),
    gene_disease = paste0(prefix, "_gene_disease"),
    disease_phenotype = paste0(prefix, "_disease_phenotype"),
    gene_phenotype = paste0(prefix, "_gene_phenotype"),
    nodes = paste0(prefix, "_nodes"),
    node_has_phenotype = paste0(prefix, "_node_has_phenotype"),
    information_content = paste0(prefix, "_information_content")
  )
}

ducksemantics_monarch_pack_attached_catalog <- function(conn, source_catalog) {
  databases <- DBI::dbGetQuery(
    conn,
    "SELECT database_name, type, readonly FROM duckdb_databases()"
  )
  current <- DBI::dbGetQuery(conn, "SELECT current_database() AS database_name")$database_name[[1L]]
  matching <- databases[tolower(databases$database_name) == tolower(source_catalog), , drop = FALSE]
  if (nrow(matching) != 1L || identical(tolower(source_catalog), tolower(current)) ||
      !identical(tolower(matching$type[[1L]]), "duckdb") ||
      !isTRUE(matching$readonly[[1L]])) {
    stop(
      "`source_catalog` must name one already attached read-only DuckDB catalog, not the current catalog.",
      call. = FALSE
    )
  }
  matching$database_name[[1L]]
}

ducksemantics_monarch_pack_validate_source <- function(conn, source_catalog) {
  table_names <- c("denormalized_edges", "nodes", "information_content", "node_has_phenotype")
  tables <- DBI::dbGetQuery(conn, glue::glue_sql(
    "SELECT table_name, table_type
     FROM information_schema.tables
     WHERE table_catalog = {source_catalog}
       AND table_schema = 'main'
       AND table_name IN ('denormalized_edges', 'nodes', 'information_content', 'node_has_phenotype')",
    .con = conn
  ))
  expected_types <- c(
    denormalized_edges = "VIEW", nodes = "BASE TABLE",
    information_content = "BASE TABLE", node_has_phenotype = "BASE TABLE"
  )
  relation_types <- stats::setNames(tables$table_type, tables$table_name)
  incompatible <- table_names[
    is.na(relation_types[table_names]) |
      relation_types[table_names] != expected_types[table_names]
  ]
  if (length(incompatible)) {
    stop(
      "Attached Monarch pack is missing required official relations or relation types: ",
      paste(incompatible, collapse = ", "), ".",
      call. = FALSE
    )
  }
  columns <- DBI::dbGetQuery(conn, glue::glue_sql(
    "SELECT table_name, column_name, data_type
     FROM information_schema.columns
     WHERE table_catalog = {source_catalog}
       AND table_schema = 'main'
       AND table_name IN ('denormalized_edges', 'nodes', 'information_content', 'node_has_phenotype')",
    .con = conn
  ))
  ducksemantics_monarch_pack_validate_table(
    columns, "denormalized_edges",
    text = c(
      "id", "predicate", "category", "agent_type", "knowledge_level",
      "primary_knowledge_source", "file_source", "provided_by",
      "disease_context_qualifier", "original_predicate", "frequency_qualifier",
      "onset_qualifier", "sex_qualifier", "object_aspect_qualifier",
      "species_context_qualifier", "stage_qualifier", "qualifier",
      "object_specialization_qualifier", "negated", "subject", "object",
      "original_subject", "original_object", "grouping_key", "subject_label",
      "subject_category", "subject_namespace", "subject_taxon", "object_label",
      "object_category", "object_namespace", "object_taxon"
    ),
    text_list = c(
      "aggregator_knowledge_source", "has_evidence", "publications",
      "qualifiers", "subject_closure", "object_closure"
    ),
    numeric = "evidence_count"
  )
  ducksemantics_monarch_pack_validate_table(
    columns, "nodes",
    text = c("id", "category", "name", "namespace"),
    text_list = c("xref", "synonym")
  )
  ducksemantics_monarch_pack_validate_table(
    columns, "information_content", text = "term", numeric = "ic"
  )
  ducksemantics_monarch_pack_validate_table(
    columns, "node_has_phenotype", text = "id",
    text_list = c(
      "has_phenotype", "has_phenotype_label", "has_phenotype_closure",
      "has_phenotype_closure_label"
    ),
    numeric = "has_phenotype_count"
  )
  invisible(TRUE)
}

ducksemantics_monarch_pack_validate_table <- function(
  columns, table, text = character(), text_list = character(), numeric = character()
) {
  table_columns <- columns[columns$table_name == table, , drop = FALSE]
  required <- c(text, text_list, numeric)
  missing <- setdiff(required, table_columns$column_name)
  if (length(missing)) {
    stop(
      "Attached Monarch `", table, "` is missing required columns: ",
      paste(missing, collapse = ", "), ".",
      call. = FALSE
    )
  }
  types <- stats::setNames(toupper(table_columns$data_type), table_columns$column_name)
  invalid_text <- text[!vapply(types[text], ducksemantics_monarch_pack_is_text, logical(1))]
  invalid_lists <- text_list[!vapply(types[text_list], ducksemantics_monarch_pack_is_text_list, logical(1))]
  invalid_numeric <- numeric[!vapply(types[numeric], ducksemantics_monarch_pack_is_numeric, logical(1))]
  invalid <- c(invalid_text, invalid_lists, invalid_numeric)
  if (length(invalid)) {
    stop(
      "Attached Monarch `", table, "` has incompatible column types: ",
      paste(invalid, collapse = ", "), ".",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

ducksemantics_monarch_pack_is_text <- function(type) {
  is.character(type) && length(type) == 1L && !is.na(type) && type %in% c("VARCHAR", "TEXT", "STRING")
}

ducksemantics_monarch_pack_is_text_list <- function(type) {
  is.character(type) && length(type) == 1L && !is.na(type) && grepl("^(VARCHAR|TEXT|STRING)\\[\\]$", type)
}

ducksemantics_monarch_pack_is_numeric <- function(type) {
  is.character(type) && length(type) == 1L && !is.na(type) && grepl("^(U?TINYINT|U?SMALLINT|U?INTEGER|U?BIGINT|HUGEINT|UHUGEINT|FLOAT|DOUBLE|DECIMAL.*)$", type)
}

ducksemantics_monarch_pack_role_view_sql <- function(conn, association_view, view,
                                                      first, first_category,
                                                      second, second_category) {
  subject_first <- ducksemantics_monarch_pack_category_sql(conn, "subject_category", first_category)
  object_first <- ducksemantics_monarch_pack_category_sql(conn, "object_category", first_category)
  subject_second <- ducksemantics_monarch_pack_category_sql(conn, "subject_category", second_category)
  object_second <- ducksemantics_monarch_pack_category_sql(conn, "object_category", second_category)
  subject_missing <- ducksemantics_monarch_pack_missing_category_sql(conn, "subject_category")
  object_missing <- ducksemantics_monarch_pack_missing_category_sql(conn, "object_category")
  subject_malformed <- ducksemantics_monarch_pack_malformed_category_sql(conn, "subject_category")
  object_malformed <- ducksemantics_monarch_pack_malformed_category_sql(conn, "object_category")
  role_columns <- DBI::SQL(paste(c(
    ducksemantics_monarch_pack_role_columns_sql(conn, first, "subject", "object"),
    ducksemantics_monarch_pack_role_columns_sql(conn, second, "object", "subject")
  ), collapse = ",\n"))
  glue::glue_sql(
    "CREATE OR REPLACE TEMP VIEW {`view`} AS
     WITH typed AS (
       SELECT a.*,
         CASE
           WHEN a.negated IS NULL OR trim(a.negated) = '' THEN 'not_provided'
           WHEN lower(trim(a.negated)) = 'false' THEN 'not_negated'
           WHEN lower(trim(a.negated)) = 'true' THEN 'negated'
           ELSE 'malformed'
         END AS negation_status
       FROM {`association_view`} AS a
     ), classified AS (
       SELECT a.*,
         CASE
           WHEN a.subject IS NULL OR trim(a.subject) = '' OR a.object IS NULL OR trim(a.object) = '' THEN 'missing_endpoint'
           WHEN a.predicate IS NULL OR trim(a.predicate) = '' THEN 'missing_predicate'
           WHEN a.negation_status = 'negated' THEN 'negated'
           WHEN a.negation_status = 'malformed' THEN 'malformed_negation'
           WHEN {subject_missing} OR {object_missing} THEN 'missing_role_category'
           WHEN {subject_malformed} OR {object_malformed} THEN 'malformed_role_category'
           WHEN {subject_first} AND {object_second} THEN 'supported_subject_to_object'
           WHEN {subject_second} AND {object_first} THEN 'supported_object_to_subject'
           ELSE 'unsupported_orientation'
         END AS association_status
       FROM typed AS a
     )
     SELECT classified.*, classified.id AS edge_id,
       classified.subject AS source_subject,
       classified.object AS source_object,
       CASE
         WHEN classified.association_status = 'supported_subject_to_object' THEN 'subject_to_object'
         WHEN classified.association_status = 'supported_object_to_subject' THEN 'object_to_subject'
         ELSE NULL
       END AS source_direction,
       {role_columns}
     FROM classified",
    .con = conn
  )
}

ducksemantics_monarch_pack_category_sql <- function(conn, column, category) {
  glue::glue_sql("a.{`column`} = {category}", .con = conn)
}

ducksemantics_monarch_pack_missing_category_sql <- function(conn, column) {
  glue::glue_sql("a.{`column`} IS NULL OR trim(a.{`column`}) = ''", .con = conn)
}

ducksemantics_monarch_pack_malformed_category_sql <- function(conn, column) {
  glue::glue_sql("NOT starts_with(a.{`column`}, 'biolink:')", .con = conn)
}

ducksemantics_monarch_pack_role_columns_sql <- function(conn, role, forward, reverse) {
  suffixes <- c(id = "", label = "label", category = "category", namespace = "namespace", closure = "closure", taxon = "taxon")
  vapply(names(suffixes), function(suffix) {
    forward_column <- if (identical(suffix, "id")) forward else paste0(forward, "_", suffixes[[suffix]])
    reverse_column <- if (identical(suffix, "id")) reverse else paste0(reverse, "_", suffixes[[suffix]])
    output_column <- paste0(role, "_", suffix)
    as.character(glue::glue_sql(
      "CASE
         WHEN classified.association_status = 'supported_subject_to_object' THEN classified.{`forward_column`}
         WHEN classified.association_status = 'supported_object_to_subject' THEN classified.{`reverse_column`}
         ELSE NULL
       END AS {`output_column`}",
      .con = conn
    ))
  }, character(1))
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
