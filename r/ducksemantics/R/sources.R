#' Validate source release availability and receipts
#'
#' `available_at` is the provider availability epoch, not an ontology version
#' string, commit author date, or ingestion time. Callers declare this timestamp
#' from a release receipt. Additional receipt columns are retained.
#'
#' @param releases Data frame containing `provider_id`, `release_id`,
#'   `available_at` (POSIXct), and `receipt_id`.
#' @param conn Optional DBI connection for a caller-owned table.
#' @return Validated release catalog.
#' @export
ducksemantics_source_release_catalog <- function(releases, conn = NULL) {
  releases <- ducksemantics_require_data_frame(
    ducksemantics_collect_relation(releases, conn, "releases"), "releases",
    c("provider_id", "release_id", "available_at", "receipt_id")
  )
  ducksemantics_validate_required_text(releases,
    c("provider_id", "release_id", "receipt_id"), "releases")
  if (!inherits(releases$available_at, "POSIXct")) {
    stop("`available_at` must be POSIXct provider availability timestamps.", call. = FALSE)
  }
  ducksemantics_typed_time(releases$available_at, "available_at")
  if (anyDuplicated(releases[c("provider_id", "release_id")])) {
    stop("Provider/release identities must be unique.", call. = FALSE)
  }
  if (anyDuplicated(releases[c("provider_id", "available_at")])) {
    stop("Provider availability epochs must be unambiguous.", call. = FALSE)
  }
  releases
}

#' Select a source release by provider availability
#'
#' Selection reuses the Monarch release-selection authority, binding its time
#' column to `available_at`. Exactly one of `release_id` and `as_of` is required.
#'
#' @inheritParams ducksemantics_source_release_catalog
#' @param provider_id Provider to select.
#' @param release_id Exact release identity.
#' @param as_of Typed POSIXct provider availability cutoff.
#' @return One catalog row. A cutoff before provider availability is an error.
#' @export
ducksemantics_source_release_select <- function(
  releases, provider_id, release_id = NULL, as_of = NULL, conn = NULL
) {
  catalog <- ducksemantics_source_release_catalog(releases, conn)
  if (!is.null(as_of) && !inherits(as_of, "POSIXct")) {
    stop("`as_of` must be a POSIXct provider availability cutoff.", call. = FALSE)
  }
  selection <- catalog
  selection$effective_date <- selection$available_at
  selection$source_ordinal <- NA_real_
  selected <- ducksemantics_monarch_select_release(
    selection, ducksemantics_require_text(provider_id, "provider_id"),
    release_id = release_id, as_of = as_of
  )
  catalog[catalog$provider_id == selected$provider_id &
          catalog$release_id == selected$release_id, , drop = FALSE]
}

#' Import release-bound Mondo source relations
#'
#' Reads the existing OBO projection with obsolete terms included. Raw tags
#' retain synonym qualifiers, mapping annotations, replacement/consider tags,
#' and source-line identity. An OBO xref is a mapping assertion, not a claim of
#' exact disease equivalence. Nothing is written to the unversioned graph.
#'
#' @param path Local source file. No network access is performed.
#' @param release One selected row of [ducksemantics_source_release_catalog()].
#' @param source_id Exact source artifact identity within the release.
#' @param source_uri Pinned public source URI for the artifact.
#' @param sha256 Expected SHA-256 of the supplied bytes.
#' @return A list of release, artifact, term, synonym, mapping, obsolescence,
#'   edge, and exact tag relations, all carrying provider/release/source keys.
#' @export
ducksemantics_mondo_import <- function(path, release, source_id, source_uri, sha256) {
  artifact <- ducksemantics_source_artifact(path, release, source_id, source_uri, sha256)
  graph <- ducksemantics_read_obo(path, "MONDO", source_id, include_obsolete = TRUE)
  ids <- graph$nodes$node_id[startsWith(graph$nodes$node_id, "MONDO:")]
  if (anyDuplicated(ids)) stop("Mondo identifiers must be unique.", call. = FALSE)
  graph$aliases <- graph$aliases[graph$aliases$node_id %in% ids, , drop = FALSE]
  graph$edges <- graph$edges[graph$edges$from_id %in% ids, , drop = FALSE]
  graph$tags <- graph$tags[graph$tags$node_id %in% ids, , drop = FALSE]
  terms <- graph$nodes[graph$nodes$node_id %in% ids, , drop = FALSE]
  terms$is_obsolete <- terms$node_id %in% graph$tags$node_id[
    graph$tags$tag == "is_obsolete" & tolower(graph$tags$value) == "true"
  ]
  tags <- graph$tags
  tags$source_row_id <- paste0(source_id, "#", tags$node_id, ":", tags$tag_ordinal)
  list(
    release = release, artifact = artifact,
    terms = ducksemantics_source_rows(terms, artifact),
    synonyms = ducksemantics_source_rows(graph$aliases, artifact),
    mappings = ducksemantics_source_rows(tags[
      tags$tag == "xref" | (tags$tag == "property_value" &
        grepl("^skos:(exactMatch|closeMatch|broadMatch|narrowMatch|relatedMatch) ", tags$value)),
      , drop = FALSE], artifact),
    obsolescence = ducksemantics_source_rows(tags[
      tags$tag %in% c("is_obsolete", "replaced_by", "consider"), , drop = FALSE], artifact),
    edges = ducksemantics_source_rows(graph$edges, artifact),
    tags = ducksemantics_source_rows(tags, artifact)
  )
}

#' Import a pinned DisMech source record
#'
#' Retains the decoded YAML record and every scalar field at its RFC 6901
#' JSON-pointer location (zero-based array indices). `source_row_id` is a local
#' record/field locator, not an invented upstream evidence accession. PMID,
#' ontology, mechanism, and snippet values remain exactly those in the record.
#' The import is a source projection, not contribution-schema validation and
#' not independent verification of cited literature. It emits no proposals.
#'
#' @inheritParams ducksemantics_mondo_import
#' @return A list of release, artifact, record, and field relations. Records and
#'   fields carry provider, release, receipt, and source identities.
#' @export
ducksemantics_dismech_import <- function(path, release, source_id, source_uri, sha256) {
  artifact <- ducksemantics_source_artifact(path, release, source_id, source_uri, sha256)
  record <- yaml::yaml.load_file(path, handlers = list(seq = as.list), eval.expr = FALSE)
  if (!is.list(record)) stop("Expected a DisMech YAML mapping.", call. = FALSE)
  name <- ducksemantics_require_text(record$name, "DisMech disease name")
  fields <- list()
  walk <- function(value, pointer) {
    if (is.list(value) || length(value) > 1L) {
      keys <- if (is.null(names(value))) as.character(seq_along(value) - 1L) else names(value)
      keys <- gsub("/", "~1", gsub("~", "~0", keys, fixed = TRUE), fixed = TRUE)
      for (i in seq_along(value)) walk(value[[i]], paste0(pointer, "/", keys[[i]]))
    } else {
      fields[[length(fields) + 1L]] <<- data.frame(
        field_path = pointer, value_type = if (is.null(value)) "null" else typeof(value),
        value = if (is.null(value)) NA_character_ else as.character(value),
        source_row_id = paste0(source_id, "#", pointer), stringsAsFactors = FALSE
      )
    }
  }
  walk(record, "")
  fields <- do.call(rbind, fields)
  if (anyDuplicated(fields$field_path)) stop("Duplicate source field locators.", call. = FALSE)
  text <- rawToChar(readBin(path, "raw", n = file.info(path)$size))
  records <- data.frame(name = name, yaml_text = text, stringsAsFactors = FALSE)
  list(release = release, artifact = artifact,
       records = ducksemantics_source_rows(records, artifact),
       fields = ducksemantics_source_rows(fields, artifact))
}

# Source admission verifies bytes against a caller-declared receipt.
ducksemantics_source_artifact <- function(path, release, source_id, source_uri, sha256) {
  release <- ducksemantics_source_release_catalog(release)
  if (nrow(release) != 1L) stop("Select exactly one source release.", call. = FALSE)
  path <- normalizePath(ducksemantics_require_text(path, "path"), mustWork = TRUE)
  source_id <- ducksemantics_require_text(source_id, "source_id")
  source_uri <- ducksemantics_require_text(source_uri, "source_uri")
  sha256 <- ducksemantics_require_text(sha256, "sha256")
  if (!grepl("^[0-9a-f]{64}$", sha256)) stop("Expected a lowercase SHA-256 digest.", call. = FALSE)
  if (!identical(digest::digest(file = path, algo = "sha256"), sha256)) {
    stop("Source artifact SHA-256 does not match its receipt.", call. = FALSE)
  }
  data.frame(release[c("provider_id", "release_id", "available_at", "receipt_id")],
             source_id = source_id, source_uri = source_uri, sha256 = sha256,
             stringsAsFactors = FALSE)
}

ducksemantics_source_rows <- function(rows, artifact) {
  cbind(rows, artifact[rep.int(1L, nrow(rows)), , drop = FALSE])
}
