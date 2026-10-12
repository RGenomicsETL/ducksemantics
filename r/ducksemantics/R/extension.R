#' Locate the bundled DuckDB extension
#'
#' The artifact uses DuckDB's stable C extension ABI, version 1.2.0. It is
#' built for the installation platform; no model weights are bundled.
#'
#' @return Absolute path to the installed extension.
#' @export
ducksemantics_extension_path <- function() {
  system.file("duckdb", "ducksemantics.duckdb_extension",
              package = "ducksemantics", mustWork = TRUE)
}

#' Load native semantic kernels into DuckDB
#'
#' The connection must permit unsigned extensions. [ducksemantics_connect()]
#' configures an isolated database handle and loads the bundled extension.
#'
#' @param conn A DuckDB DBI connection.
#' @param path Extension artifact path, defaulting to the installed artifact.
#' @return `conn`, invisibly.
#' @export
#' @examples
#' if (requireNamespace("duckdb", quietly = TRUE)) {
#'   conn <- ducksemantics_connect()
#'   DBI::dbGetQuery(conn,
#'     "SELECT semantic_maxsim([[1.0, 0.0]], [[1.0, 0.0]]) AS score")
#'   DBI::dbDisconnect(conn, shutdown = TRUE)
#' }
ducksemantics_load <- function(conn, path = ducksemantics_extension_path()) {
  ducksemantics_require_connection(conn)
  path <- normalizePath(ducksemantics_require_text(path, "path"), mustWork = TRUE)
  DBI::dbExecute(conn, paste("LOAD", DBI::dbQuoteString(conn, path)))
  invisible(conn)
}
