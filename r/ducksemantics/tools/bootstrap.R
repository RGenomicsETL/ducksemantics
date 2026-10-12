#!/usr/bin/env Rscript
# Stage the canonical native sources and offline fixtures for a source package.
args <- commandArgs(trailingOnly = TRUE)
check <- "--check" %in% args
root <- normalizePath(".", mustWork = TRUE)
if (!file.exists(file.path(root, "description.yml"))) {
  stop("Run from the DuckSemantics repository root.", call. = FALSE)
}
package <- file.path(root, "r", "ducksemantics")
bundle <- file.path(package, "inst", "extension")
paths <- c("Makefile", "description.yml", "LICENSE", "src", "duckdb_capi",
           "scripts/append_metadata.sh", "test")
files <- unlist(lapply(paths, function(path) {
  if (dir.exists(path)) list.files(path, recursive = TRUE, full.names = TRUE)
  else path
}), use.names = FALSE)
if (check) {
  staged <- file.path(bundle, files)
  if (any(!file.exists(staged)) ||
      !identical(unname(tools::md5sum(files)), unname(tools::md5sum(staged)))) {
    stop("Packaged native sources differ; run make r-bootstrap.", call. = FALSE)
  }
  extra <- setdiff(list.files(bundle, recursive = TRUE), files)
  if (length(extra) > 0L) stop("Unexpected bundled files: ", paste(extra, collapse = ", "))
  cat("Packaged native sources match.\n")
} else {
  unlink(bundle, recursive = TRUE)
  for (file in files) {
    destination <- file.path(bundle, file)
    dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
    if (!file.copy(file, destination, overwrite = TRUE)) stop("Could not stage ", file)
  }
  status <- system2(Sys.getenv("MAKE", "make"), "fixtures")
  if (status != 0L) stop("Could not generate offline GGUF fixtures.", call. = FALSE)
  fixtures <- c("valid.gguf", "bad_magic.gguf", "tokenizer_gemma.gguf", "tokenizer_lfm.gguf")
  if (!all(file.copy(file.path("build", "fixtures", fixtures),
                     file.path(package, "inst", "extdata"), overwrite = TRUE))) {
    stop("Could not stage offline GGUF fixtures.", call. = FALSE)
  }
  cat("Staged native sources and offline GGUF fixtures.\n")
}
