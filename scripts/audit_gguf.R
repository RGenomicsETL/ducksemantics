#!/usr/bin/env Rscript
# Differential audit: DuckSemantics GGUF views against Rgguf, which parses with
# ggml's official GGUF implementation (Rfmalloc monorepo).
#
# Usage: Rscript scripts/audit_gguf.R EXTENSION FILE.gguf [...]
#
# Compares every metadata key and value, every array element, and every
# tensor's name, type, dims, element count, byte size and absolute offset.
# Rgguf returns R values rather than GGUF type names, so float32 values are
# compared at float32 precision and everything else exactly. Stops at the
# first mismatch.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) stop("usage: audit_gguf.R EXTENSION FILE.gguf [...]")
extension <- normalizePath(args[[1L]], mustWork = TRUE)

con <- DBI::dbConnect(duckdb::duckdb(config = list(allow_unsigned_extensions = "true")))
on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)
DBI::dbExecute(con, paste0("LOAD ", DBI::dbQuoteString(con, extension)))
q <- function(sql, ...) DBI::dbGetQuery(con, sql, params = list(...))

check <- function(label, ours, theirs) {
  if (!identical(ours, theirs)) {
    stop(sprintf("MISMATCH %s\n  ours:      %s\n  reference: %s", label,
                 paste(utils::head(format(ours), 5L), collapse = " "),
                 paste(utils::head(format(theirs), 5L), collapse = " ")), call. = FALSE)
  }
}

# Our values are text; parse them to the reference's R type before comparing.
same_value <- function(text, type, ref) {
  if (type %in% c("string")) return(identical(text, as.character(ref)))
  if (type == "bool") return(identical(text == "true", as.logical(ref)))
  value <- as.numeric(text)
  if (type == "float32") return(isTRUE(all(abs(value - ref) <= abs(ref) * 2^-23)))
  identical(value, as.numeric(ref))
}

for (path in args[-1L]) {
  path <- normalizePath(path, mustWork = TRUE)
  ref_meta <- Rgguf::gguf_metadata(path)
  ref_tensors <- Rgguf::gguf_tensors(path)

  header <- q("SELECT tensor_count, metadata_count FROM gguf_header(?)", path)
  check(paste(path, "tensor_count"), as.numeric(header$tensor_count), as.numeric(nrow(ref_tensors)))
  check(paste(path, "metadata_count"), as.numeric(header$metadata_count), as.numeric(length(ref_meta)))

  meta <- q("SELECT key, value_type, array_type, array_length, value FROM gguf_metadata(?)", path)
  check(paste(path, "metadata keys"), meta$key, names(ref_meta))
  elements <- 0
  for (i in seq_len(nrow(meta))) {
    key <- meta$key[[i]]
    ref <- ref_meta[[key]]
    if (meta$value_type[[i]] == "array") {
      check(paste(path, key, "array_length"), as.numeric(meta$array_length[[i]]), as.numeric(length(ref)))
      ours <- q("SELECT value FROM gguf_metadata_array(?, ?) ORDER BY array_index", path, key)$value
      if (meta$array_type[[i]] != "array" &&
          !(length(ours) == length(ref) &&
            (length(ref) == 0L || same_value(ours, meta$array_type[[i]], ref)))) {
        stop(sprintf("MISMATCH %s %s array elements", path, key), call. = FALSE)
      }
      elements <- elements + length(ours)
    } else if (!same_value(meta$value[[i]], meta$value_type[[i]], ref)) {
      stop(sprintf("MISMATCH %s %s: ours %s, reference %s", path, key,
                   meta$value[[i]], format(ref)), call. = FALSE)
    }
  }

  tensors <- q("SELECT name, ggml_type, dims, n_elements, file_offset, n_bytes
                FROM gguf_tensors(?) ORDER BY tensor_index", path)
  check(paste(path, "tensor names"), tensors$name, ref_tensors$name)
  check(paste(path, "tensor types"), tolower(tensors$ggml_type), tolower(ref_tensors$type))
  # We report dims exactly as stored; ggml drops trailing 1s (ggml_n_dims).
  ggml_dims <- function(d) {
    d <- as.numeric(d)
    while (length(d) > 1L && d[[length(d)]] == 1) d <- d[-length(d)]
    d
  }
  check(paste(path, "tensor dims"), lapply(tensors$dims, ggml_dims), lapply(ref_tensors$dims, ggml_dims))
  check(paste(path, "tensor n_elements"), as.numeric(tensors$n_elements), as.numeric(ref_tensors$n_elements))
  check(paste(path, "tensor n_bytes"), as.numeric(tensors$n_bytes), as.numeric(ref_tensors$nbytes))
  check(paste(path, "tensor offsets"), as.numeric(tensors$file_offset), as.numeric(ref_tensors$offset))
  cat(sprintf("ok %s: %d keys, %.0f array elements, %d tensors\n",
              path, nrow(meta), elements, nrow(tensors)))
}
