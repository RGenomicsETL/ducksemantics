# Reproduce the Mondo subset from the exact upstream release bytes.
# DisMech's file is distributed verbatim at the declared commit.
source("r/ducksemantics/R/graph.R")
input <- "build/sources/mondo-v2026-10-06.obo"
upstream_sha256 <- "2631e8e165d41def1a46749adb1ce0a336c8aba7f6bc0978729592cee8fed60b"
stopifnot(identical(digest::digest(file = input, algo = "sha256"), upstream_sha256))
lines <- readLines(input, warn = FALSE, encoding = "UTF-8")
stanzas <- ducksemantics_obo_term_stanzas(lines)
ids <- vapply(stanzas, ducksemantics_obo_first, "", tag = "id")
replacements <- vapply(stanzas, ducksemantics_obo_first, "", tag = "replaced_by")
obsolete <- which(startsWith(ids, "MONDO:") & !is.na(replacements) &
                    startsWith(replacements, "MONDO:"))[[1L]]
selected <- unique(c("MONDO:0007947", ids[[obsolete]], replacements[[obsolete]]))
keep <- stanzas[match(selected, ids)]
stopifnot(!anyNA(match(selected, ids)))
header <- lines[seq_len(which(lines == "[Term]")[[1L]] - 1L)]
# Keep only release-identifying header fields in the compact subset.
header <- header[grepl("^(format-version|data-version|ontology):", header)]
output <- "r/ducksemantics/inst/extdata/mondo-subset.obo"
writeLines(c(header, "", unlist(lapply(keep, function(x) c("[Term]", x, "")))),
           output, useBytes = TRUE)
release <- data.frame(
  provider_id = c("Mondo", "DisMech"),
  release_id = c("v2026-10-06", "fac61860115b1fdb194de4f4cacef3a8f6a398af"),
  available_at = c("2026-10-06T19:55:40Z", "2026-10-12T13:37:29Z"),
  receipt_id = c("mondo-v2026-10-06-subset", "dismech-fac61860-marfan"),
  source_id = c("mondo.obo", "kb/disorders/Marfan_Syndrome.yaml"),
  source_uri = c(
    "https://github.com/monarch-initiative/mondo/releases/download/v2026-10-06/mondo.obo",
    "https://raw.githubusercontent.com/monarch-initiative/dismech/fac61860115b1fdb194de4f4cacef3a8f6a398af/kb/disorders/Marfan_Syndrome.yaml"
  ),
  sha256 = c(digest::digest(file = output, algo = "sha256"),
    "9c098d1020e99b9c3ab6b9c28c2c4a884a0a122625e7f9385c5d6307e6924b29"),
  upstream_sha256 = c(upstream_sha256,
    "9c098d1020e99b9c3ab6b9c28c2c4a884a0a122625e7f9385c5d6307e6924b29"),
  selection = c(paste(selected, collapse = ";"), "verbatim record"),
  stringsAsFactors = FALSE
)
stopifnot(identical(digest::digest(file =
  "r/ducksemantics/inst/extdata/dismech-marfan.yaml", algo = "sha256"), release$sha256[[2L]]))
write.csv(release, "r/ducksemantics/inst/extdata/source-releases.csv", row.names = FALSE)
cat("Mondo subset:", paste(selected, collapse = ", "), "\n")
