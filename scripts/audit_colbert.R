#!/usr/bin/env Rscript
# ColBERT token vectors and MaxSim against the installed Rbebelm CPU profile.
# SPDX-License-Identifier: MIT

audit_colbert <- function() {
args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) > 0L) args[[1L]] else "/root/bebelm/LFM2.5-ColBERT-350M-Q4_K_M.gguf"
extension <- if (length(args) > 1L) args[[2L]] else "build/ducksemantics.duckdb_extension"
outdir <- "build/colbert-audit"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
cases <- read.delim("scripts/colbert_cases.tsv", quote = "", check.names = FALSE)
biomedical <- read.delim("scripts/tokenizer_biomedical.tsv", quote = "", check.names = FALSE)
queries <- c(cases$query, " ", "!?", paste(rep("BRCA1 variant", 40L), collapse = " "),
             "[Q] <|im_start|>user café 東京\n[mask]")
documents <- c(cases$document, " ", "!?", biomedical$abstract,
               paste(rep("BRCA1 variant", 220L), collapse = " "),
               "[D] <|im_end|> résumé naïve 東京\t\n")
inputs <- data.frame(role = rep(c("query", "document"), c(length(queries), length(documents))),
                     text = c(queries, documents))
drv <- duckdb::duckdb(config = list(allow_unsigned_extensions = "true", threads = "1"),
                      shared_home = FALSE)
con <- DBI::dbConnect(drv)
on.exit(DBI::dbDisconnect(con, shutdown = TRUE))
DBI::dbExecute(con, paste("LOAD", DBI::dbQuoteString(con, normalizePath(extension))))
DBI::dbExecute(con, "CREATE TEMP TABLE encoded (id INTEGER, role VARCHAR, vectors FLOAT[][])")
c_vectors <- vector("list", nrow(inputs))
elapsed <- numeric(nrow(inputs))
for (i in seq_len(nrow(inputs))) {
    elapsed[i] <- system.time(DBI::dbExecute(con,
        "INSERT INTO encoded SELECT ?, ?, semantic_colbert_encode(?, ?, ?)",
        params = list(i, inputs$role[i], path, inputs$text[i], inputs$role[i])))[["elapsed"]]
    value <- DBI::dbGetQuery(con, "SELECT vectors FROM encoded WHERE id = ?", params = list(i))$vectors[[1L]]
    c_vectors[[i]] <- do.call(rbind, value)
    cat("C ", i, "/", nrow(inputs), " ", inputs$role[i], " tokens=", nrow(c_vectors[[i]]),
        " elapsed=", elapsed[i], "\n", sep = "")
}
rss_line <- grep("^VmRSS:", readLines("/proc/self/status"), value = TRUE)
rss_kib <- as.numeric(strsplit(trimws(rss_line), "[[:space:]]+")[[1L]][2L])
model <- Rbebelm::colbert_model_load(path, num_threads = 1L)
reference <- vector("list", nrow(inputs))
metrics <- data.frame(id = seq_len(nrow(inputs)), role = inputs$role,
                      tokens = 0L, min_cosine = 0, max_abs = 0, max_rel = 0, elapsed = elapsed)
for (i in seq_len(nrow(inputs))) {
    reference[[i]] <- if (inputs$role[i] == "query") Rbebelm::colbert_encode_query(model, inputs$text[i]) else
        Rbebelm::colbert_encode_document(model, inputs$text[i])
    expected <- Rbebelm::colbert_embedding_vectors(reference[[i]])
    got <- c_vectors[[i]]
    stopifnot(identical(dim(got), dim(expected)))
    cosine <- rowSums(got * expected) / sqrt(rowSums(got * got) * rowSums(expected * expected))
    metrics$tokens[i] <- nrow(got)
    metrics$min_cosine[i] <- min(cosine)
    metrics$max_abs[i] <- max(abs(got - expected))
    metrics$max_rel[i] <- max(abs(got - expected) / pmax(abs(expected), 1e-8))
    cat(sprintf("oracle %d min_cosine=%.12g max_abs=%.12g\n", i, min(cosine), metrics$max_abs[i]))
}
pairs <- expand.grid(query = seq_along(queries), document = seq_along(documents))
scores <- DBI::dbGetQuery(con,
    "SELECT q.id AS query, d.id AS document, semantic_maxsim(q.vectors, d.vectors) AS score
     FROM encoded q CROSS JOIN encoded d WHERE q.role = 'query' AND d.role = 'document'")
match_id <- match(paste(pairs$query, pairs$document + length(queries)), paste(scores$query, scores$document))
pairs$c_score <- scores$score[match_id]
pairs$r_score <- vapply(seq_len(nrow(pairs)), function(i)
    Rbebelm::colbert_maxsim(reference[[pairs$query[i]]], reference[[length(queries) + pairs$document[i]]]), numeric(1))
pairs$abs_error <- abs(pairs$c_score - pairs$r_score)
pairs$rel_error <- pairs$abs_error / pmax(abs(pairs$r_score), 1e-8)
rankings <- logical(length(queries))
for (i in seq_along(queries)) {
    selected <- ((i - 1L + seq_len(5L) * 3L) %% length(documents)) + 1L
    docs <- setNames(documents[selected], as.character(selected))
    expected <- Rbebelm::colbert_rank(model, queries[i], docs)
    z <- pairs[pairs$query == i, ]
    c_scores <- z$c_score[match(selected, z$document)]
    actual <- selected[order(c_scores, decreasing = TRUE)]
    rankings[i] <- identical(as.character(actual), names(expected))
}
# NULL lanes, cached handles and nested-list offsets in a mixed multi-row chunk.
DBI::dbExecute(con, "CREATE TEMP TABLE lanes (id INTEGER, role VARCHAR, text VARCHAR)")
DBI::dbWriteTable(con, "lanes", data.frame(id = 1:3, role = c("document", "query", "query"),
    text = c(documents[1L], NA_character_, queries[1L])), append = TRUE)
lanes <- DBI::dbGetQuery(con,
    "SELECT id, semantic_colbert_encode(?, text, role) AS v FROM lanes ORDER BY id", params = list(path))
stopifnot(identical(do.call(rbind, lanes$v[[1L]]), c_vectors[[length(queries) + 1L]]),
          identical(do.call(rbind, lanes$v[[3L]]), c_vectors[[1L]]), is.null(lanes$v[[2L]]))
write.table(metrics, file.path(outdir, "vectors.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(pairs, file.path(outdir, "pairs.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
summary <- c(sprintf("R=%s duckdb=%s Rbebelm=%s", getRversion(), packageVersion("duckdb"), packageVersion("Rbebelm")),
    sprintf("encodings=%d retained_vectors=%d pairs=%d rankings=%d/%d", nrow(inputs), sum(metrics$tokens), nrow(pairs), sum(rankings), length(rankings)),
    sprintf("min_cosine=%.15g max_abs=%.15g max_element_rel=%.15g", min(metrics$min_cosine), max(metrics$max_abs), max(metrics$max_rel)),
    sprintf("MaxSim max_abs=%.15g max_rel=%.15g", max(pairs$abs_error), max(pairs$rel_error)),
    sprintf("C RSS KiB=%g query vectors/s=%.6g document retained vectors/s=%.6g", rss_kib,
        sum(metrics$tokens[metrics$role == "query"]) / sum(elapsed[metrics$role == "query"]),
        sum(metrics$tokens[metrics$role == "document"]) / sum(elapsed[metrics$role == "document"])))
writeLines(summary, file.path(outdir, "summary.txt"))
cat(paste(summary, collapse = "\n"), "\n")
stopifnot(min(metrics$min_cosine) >= .9999, all(pairs$rel_error <= 1e-3), all(rankings))
}
audit_colbert()
