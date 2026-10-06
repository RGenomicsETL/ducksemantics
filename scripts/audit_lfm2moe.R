#!/usr/bin/env Rscript
# LFM2-MoE token sequences, SQL results, fork isolation and one-core timing.
# SPDX-License-Identifier: MIT

audit_lfm2moe <- function(path = "/root/bebelm/LFM2.5-8B-A1B-Q4_K_M.gguf",
                         extension = "build/ducksemantics.duckdb_extension", reuse_native = FALSE) {
    runner <- normalizePath("build/lfm2_oracle", mustWork = TRUE)
    outdir <- "build/lfm2moe-audit"
    promptdir <- file.path(outdir, "prompts")
    dir.create(promptdir, recursive = TRUE, showWarnings = FALSE)
    topics <- c(read.delim("scripts/colbert_cases.tsv", quote = "")$query,
        "clinical variant interpretation", "RNA splicing", "DNA methylation", "single cell sequencing",
        "polygenic risk scores", "genome assembly", "PCR amplification", "protein folding",
        "mitochondrial inheritance", "autosomal recessive inheritance", "long read sequencing",
        "genetic counseling", "variant allele frequency", "transcript isoforms", "ribosome function",
        "promoter activity", "linkage disequilibrium", "gene expression regulation", "immune cell signaling",
        "pharmacogenetics", "confidence intervals", "statistical power", "data provenance",
        "causal inference", "randomized trials", "sequence alignment", "structural variants",
        "copy number variation", "genome wide association studies", "reproducible research")
    stopifnot(length(topics) == 50L)
    prompts <- c(paste0("<|im_start|>user\nExplain ", topics,
        " in at least 100 words.<|im_end|>\n<|im_start|>assistant\n"), "Hello", "The first ten prime numbers are")
    for (i in seq_along(prompts)) {
        connection <- file(file.path(promptdir, paste0(i, ".txt")), "wb")
        writeChar(prompts[i], connection, eos = NULL, useBytes = TRUE)
        close(connection)
    }
    native_path <- file.path(outdir, "native.tsv")
    fork_path <- file.path(outdir, "fork.log")
    inputs <- list(model = normalizePath(path), model_info = file.info(path)[c("size", "mtime")],
                   runner_md5 = unname(tools::md5sum(runner)), extension_md5 = unname(tools::md5sum(extension)),
                   prompts = prompts, max_tokens = 32L)
    receipt_path <- file.path(outdir, "native-receipt.rds")
    if (reuse_native) {
        receipt <- readRDS(receipt_path)
        stopifnot(identical(inputs, receipt$inputs),
                  identical(tools::md5sum(c(native_path, fork_path)), receipt$outputs))
    } else {
        status <- system2(runner, c(shQuote(path), shQuote(normalizePath(promptdir)), length(prompts), 32L),
                          stdout = native_path, stderr = fork_path)
        if (status != 0L) stop(paste(readLines(fork_path), collapse = "\n"))
        saveRDS(list(inputs = inputs, outputs = tools::md5sum(c(native_path, fork_path))), receipt_path)
    }
    native <- read.delim(native_path, quote = "", na.strings = NULL,
                         colClasses = c(rep("numeric", 9L), "character", "character"))
    stopifnot(nrow(native) == length(prompts), identical(native$case, as.numeric(seq_along(prompts))))
    ids <- lapply(native$ids, function(x) if (nzchar(x)) as.integer(strsplit(x, ",", fixed = TRUE)[[1L]]) else integer())
    margins <- lapply(native$margins, function(x) if (nzchar(x)) as.numeric(strsplit(x, ",", fixed = TRUE)[[1L]]) else numeric())
    stopifnot(identical(lengths(ids), as.integer(native$tokens)), identical(lengths(margins), lengths(ids)),
              all(native$tokens[1:50] == 32), all(is.finite(unlist(margins))))
    fork_lines <- readLines(fork_path)
    stopifnot(identical(fork_lines, c(
        "fork 1: 12 shared + 13 appended; logits exact; 8 generated ids exact",
        "fork 2: 12 shared + 12 appended; logits exact; 8 generated ids exact")))
    cat(paste(fork_lines, collapse = "\n"), "\n")
    model <- Rbebelm::bebel_model_load(path, num_threads = 1L)
    drv <- duckdb::duckdb(config = list(allow_unsigned_extensions = "true", threads = "1"), shared_home = FALSE)
    con <- DBI::dbConnect(drv)
    on.exit(DBI::dbDisconnect(con, shutdown = TRUE))
    DBI::dbExecute(con, paste("LOAD", DBI::dbQuoteString(con, normalizePath(extension))))
    checks <- data.frame(case = seq_along(prompts), tokens = 0L, ids_equal = FALSE, sql_equal = FALSE,
                         first_divergence = NA_integer_, margin = NA_real_, sql_elapsed = 0)
    for (i in seq_along(prompts)) {
        expected <- Rbebelm::bebel_generate(model, prompts[i], greedy = TRUE, max_gen = 32L, check_interrupt = FALSE)
        common <- min(length(ids[[i]]), length(expected$ids))
        differing <- which(ids[[i]][seq_len(common)] != expected$ids[seq_len(common)])
        divergence <- if (length(differing) > 0L) differing[1L] else
            if (length(ids[[i]]) != length(expected$ids)) common + 1L else NA_integer_
        checks$tokens[i] <- length(expected$ids)
        checks$ids_equal[i] <- identical(ids[[i]], expected$ids)
        if (!is.na(divergence)) {
            margin <- if (divergence <= length(margins[[i]])) margins[[i]][divergence] else native$terminal_margin[i]
            checks$first_divergence[i] <- divergence; checks$margin[i] <- margin
            actual_id <- if (divergence <= length(ids[[i]])) ids[[i]][divergence] else native$terminal_id[i]
            wanted <- if (divergence <= length(expected$ids)) expected$ids[divergence] else "stop"
            cat(sprintf("FIRST DIVERGENCE prompt=%d step=%d C=%s R=%s C_top2_margin=%.9g\n",
                        i, divergence, actual_id, wanted, margin))
            write.table(checks, file.path(outdir, "verification.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
            stop("greedy token sequences differ")
        }
        checks$sql_elapsed[i] <- system.time(actual <- DBI::dbGetQuery(con,
            "SELECT semantic_generate(?, ?, 32) AS text", params = list(path, prompts[i]))$text[[1L]])[["elapsed"]]
        checks$sql_equal[i] <- identical(actual, expected$text)
        cat(sprintf("prompt=%d tokens=%d ids_equal=%s sql_equal=%s\n", i, checks$tokens[i],
                    checks$ids_equal[i], checks$sql_equal[i]))
        write.table(checks[seq_len(i), ], file.path(outdir, "progress.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
        if (!checks$sql_equal[i]) stop(sprintf("SQL generation differs for prompt %d", i))
    }
    # Requested ids retain their order and duplicates; empty requests are typed empty lists.
    options <- c(0L, 16L, 124900L, 124901L, 127999L)
    one <- DBI::dbGetQuery(con, "SELECT semantic_next_token_logits(?, ?, ?::INTEGER[]) AS v",
                           params = list(path, prompts[1L], list(options)))$v[[1L]]
    reversed <- DBI::dbGetQuery(con, "SELECT semantic_next_token_logits(?, ?, ?::INTEGER[]) AS v",
                                params = list(path, prompts[1L], list(rev(options))))$v[[1L]]
    stopifnot(identical(one, rev(reversed)), length(one) == 5L, all(is.finite(one)))
    duplicate <- DBI::dbGetQuery(con, "SELECT semantic_next_token_logits(?, ?, [16,16]) AS v",
                                params = list(path, prompts[1L]))$v[[1L]]
    empty <- DBI::dbGetQuery(con, "SELECT semantic_next_token_logits(?, ?, []::INTEGER[]) AS v",
                            params = list(path, prompts[1L]))$v[[1L]]
    stopifnot(identical(duplicate, rep(one[2L], 2L)), length(empty) == 0L)
    write.table(checks, file.path(outdir, "verification.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
    summary <- c(sprintf("R=%s duckdb=%s Rbebelm=%s", getRversion(), packageVersion("duckdb"), packageVersion("Rbebelm")),
        sprintf("prompts=%d required_32_token_prompts=%d required_generated=%d total_generated=%d",
                nrow(checks), sum(checks$tokens[1:50] == 32L), sum(checks$tokens[1:50]), sum(checks$tokens)),
        sprintf("ids_equal=%d/%d SQL_text_equal=%d/%d fork_branches=2 exact", sum(checks$ids_equal), nrow(checks), sum(checks$sql_equal), nrow(checks)),
        sprintf("native prefill tokens/s=%.9g decode tokens/s=%.9g max_RSS_KiB=%.9g",
                sum(native$prompt_tokens) / sum(native$prefill_seconds), sum(native$tokens) / sum(native$decode_seconds), max(native$rss_kib)),
        sprintf("SQL wall tokens/s including prefill/load=%.9g", sum(checks$tokens) / sum(checks$sql_elapsed)),
        sprintf("first_divergence=none minimum_C_top2_margin=%.9g", min(unlist(margins))))
    writeLines(summary, file.path(outdir, "summary.txt"))
    cat(paste(summary, collapse = "\n"), "\n")
    stopifnot(all(checks$tokens[1:50] == 32L), all(checks$ids_equal), all(checks$sql_equal))
}
if (sys.nframe() == 0L) {
    args <- commandArgs(trailingOnly = TRUE)
    reuse_native <- "--reuse-native" %in% args
    args <- args[args != "--reuse-native"]
    path <- if (length(args) > 0L) args[[1L]] else "/root/bebelm/LFM2.5-8B-A1B-Q4_K_M.gguf"
    extension <- if (length(args) > 1L) args[[2L]] else "build/ducksemantics.duckdb_extension"
    audit_lfm2moe(path, extension, reuse_native)
}
