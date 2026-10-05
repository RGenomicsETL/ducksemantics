#!/usr/bin/env Rscript
# SQL quant audit against GGML, with Rgguf numeric tensors where bounded in size.
# SPDX-License-Identifier: MIT

models <- c(
    "/root/bebelm/LFM2.5-8B-A1B-Q4_K_M.gguf",
    "/root/bebelm/LFM2.5-ColBERT-350M-Q4_K_M.gguf",
    "/root/bebelm/embeddinggemma-300M-Q8_0.gguf",
    "/root/modular/max/tests/integration/architectures/llama3/testdata/tiny_llama.gguf",
    "/root/modular/max/tests/integration/architectures/llama3/testdata/tiny_llama_bf16.gguf",
    "build/fixtures/quant.gguf", "build/fixtures/unaligned.gguf", "build/fixtures/rank3.gguf"
)

audit_quant <- function(paths, extension) {
    if (system2("make", "quant-oracle") != 0L) stop("oracle build failed")
    dll <- dyn.load(normalizePath("build/oracle/ggml.so"))
    on.exit(dyn.unload(dll[["path"]]), add = TRUE)
    self <- .C("oracle_selftest", failures = integer(1), widen = integer(1),
               narrow = integer(1), activations = integer(1))
    stopifnot(self$failures == 0L)
    cat("Conversions: ", self$widen, " widening and ", self$narrow,
        " narrowing patterns per type, all bits exact. Activations: ",
        self$activations, " vectors / ", self$activations * 1024,
        " values per type, packed bytes exact.\n", sep = "")

    drv <- duckdb::duckdb(config = list(allow_unsigned_extensions = "true", threads = "1"),
                         shared_home = FALSE)
    con <- DBI::dbConnect(drv)
    on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)
    DBI::dbExecute(con, paste("LOAD", DBI::dbQuoteString(con, normalizePath(extension))))
    scratch <- tempfile("ggml-audit-", tmpdir = "build/oracle", fileext = ".bin")
    on.exit(unlink(scratch), add = TRUE)
    runtime <- Rfmalloc::open_fmalloc(scratch, size_gb = 0.05, mode = "scratch")
    types <- c("F32", "F16", "BF16", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K")
    fields <- c("tensors", "rows", "values", "matvec_tensors", "matvec_rows",
                "max_relative_error", "max_scaled_error", "max_bound_fraction")
    stats <- matrix(0, length(types), length(fields), dimnames = list(types, fields))
    reference_counts <- c(Rgguf = 0L, C_ggml = 0L)
    set.seed(20260710)
    eps <- 2^-23
    for (path in paths) {
        directory <- Rgguf::gguf_tensors(path)
        for (i in seq_len(nrow(directory))) {
            name <- directory$name[i]
            type <- toupper(directory$type[i])
            stopifnot(type %in% types)
            columns <- directory$dims[[i]][1L]
            rows <- directory$n_elements[i] / columns
            selected <- if (rows <= 16) seq_len(rows) - 1 else
                sort(c(0, rows - 1, sample.int(rows - 2, 6)))
            small <- directory$n_elements[i] <= 1e6 && type != "Q8_K"
            if (small) {
                reference <- Rgguf::gguf_tensor(path, name, runtime = runtime, as = "numeric")
                reference_counts["Rgguf"] <- reference_counts["Rgguf"] + 1L
            } else {
                reference_counts["C_ggml"] <- reference_counts["C_ggml"] + 1L
            }
            weights <- matrix(NA_real_, columns, length(selected))
            for (j in seq_along(selected)) {
                first <- selected[j] * columns
                expected <- if (small) as.numeric(reference[first + seq_len(columns)]) else {
                    decoded <- .C("oracle_values", path, name, as.double(first),
                                  as.integer(columns), out = double(columns), status = integer(1))
                    stopifnot(decoded$status == 0L)
                    decoded$out
                }
                got <- DBI::dbGetQuery(con,
                    "SELECT gguf_tensor_values(?, ?, ?::UBIGINT, ?::UBIGINT) AS v",
                    params = list(path, name, first, columns))$v[[1L]]
                if (!identical(writeBin(got, raw(), size = 4L), writeBin(expected, raw(), size = 4L))) {
                    stop("dequant bits differ: ", path, " / ", name, " / row ", selected[j])
                }
                weights[, j] <- expected
            }
            stats[type, c("tensors", "rows", "values")] <-
                stats[type, c("tensors", "rows", "values")] + c(1, length(selected), length(weights))
            if (directory$n_dims[i] >= 2L) {
                x <- readBin(writeBin(rnorm(columns), raw(), size = 4L), numeric(),
                             n = columns, size = 4L)
                input <- data.frame(path = path, name = name)
                input$x <- list(x)
                duckdb::duckdb_register(con, "matvec_input", input)
                result <- DBI::dbGetQuery(con,
                    "SELECT gguf_tensor_matvec(path, name, x::FLOAT[]) AS y FROM matvec_input")$y[[1L]]
                duckdb::duckdb_unregister(con, "matvec_input")
                stopifnot(length(result) == rows)
                target <- as.vector(crossprod(weights, x))
                if (type %in% c("Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K")) {
                    activation <- .C("oracle_activation", if (type == "Q8_0") 8L else 15L,
                                     x, as.integer(columns), out = double(columns), status = integer(1))
                    stopifnot(activation$status == 0L)
                    xq <- activation$out
                } else {
                    xq <- x
                }
                absw <- abs(weights)
                scale <- as.vector(crossprod(absw, abs(x)))
                gamma <- (columns + 8) * eps / (1 - (columns + 8) * eps)
                # Triangle bound on activation error, plus fixed-order f32 rounding
                # and regrouping scales relative to GGML's rounded to_float values.
                bound <- as.vector(crossprod(absw, abs(xq - x))) +
                    gamma * as.vector(crossprod(absw, abs(xq))) + 8 * eps * scale
                error <- abs(result[selected + 1] - target)
                if (any(error > bound + 1e-30)) stop("matvec bound failed: ", path, " / ", name)
                stats[type, c("matvec_tensors", "matvec_rows")] <-
                    stats[type, c("matvec_tensors", "matvec_rows")] + c(1, length(selected))
                maxima <- c(max(error / pmax(abs(target), 1e-30)),
                            max(error / pmax(scale, 1e-30)), max(error / pmax(bound, 1e-30)))
                stats[type, c("max_relative_error", "max_scaled_error", "max_bound_fraction")] <-
                    pmax(stats[type, c("max_relative_error", "max_scaled_error", "max_bound_fraction")], maxima)
            }
            if (small) rm(reference)
            gc(verbose = FALSE)
        }
        cat(path, ": ", nrow(directory), " tensors\n", sep = "")
    }
    report <- data.frame(type = rownames(stats), stats, row.names = NULL)
    write.table(report, "build/oracle/audit.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
    print(report, digits = 8, row.names = FALSE)
    print(reference_counts)
    cat("Rgguf ", as.character(utils::packageVersion("Rgguf")), "; DuckDB R ",
        as.character(utils::packageVersion("duckdb")), "\n", sep = "")
    if (!any(grepl("matvec|vec_dot|dequant", getNamespaceExports("Rbebelm")))) {
        cat("Rbebelm exposes no raw matvec/dot/dequant oracle; direct kernel comparison unavailable.\n")
    }
    invisible(report)
}

args <- commandArgs(trailingOnly = TRUE)
audit_quant(if (length(args) > 0L) args else models,
            Sys.getenv("QUANT_EXTENSION", "build/ducksemantics.duckdb_extension"))
