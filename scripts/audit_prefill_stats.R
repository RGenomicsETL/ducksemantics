#!/usr/bin/env Rscript
# Exercise the real-model generation STRUCT and record its timing boundaries.
args <- commandArgs(trailingOnly = TRUE)
extension <- if (length(args) > 0L) args[[1L]] else "build/ducksemantics.duckdb_extension"
model <- if (length(args) >= 2L) args[[2L]] else Sys.getenv("DUCKSEMANTICS_GENERATE_MODEL")
if (!nzchar(model)) {
  stop("Set DUCKSEMANTICS_GENERATE_MODEL to a local LFM2.5-8B-A1B GGUF file.", call. = FALSE)
}
extension <- normalizePath(extension, mustWork = TRUE)
model <- normalizePath(model, mustWork = TRUE)

local({
  conn <- DBI::dbConnect(duckdb::duckdb(config = list(allow_unsigned_extensions = "true"),
                                       shared_home = FALSE))
  on.exit(DBI::dbDisconnect(conn, shutdown = TRUE))
  DBI::dbExecute(conn, paste("LOAD", DBI::dbQuoteString(conn, extension)))
  DBI::dbExecute(conn, "SET threads = 1")
  path <- as.character(DBI::dbQuoteString(conn, model))
  prompt <- as.character(DBI::dbQuoteString(conn,
    "<|im_start|>user\nName one DNA repair gene.<|im_end|>\n<|im_start|>assistant\n"))
  stats <- DBI::dbGetQuery(conn, paste0(
    "WITH runs AS MATERIALIZED (SELECT batch_size, semantic_generate_stats(", path,
    ", ", prompt, ", 4, batch_size) AS result FROM (VALUES (1), (16), (32)) AS batches(batch_size)) ",
    "SELECT batch_size, result.* FROM runs"))
  count <- as.numeric(stats$generated_tokens)
  decoded <- as.numeric(stats$decode_tokens)
  stopifnot(all(stats$text == stats$text[[1L]]),
            all(decoded == ifelse(stats$stopped, count, pmax(0, count - 1))),
            all(as.numeric(stats$prefill_batch_size) == stats$batch_size),
            all(as.numeric(stats$prefill_batches) == ceiling(as.numeric(stats$prompt_tokens) / stats$batch_size)),
            all(stats$generation_seconds >= stats$prefill_seconds + stats$decode_seconds),
            all(stats$total_seconds >= stats$generation_seconds + stats$model_load_seconds))
  default <- DBI::dbGetQuery(conn, paste0(
    "WITH run AS MATERIALIZED (SELECT semantic_generate_stats(", path, ", ", prompt,
    ", 4) AS result) SELECT result.text AS text, result.prefill_batch_size AS batch_size FROM run"))
  stopifnot(identical(default$text, stats$text[1L]), as.numeric(default$batch_size) == 16)
  one <- DBI::dbGetQuery(conn, paste0(
    "WITH run AS MATERIALIZED (SELECT semantic_generate_stats(", path, ", ", prompt,
    ", 1) AS result) SELECT result.decode_tokens AS decoded, ",
    "result.decode_tokens_per_second IS NULL AS rate_missing FROM run"))
  stopifnot(as.numeric(one$decoded) == 0, one$rate_missing)
  mixed <- DBI::dbGetQuery(conn, paste0(
    "WITH runs AS MATERIALIZED (SELECT ordinal, semantic_generate_stats(", path,
    ", prompt, budget, 16) AS result FROM (VALUES (1, ", prompt,
    ", 1), (2, NULL, 2), (3, ", prompt, ", 2)) AS inputs(ordinal, prompt, budget)) ",
    "SELECT ordinal, result IS NULL AS missing, result.text IS NULL AS text_missing, ",
    "result.decode_tokens_per_second IS NULL AS rate_missing FROM runs ORDER BY ordinal"))
  stopifnot(identical(mixed$missing, c(FALSE, TRUE, FALSE)),
            identical(mixed$text_missing, c(FALSE, TRUE, FALSE)),
            mixed$rate_missing[[1L]], mixed$rate_missing[[2L]])
  dir.create("build/prefill-audit", recursive = TRUE, showWarnings = FALSE)
  saveRDS(list(extension = extension, model = model, model_bytes = file.info(model)$size,
               stats = stats, default = default, one = one, mixed = mixed),
          "build/prefill-audit/sql-stats.rds")
  write.csv(stats, "build/prefill-audit/sql-stats.csv", row.names = FALSE)
  print(stats[, c("batch_size", "prompt_tokens", "generated_tokens", "decode_tokens",
                 "prefill_seconds", "decode_seconds", "generation_seconds", "model_load_seconds", "total_seconds")])
  message("Generation timing, rate, mixed-NULL and single-budget contracts passed.")
})
