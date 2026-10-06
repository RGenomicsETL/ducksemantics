#!/usr/bin/env Rscript
# Numeric and retrieval oracle for the EmbeddingGemma SQL encoder.
args <- commandArgs(trailingOnly = TRUE)
benchmark_only <- "--benchmark" %in% args
summary_only <- "--summary" %in% args
args <- args[!args %in% c("--benchmark", "--summary")]
tasks <- c("retrieval_query", "retrieval_document", "question_answering", "fact_verification",
  "classification", "clustering", "semantic_similarity", "code_retrieval", "summarization", "raw")
dimensions <- c(768L, 512L, 256L, 128L)
extension <- normalizePath(if (length(args) > 0L) args[[1L]] else "build/ducksemantics.duckdb_extension")
path <- normalizePath(if (length(args) >= 2L) args[[2L]] else "/root/bebelm/embeddinggemma-300M-Q8_0.gguf")
out_dir <- if (length(args) >= 3L) args[[3L]] else "build/embeddinggemma-audit"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (summary_only) {
  files <- list.files(out_dir, "^metrics[.]tsv$", recursive = TRUE, full.names = TRUE)
  metrics <- do.call(rbind, lapply(files, read.delim))
  folders <- dirname(files)
  corpus_hashes <- unname(tools::md5sum(file.path(folders, "corpus.tsv")))
  details <- unlist(lapply(file.path(folders, "worst.rds"), readRDS), recursive = FALSE)
  stopifnot(!anyNA(corpus_hashes), length(unique(corpus_hashes)) == 1L,
    length(details) == 80L, all(vapply(details, function(x)
      length(x$tokens) == 200L && all(x$tokens <= 2048L) &&
      length(x$truncated) == 200L && all(tail(x$truncated, 2L)), TRUE)))
  expected <- expand.grid(task = tasks, dimensions = dimensions, normalize = c(FALSE, TRUE))
  key <- function(x) paste(x$task, x$dimensions, x$normalize, sep = "/")
  stopifnot(nrow(metrics) == nrow(expected), setequal(key(metrics), key(expected)),
    !anyDuplicated(key(metrics)), all(metrics$vectors == 200L),
    all(metrics$values == metrics$vectors * metrics$dimensions), all(metrics$max_abs <= 1e-4),
    all(metrics$min_cosine >= 0.99999), all(metrics$rank_mismatches == 0L))
  wrapper_files <- list.files(out_dir, "^wrappers[.]tsv$", recursive = TRUE, full.names = TRUE)
  wrappers <- do.call(rbind, lapply(wrapper_files, read.delim))
  expected_wrappers <- expected[expected$task %in% c("retrieval_query", "retrieval_document"), ]
  stopifnot(nrow(wrappers) == nrow(expected_wrappers),
    setequal(key(wrappers), key(expected_wrappers)), !anyDuplicated(key(wrappers)),
    all(wrappers$max_abs <= 1e-4), all(wrappers$vectors == 200L))
  write.table(metrics, file.path(out_dir, "metrics-all.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(wrappers, file.path(out_dir, "wrappers-all.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  cat(sprintf("PASS: 200 texts x 10 tasks x 4 dimensions x 2 normalization settings; %d vectors, %d values\n",
    sum(metrics$vectors), sum(metrics$values)))
  cat(sprintf("worst abs=%.12g rel=%.12g minimum cosine=%.15g; top-10 mismatches=%d; wrapper vectors=%d\n",
    max(metrics$max_abs), max(metrics$max_rel), min(metrics$min_cosine),
    sum(metrics$rank_mismatches), sum(wrappers$vectors)))
  quit(save = "no")
}
suppressPackageStartupMessages({ library(DBI); library(duckdb); library(Rbebelm) })
con <- dbConnect(duckdb(config = list(allow_unsigned_extensions = "true", threads = "1")))
dbExecute(con, sprintf("LOAD %s", dbQuoteString(con, extension)))
model <- embeddinggemma_model_load(path, num_threads = if (benchmark_only) 1L else
  as.integer(Sys.getenv("RBEBELM_AUDIT_THREADS", "1")))

ascii <- c("", " ", "\n\t", "Hello world", "The quick brown fox jumps over the lazy dog.",
  "a", "ABC xyz 0123456789", "punctuation: !?;,:()[]{}", "Don't split contractions.",
  "a\nb\nc", "line one\r\nline two", "\tindented\tcolumns", "two  spaces   here",
  "search result about population genetics", "question: What is a coding exon?",
  "DNA RNA protein", "Numbers 1e-5 -0.001 3.141592653589793", "mailto:test@example.org",
  "https://pubmed.ncbi.nlm.nih.gov/12345678/", "foo_bar <- function(x) x + 1",
  "SELECT gene, count(*) FROM variants GROUP BY gene;", "if (x == NULL) return false;",
  "Chromosome 1:12345 A>G", "Insertion deletion substitution", "zero-shot classification",
  "relatedness and identity by descent", "The cat sat on the mat.", "Dogs like running outdoors.",
  "A patient was referred for genetic counselling.", "Protein folding and molecular interactions.",
  "Five controls and seven cases were sequenced.", "p-value < 0.05; 95% CI 0.8-1.2",
  "Summary: no statistically significant association.", "title: test | text: content",
  "task: search result | query: already formatted", "BOS EOS <bos> <eos> <pad>",
  "C:\\data\\file.txt /tmp/a-b.c", "AAAAACCCCCGGGGGTTTTT", "A 1 B 2 C 3", "End.")
unicode <- c("café naïve résumé", "cafe\u0301 nai\u0308ve", "🧬 DNA 🧪 RNA 🔬 protein",
  "中文：基因表达与遗传变异", "日本語の遺伝子解析", "한국어 유전자 연구", "Αλφα βήτα γάμμα",
  "Наследственные варианты генома", "دراسة الجينات البشرية", "हिन्दी जीन अनुक्रम",
  "שלום מחקר גנטי", "élève über Straße", "Māori whakapapa", "ไทยพันธุกรรม",
  "β-thalassaemia; α-globin", "ΔF508 CFTR p.Phe508del", "−1 × 10⁻⁶ ± 0.01",
  "‘single’ “double” — en–dash…", "non\u00a0breaking\u2009thin space", "A\u200dB joiner",
  "🧑🏽‍⚕️ patient 🧑🏻‍🔬 researcher", "🇫🇷 🇲🇱 🇺🇸", "𝛼 𝛽 ℝ ∑ ∞", "ééé 中文🧬 العربية")
variants <- c("NM_007294.4(BRCA1):c.5266dup (p.Gln1756ProfsTer74)",
  "NM_000059.4(BRCA2):c.5946del (p.Ser1982ArgfsTer22)",
  "NM_000492.4(CFTR):c.1521_1523del (p.Phe508del)",
  "NM_000546.6(TP53):c.743G>A (p.Arg248Gln)",
  "NM_000518.5(HBB):c.20A>T (p.Glu7Val)",
  "NM_000314.8(PTEN):c.388C>T (p.Arg130Ter)",
  "NM_000249.4(MLH1):c.1852_1854del (p.Lys618del)",
  "NM_000179.3(MSH2):c.942+3A>T", "NC_000017.11:g.43071077del",
  "SCN1A c.3700C>T p.Arg1234Ter", "LDLR c.681C>G p.Asp227Glu", "APOE ε2/ε3/ε4 rs429358")
variant_text <- as.vector(outer(variants,
  c("; ClinVar-style variant name", "; heterozygous germline observation",
    "; uncertain significance review", "; transcript consequence prediction",
    "; family segregation study"), paste0))
genes <- c("BRCA1", "BRCA2", "CFTR", "TP53", "HBB", "PTEN", "MLH1", "MSH2", "SCN1A", "LDLR")
conditions <- c("breast cancer", "cystic fibrosis", "epilepsy", "cardiomyopathy", "anaemia",
  "familial hypercholesterolaemia", "colorectal cancer")
# Synthetic PubMed-style abstracts are test inputs, not literature evidence.
abstracts <- vapply(seq_len(74L), function(i) sprintf(
  "Study %d. BACKGROUND: Rare %s variants were investigated in %s. METHODS: We sequenced %d unrelated participants and matched controls. RESULTS: Missense and splice-site alleles showed variable penetrance. CONCLUSIONS: Functional assays and independent replication are needed to interpret these observations.",
  i, genes[(i - 1L) %% length(genes) + 1L], conditions[(i - 1L) %% length(conditions) + 1L], 100L + i), "")
long <- c(paste(rep("genome", 2300L), collapse = " "),
  paste(rep("🧬 BRCA1:c.5266dup café 中文", 450L), collapse = " "))
texts <- c(ascii, unicode, variant_text, abstracts, long)
stopifnot(length(texts) == 200L)
corpus <- data.frame(id = seq_along(texts), text = texts)
dbWriteTable(con, "corpus", corpus)
write.table(corpus, file.path(out_dir, "corpus.tsv"), sep = "\t", quote = TRUE, row.names = FALSE)
requested_tasks <- Sys.getenv("RBEBELM_AUDIT_TASKS", "")
if (nzchar(requested_tasks)) {
  selected <- strsplit(requested_tasks, ",", fixed = TRUE)[[1L]]
  stopifnot(all(selected %in% tasks), !anyDuplicated(selected))
  tasks <- selected
}
requested_dimensions <- Sys.getenv("RBEBELM_AUDIT_DIMENSIONS", "")
if (nzchar(requested_dimensions)) {
  selected <- as.integer(strsplit(requested_dimensions, ",", fixed = TRUE)[[1L]])
  stopifnot(all(selected %in% dimensions), !anyDuplicated(selected))
  dimensions <- selected
}
model_hash <- unname(tools::md5sum(path))
extension_hash <- unname(tools::md5sum(extension))
quoted_path <- dbQuoteString(con, path)

if (benchmark_only) {
  # Fresh inputs ensure every row runs a forward pass; loading is outside the timer.
  bench <- corpus[seq_len(198L), ]
  dbWriteTable(con, "bench", bench)
  sql <- sprintf("SELECT semantic_embed(%s,text,'retrieval_query',768,true) AS v FROM bench", quoted_path)
  invisible(dbGetQuery(con, sprintf("SELECT semantic_embed(%s,'warmup','raw',768,true)", quoted_path)))
  tokens <- sum(vapply(bench$text, function(text)
    length(embeddinggemma_tokenize(model, text, task = "retrieval_query")$ids), 0L))
  times <- replicate(3L, system.time(dbGetQuery(con, sql))[c("user.self", "sys.self", "elapsed")])
  elapsed <- median(times[3L, ])
  cpu <- median(colSums(times[1:2, , drop = FALSE]))
  write.table(t(times), file.path(out_dir, "benchmark-samples.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  result <- data.frame(texts = nrow(bench), tokens = tokens, elapsed_median = elapsed,
    tokens_per_second = tokens / elapsed, texts_per_second = nrow(bench) / elapsed,
    cpu_median = cpu, cpu_tokens_per_second = tokens / cpu)
  write.table(result, file.path(out_dir, "benchmark.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  print(result, digits = 12)
} else {
  metrics <- list()
  wrappers <- list()
  worst <- list()
  for (task in tasks) {
    # Adjacent dimension/normalization variants share one full C projection.
    jobs <- expand.grid(normalize = c(FALSE, TRUE), dimensions = dimensions,
      id = seq_along(texts), KEEP.OUT.ATTRS = FALSE)
    jobs$text <- texts[jobs$id]
    jobs$task <- task
    dbWriteTable(con, "jobs", jobs, overwrite = TRUE)
    elapsed <- system.time(sql_result <- dbGetQuery(con, sprintf(
      "SELECT id, dimensions, normalize, semantic_embed(%s,text,task,dimensions,normalize) AS v FROM jobs",
      quoted_path)))[["elapsed"]]
    cat(sprintf("C task=%s rows=%d elapsed=%.3f\n", task, nrow(sql_result), elapsed)); flush.console()
    for (dim in dimensions) for (norm in c(FALSE, TRUE)) {
      reference_time <- system.time(reference <- embeddinggemma_embed(model, texts,
        task = task, dimensions = dim, normalize = norm, truncate = TRUE))[["elapsed"]]
      selected <- sql_result[sql_result$dimensions == dim & sql_result$normalize == norm, ]
      selected <- selected[order(selected$id), ]
      actual <- do.call(rbind, selected$v)
      delta <- abs(actual - reference)
      cosine <- rowSums(actual * reference) / sqrt(rowSums(actual^2) * rowSums(reference^2))
      # Self is excluded. Ties are resolved by corpus id on both sides.
      a <- actual / sqrt(rowSums(actual^2))
      b <- reference / sqrt(rowSums(reference^2))
      score_a <- tcrossprod(a); score_b <- tcrossprod(b)
      diag(score_a) <- -Inf; diag(score_b) <- -Inf
      rank_a <- apply(score_a, 1L, function(x) order(-x, seq_along(x))[seq_len(10L)])
      rank_b <- apply(score_b, 1L, function(x) order(-x, seq_along(x))[seq_len(10L)])
      mismatch <- sum(rank_a != rank_b)
      k <- length(metrics) + 1L
      metrics[[k]] <- data.frame(task, dimensions = dim, normalize = norm,
        vectors = nrow(actual), values = length(actual), max_abs = max(delta),
        max_rel = max(delta / pmax(abs(reference), 1e-12)), min_cosine = min(cosine),
        rank_mismatches = mismatch, reference_elapsed = reference_time)
      info <- attr(reference, "embedding_info")
      worst[[k]] <- list(task = task, dimensions = dim, normalize = norm,
        index = which(delta == max(delta), arr.ind = TRUE)[1L, ],
        truncated = info$truncated, tokens = info$token_count)
      print(metrics[[k]], digits = 12, row.names = FALSE); flush.console()
      stopifnot(all(is.finite(actual)), all(cosine >= 0.99999), all(apply(delta, 1L, max) <= 1e-4), mismatch == 0L,
        length(info$token_count) == length(texts), all(info$token_count <= 2048L),
        length(info$truncated) == length(texts), all(tail(info$truncated, 2L)))
      write.table(do.call(rbind, metrics), file.path(out_dir, "metrics.tsv"),
        sep = "\t", row.names = FALSE, quote = FALSE)
      saveRDS(worst, file.path(out_dir, "worst.rds"))
      if (task %in% c("retrieval_query", "retrieval_document")) {
        # Convenience wrappers format query/document inputs through their own API.
        wrapper <- if (task == "retrieval_query") embeddinggemma_embed_query(model, texts,
          dimensions = dim, normalize = norm) else embeddinggemma_embed_document(model, texts,
          dimensions = dim, normalize = norm)
        wrapper_delta <- max(abs(actual - wrapper))
        stopifnot(wrapper_delta <= 1e-4)
        wrappers[[length(wrappers) + 1L]] <- data.frame(task, dimensions = dim, normalize = norm,
          vectors = nrow(actual), max_abs = wrapper_delta)
        write.table(do.call(rbind, wrappers), file.path(out_dir, "wrappers.tsv"),
          sep = "\t", row.names = FALSE, quote = FALSE)
      }
    }
  }
  # The documented title=none prompt; the SQL surface has no separate title argument.
  wrappers_result <- do.call(rbind, wrappers)
  summary <- do.call(rbind, metrics)
  cat(sprintf("PASS: %d texts x %d tasks x %d dimensions x 2 normalization settings; %d vectors, %d values\n",
    length(texts), length(tasks), length(dimensions), sum(summary$vectors), sum(summary$values)))
  cat(sprintf("worst abs=%.12g rel=%.12g minimum cosine=%.15g; top-10 mismatches=%d; wrapper vectors=%d\n",
    max(summary$max_abs), max(summary$max_rel), min(summary$min_cosine),
    sum(summary$rank_mismatches), sum(wrappers_result$vectors)))
}
writeLines(c(capture.output(sessionInfo()), paste("Rbebelm", packageVersion("Rbebelm")),
  capture.output(embeddinggemma_model_info(model)), capture.output(Sys.info()),
  paste("model md5", model_hash), paste("extension md5", extension_hash)),
  file.path(out_dir, if (benchmark_only) "benchmark-runtime.txt" else "runtime.txt"))
dbDisconnect(con, shutdown = TRUE)
