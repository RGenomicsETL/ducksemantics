#!/usr/bin/env Rscript
# Exact token-id audit against the installed Rbebelm CPU models.
# Run from the worktree root: Rscript scripts/audit_tokenizer.R [extension] [output-dir]
library(DBI)
library(duckdb)
library(Rbebelm)
library(xml2)

args <- commandArgs(trailingOnly = TRUE)
extension <- normalizePath(if (length(args) > 0L) args[[1L]] else "build/ducksemantics.duckdb_extension")
outdir <- if (length(args) > 1L) args[[2L]] else "build/tokenizer-audit"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
con <- dbConnect(duckdb(config = list(allow_unsigned_extensions = "true", threads = "1"),
                        home = file.path(normalizePath(outdir), "duckdb-home")))
quote_sql <- function(x) as.character(dbQuoteString(con, x))
dbExecute(con, paste("LOAD", quote_sql(extension)))

xml_paths <- Sys.glob("/root/RClinVarbitration/r/RClinVarbitration/inst/extdata/*.xml")
stopifnot(length(xml_paths) > 0L)
fixture_text <- unlist(lapply(xml_paths, function(path) {
  nodes <- xml_find_all(read_xml(path), "//ArticleTitle | //AbstractText | //BookTitle")
  xml_text(nodes)
}), use.names = FALSE)
papers <- read.delim("scripts/tokenizer_biomedical.tsv", quote = "", check.names = FALSE)
seeds <- unique(c(
  "", "Hello world", "ASCII", "I'm don't we're they've I'll I'D it's", "'s 't 'll 're 've 'm 'd",
  "1234 1234567890", "١٢٣٤ １２３４ ²³ Ⅷ ½", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
  " \t\n\r\n  ", "  hi", "a\n\nb\r\n c\t\t", "a \n \t b  ", "\tword\nword\rword",
  "😀😃🧬🧪🩺", "👩🏽‍🔬 👨‍👩‍👧‍👦 🇫🇷", "中文汉字 日本語 한국어", "مرحبا بالعالم العربية",
  "café naïve résumé Straße Łódź Ελληνικά", "e\u0301 a\u0308 \u0345 \u05b0 \u093e",
  "\u00a0\u1680\u2000\u2001\u2002\u2003\u2009\u2028\u2029\u202f\u205f\u3000",
  "\u200b\u200c\u200d\ufeff", "a\u0301b\u0302c", "אבג русский हिन्दी ไทย বাংলা",
  "\U00010300\U00010400\U0001d400\U0001e900 \U0001d7ce\U0001d7cf", "<|im_start|>user\nhi<|im_end|>",
  "<|startoftext|><|endoftext|>", "<bos><eos><pad><unk>", "[Q] BRCA1 [D] disease",
  "<|not_a_special|> </s> <s>", "if (x <= 10) {\n\treturn x + 1;\n}",
  "SELECT gene, count(*) FROM variants WHERE p < 5e-8 GROUP BY gene;",
  "f <- function(x) { sum(x, na.rm = TRUE) } # αβ", "{\"name\":\"BRCA1\",\"position\":43071077}",
  "NM_007294.4(BRCA1):c.5266dup (p.Gln1756fs)", "NC_000017.11:g.43071077dup",
  "NM_000546.6(TP53):c.743G>A (p.Arg248Gln)", "BRAF V600E; EGFR L858R; HLA-B*57:01",
  "IL-6 TNF-α β-catenin γ-H2AX ΔF508", "P = 2.1×10−8; 95% CI: 1.2–3.4; n=1,024",
  "A\r\nB\n\rC\tD\vE\fF", " \n\n   ", "\n!\n\n?\r\n", "123abc456def7890",
  "foo_bar camelCase HTTPServer 0xDEADBEEF", "https://pubmed.ncbi.nlm.nih.gov/8524414/",
  "RNA-seq reads were aligned to GRCh38; variants were classified using ACMG/AMP criteria.",
  fixture_text, papers$title, papers$abstract
))
corpus <- unique(c(seeds, paste0(" ", seeds), paste0(seeds, "\t\n"),
                   paste0("α ", seeds, "  β"), paste0("1234: ", seeds, "!?"),
                   paste0("\t", seeds, "\r\nEND")))
set.seed(8401)
fragments <- c("a", "abc", "1234", " café", "\t", "\n", " ", "!", "😀", "中文", "العربية", "e\u0301")
random <- replicate(80L, paste0(sample(fragments, sample(3:16, 1L), replace = TRUE), collapse = ""))
corpus <- unique(c(corpus, random, paste(rep("BRCA1 café 1234\t\n", 800L), collapse = ""),
                   paste(rep("α😀漢 e\u0301 ", 1600L), collapse = ""),
                   paste(rep(" mitochondrial", 1000L), collapse = ""),
                   paste(rep(" internationalization", 600L), collapse = "")))
stopifnot(length(corpus) >= 300L, max(nchar(corpus)) > 10000L)
saveRDS(corpus, file.path(outdir, "corpus.rds"))
dbWriteTable(con, "corpus", data.frame(row_id = seq_along(corpus), text = corpus))

models <- c(lfm2moe = "/root/bebelm/LFM2.5-8B-A1B-Q4_K_M.gguf",
            colbert = "/root/bebelm/LFM2.5-ColBERT-350M-Q4_K_M.gguf",
            gemma = "/root/bebelm/embeddinggemma-300M-Q8_0.gguf")
results <- list()
check_ids <- function(actual, expected, label, texts = corpus) {
  good <- vapply(seq_along(expected), function(i) identical(as.integer(actual[[i]]), as.integer(expected[[i]])), logical(1))
  if (!all(good)) {
    bad <- which(!good)
    saveRDS(list(label = label, rows = bad, text = texts[bad],
                 actual = actual[bad], expected = expected[bad]),
            file.path(outdir, paste0(label, "-mismatches.rds")))
    stop(label, ": ", length(bad), " mismatches; see audit output")
  }
  cat(label, ": ", length(good), " exact sequences; max id error 0\n", sep = "")
  length(good)
}
sql_ids <- function(path, special, prefix = "") {
  query <- sprintf("SELECT semantic_tokenize(%s, %s || text, %s) AS ids FROM corpus ORDER BY row_id",
                   quote_sql(path), quote_sql(prefix), if (special) "true" else "false")
  dbGetQuery(con, query)$ids
}
sql_decode <- function(path) {
  dbGetQuery(con, sprintf("SELECT semantic_detokenize(%s, semantic_tokenize(%s,text,true)) AS text FROM corpus ORDER BY row_id",
                         quote_sql(path), quote_sql(path)))$text
}
benchmark <- function(path, token_count) {
  query <- sprintf("SELECT sum(len(semantic_tokenize(%s,text,false))) AS n FROM corpus", quote_sql(path))
  dbGetQuery(con, query)
  reps <- 200L
  elapsed <- system.time(for (i in seq_len(reps)) dbGetQuery(con, query))[["elapsed"]]
  list(tokens = token_count * reps, seconds = elapsed, tokens_per_second = token_count * reps / elapsed,
       repetitions = reps, sql_threads = 1L)
}

cat("Runtime: R ", getRversion() |> as.character(), "; Rbebelm ", as.character(packageVersion("Rbebelm")),
    "; duckdb ", as.character(packageVersion("duckdb")), "; corpus ", length(corpus), "\n", sep = "")
for (family in names(models)) {
  path <- models[[family]]
  cat("Loading ", family, " once\n", sep = "")
  meta <- dbGetQuery(con, sprintf("SELECT key,value FROM gguf_metadata(%s)", quote_sql(path)))
  metadata_id <- function(key) as.integer(meta$value[match(paste0("tokenizer.ggml.", key), meta$key)])
  actual <- sql_ids(path, TRUE)
  plain <- sql_ids(path, FALSE)
  truncated_count <- 0L
  if (family == "lfm2moe") {
    model <- bebel_model_load(path, num_threads = 4L)
    encode <- function(text, special) {
      if (!nzchar(text)) model$encode(text, add_bos = special)
      else bebel_tokenize(model, text, add_bos = special)
    }
    expected <- lapply(corpus, encode, special = TRUE)
    expected_plain <- lapply(corpus, encode, special = FALSE)
    n <- check_ids(actual, expected, family)
    check_ids(plain, expected_plain, paste0(family, "-no-special"))
    decoded <- vapply(expected, function(ids) bebel_detokenize(model, ids), character(1))
    stopifnot(identical(sql_decode(path), decoded))
    cat("lfm2moe decode: ", length(decoded), " exact strings\n", sep = "")
    extra_ids <- list(integer(), c(metadata_id("bos_token_id"), metadata_id("eos_token_id")), 0:93, 94:120)
    for (ids in extra_ids) {
      literal <- paste0("[", paste(ids, collapse = ","), "]::INTEGER[]")
      got <- dbGetQuery(con, sprintf("SELECT semantic_detokenize(%s,%s) AS text", quote_sql(path), literal))$text
      stopifnot(identical(got, bebel_detokenize(model, ids)))
    }
    decode_count <- length(decoded) + length(extra_ids)
  } else if (family == "colbert") {
    model <- colbert_model_load(path, num_threads = 4L)
    info <- colbert_model_info(model)
    print(info)
    punctuation <- strsplit("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", "", fixed = TRUE)[[1L]]
    skip <- unique(unlist(lapply(punctuation, function(text) {
      dbGetQuery(con, sprintf("SELECT semantic_tokenize(%s,%s,false) AS ids", quote_sql(path), quote_sql(text)))$ids[[1L]]
    })))
    query_ids <- sql_ids(path, TRUE, "[Q] ")
    document_ids <- sql_ids(path, TRUE, "[D] ")
    expected_query <- lapply(query_ids, function(ids) head(c(ids, rep(metadata_id("padding_token_id"), info$query_length)), info$query_length))
    expected_doc <- lapply(document_ids, function(ids) {
      ids <- head(ids, info$document_length)
      ids[!ids %in% skip]
    })
    reference_query <- reference_doc <- vector("list", length(corpus))
    supported <- which(nzchar(corpus))
    for (i in supported) {
      query <- colbert_encode_query(model, corpus[[i]])
      document <- colbert_encode_document(model, corpus[[i]])
      reference_query[[i]] <- colbert_embedding_ids(query)
      reference_doc[[i]] <- colbert_embedding_ids(document)
      if (i %% 25L == 0L) cat("colbert oracle ", i, "/", length(corpus), "\n", sep = "")
    }
    n <- check_ids(expected_query[supported], reference_query[supported], "colbert-query", corpus[supported])
    check_ids(expected_doc[supported], reference_doc[supported], "colbert-document", corpus[supported])
    cat("ColBERT R wrapper rejects empty text; empty SQL input is covered by fixtures.\n")
    # ColBERT exposes contextualized retained ids, not a raw tokenization/decode method.
    decode_count <- NA_integer_
  } else {
    model <- embeddinggemma_model_load(path, num_threads = 4L)
    context <- as.integer(meta$value[match("gemma-embedding.context_length", meta$key)])
    expected <- lapply(corpus, function(text) embeddinggemma_tokenize(model, text, task = "raw", truncate = TRUE)$ids)
    supported <- which(lengths(actual) <= context)
    truncated <- which(lengths(actual) > context)
    expected_plain <- lapply(expected, function(ids) ids[-c(1L, length(ids))])
    n <- check_ids(actual[supported], expected[supported], family, corpus[supported])
    check_ids(plain[supported], expected_plain[supported], paste0(family, "-no-special"), corpus[supported])
    limited <- lapply(actual[truncated], function(ids) c(head(ids, context - 1L), tail(ids, 1L)))
    truncated_count <- check_ids(limited, expected[truncated], "gemma-truncated", corpus[truncated])
    decode_count <- NA_integer_
  }
  bench <- benchmark(path, sum(lengths(plain)))
  print(bench)
  results[[family]] <- list(path = path, strings = n, truncated_strings = truncated_count, max_id_error = 0L, decode_strings = decode_count,
                             benchmark = bench)
  saveRDS(list(family = family, path = path, text = corpus, ids = actual, plain_ids = plain),
          file.path(outdir, paste0(family, "-ids.rds")))
  saveRDS(results, file.path(outdir, "results.rds"))
  rm(model); gc()
}
writeLines(capture.output(sessionInfo()), file.path(outdir, "session.txt"))
dbDisconnect(con, shutdown = TRUE)
cat("PASS: exact Rbebelm sequences for all three model families\n")
