if (requireNamespace("duckdb", quietly = TRUE)) {
  local({
    conn <- ducksemantics::ducksemantics_connect()
    on.exit(DBI::dbDisconnect(conn, shutdown = TRUE))
    fixture <- system.file("extdata", "valid.gguf", package = "ducksemantics",
                           mustWork = TRUE)
    quoted <- as.character(DBI::dbQuoteString(conn, fixture))
    header <- DBI::dbGetQuery(conn, paste0("SELECT * FROM gguf_header(", quoted, ")"))
    expect_equal(header$version, 3)
    expect_equal(header$tensor_count, 3)
    expect_equal(DBI::dbGetQuery(conn, paste0(
      "SELECT value FROM gguf_metadata(", quoted,
      ") WHERE key = 'general.architecture'"))$value, "fixture")
    expect_equal(DBI::dbGetQuery(conn,
      "SELECT semantic_maxsim([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]]) AS score")$score, 1)
    expect_error(DBI::dbGetQuery(conn,
      "SELECT semantic_maxsim([[1.0]], [[1.0, 0.0]])"), "different widths")
    expect_equal(DBI::dbGetQuery(conn,
      "SELECT typeof(semantic_embed(NULL, NULL, NULL, NULL, NULL)) AS type")$type,
      "FLOAT[]")
    expect_equal(DBI::dbGetQuery(conn,
      "SELECT typeof(semantic_colbert_encode(NULL, NULL, NULL)) AS type")$type,
      "FLOAT[][]")
    expect_true(DBI::dbGetQuery(conn,
      "SELECT semantic_generate(NULL, 'Hello', 8) IS NULL AS missing")$missing)
    tokenizer <- system.file("extdata", "tokenizer_gemma.gguf", package = "ducksemantics",
                             mustWork = TRUE)
    tokens <- DBI::dbGetQuery(conn, paste0(
      "SELECT semantic_detokenize(", DBI::dbQuoteString(conn, tokenizer),
      ", semantic_tokenize(", DBI::dbQuoteString(conn, tokenizer),
      ", 'abc', false)) AS text"))
    expect_equal(tokens$text, "abc")
    bad <- system.file("extdata", "bad_magic.gguf", package = "ducksemantics",
                       mustWork = TRUE)
    expect_error(DBI::dbGetQuery(conn, paste0("SELECT * FROM gguf_header(",
      DBI::dbQuoteString(conn, bad), ")")), "magic")
    expect_true(file.exists(ducksemantics::ducksemantics_extension_path()))
  })
  local({
    conn <- ducksemantics::ducksemantics_connect(extension = FALSE)
    on.exit(DBI::dbDisconnect(conn, shutdown = TRUE))
    expect_equal(DBI::dbGetQuery(conn,
      "SELECT count(*) AS n FROM duckdb_functions() WHERE function_name = 'semantic_maxsim'")$n,
      0)
    ducksemantics::ducksemantics_init(conn)
    expect_true(DBI::dbExistsTable(conn, "semantic_nodes"))
  })
}
