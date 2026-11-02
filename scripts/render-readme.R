# Render README.Rmd with fail-fast chunk evaluation, close its DuckDB connection
# whether rendering succeeds or fails, then build the static site from README.md.
local({
  input <- "README.Rmd"
  model <- Sys.getenv("DUCKSEMANTICS_EMBED_MODEL")
  if (!nzchar(model) || !file.exists(model)) {
    stop("DUCKSEMANTICS_EMBED_MODEL must name a local EmbeddingGemma GGUF file.",
         call. = FALSE)
  }
  headers <- grep(
    "^```\\{(r|sql)",
    readLines(input, warn = FALSE, encoding = "UTF-8"),
    value = TRUE
  )
  checked <- headers[!grepl("^```\\{r setup([,}])", headers)]
  disabled <- grepl(
    "(eval|include)\\s*=\\s*FALSE|error\\s*=\\s*TRUE",
    checked
  )
  if (any(disabled)) {
    stop("README model and relation examples must execute: ",
         paste(checked[disabled], collapse = "; "), call. = FALSE)
  }

  env <- new.env(parent = globalenv())
  on.exit({
    if (exists("conn", envir = env, inherits = FALSE) && DBI::dbIsValid(env$conn)) {
      DBI::dbDisconnect(env$conn, shutdown = TRUE)
    }
  }, add = TRUE)
  rmarkdown::render(input, envir = env, quiet = TRUE)
  rendered <- readLines("README.md", warn = FALSE, encoding = "UTF-8")
  required <- c("readme-hypothesis-1", "semantic_embed")
  absent <- required[!vapply(required, function(x) any(grepl(x, rendered, fixed = TRUE)), logical(1))]
  if (length(absent) > 0L) {
    stop("Rendered README is missing required results: ",
         paste(absent, collapse = ", "), call. = FALSE)
  }
})
source(file.path("docs", "build-site.R"))
