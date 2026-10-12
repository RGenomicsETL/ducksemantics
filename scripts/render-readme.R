# Render README.Rmd with fail-fast chunk evaluation, close its DuckDB connection
# whether rendering succeeds or fails, then build the static site from README.md.
local({
  env <- new.env(parent = globalenv())
  on.exit({
    if (exists("conn", envir = env, inherits = FALSE) && DBI::dbIsValid(env$conn)) {
      DBI::dbDisconnect(env$conn, shutdown = TRUE)
    }
  }, add = TRUE)
  rmarkdown::render("README.Rmd", envir = env, quiet = TRUE)
})
source(file.path("docs", "build-site.R"))
