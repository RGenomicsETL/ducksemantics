# Render with fail-fast SQL evaluation and cleanup even when a chunk fails.
local({
  env <- new.env(parent = globalenv())
  on.exit({
    if (exists("conn", envir = env, inherits = FALSE) && DBI::dbIsValid(env$conn)) {
      DBI::dbDisconnect(env$conn, shutdown = TRUE)
    }
  })
  rmarkdown::render("README.Rmd", envir = env, quiet = TRUE)
})
