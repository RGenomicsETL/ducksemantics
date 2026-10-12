# Convert the rendered README.md and docs/ pages to static HTML in build/site.
# Chunks are not executed here: README.md is the output of README.Rmd.
local({
  repository <- "https://github.com/RGenomicsETL/ducksemantics/blob/main/"
  destination <- file.path("build", "site")
  pages <- c(
    "README.md" = "index.html",
    "docs/native-kernels.md" = "native-kernels.html",
    "docs/semantic-relations.md" = "semantic-relations.html"
  )
  titles <- c("Walkthrough", "Native kernels", "Semantic relations")
  stopifnot(file.exists(names(pages)))
  dir.create(destination, recursive = TRUE, showWarnings = FALSE)
  destination <- normalizePath(destination)
  file.copy(file.path("docs", "site.css"), destination, overwrite = TRUE)

  # Each Markdown source links to its rendered page; other relative links
  # resolve to the repository file.
  relink <- function(text, source) {
    base <- dirname(source)
    for (target in names(pages)) {
      relative <- if (base == ".") target else basename(target)
      text <- gsub(paste0("](", relative), paste0("](", pages[[target]]), text, fixed = TRUE)
    }
    prefix <- if (base == ".") repository else paste0(repository, base, "/")
    gsub("\\]\\((?![[:alpha:]][[:alnum:]+.-]*:|#|[^)]*\\.html)([^)]+)\\)",
         paste0("](", prefix, "\\1)"), text, perl = TRUE)
  }

  nav <- tempfile(fileext = ".html")
  on.exit(unlink(nav), add = TRUE)
  writeLines(c(
    '<header class="site-header"><a class="site-name" href="index.html">DuckSemantics</a><nav>',
    sprintf('<a href="%s">%s</a>', pages, titles),
    sprintf('<a href="%s">Source</a>', sub("/blob/main/$", "", repository)),
    "</nav></header>"
  ), nav)

  for (source in names(pages)) {
    text <- readLines(source, warn = FALSE, encoding = "UTF-8")
    title <- sub("^# ", "", text[startsWith(text, "# ")][1L])
    input <- tempfile(fileext = ".md")
    writeLines(relink(text, source), input, useBytes = TRUE)
    rmarkdown::pandoc_convert(
      input, to = "html5", from = "gfm",
      output = file.path(destination, pages[[source]]),
      options = c("--standalone", "--toc", "--toc-depth=2",
                  "--css=site.css", "--highlight-style=tango",
                  paste0("--metadata=pagetitle:", title, " | DuckSemantics"),
                  paste0("--include-before-body=", nav))
    )
    unlink(input)
  }
  message("Site written to ", destination)
})
