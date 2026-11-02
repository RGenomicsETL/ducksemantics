# Convert rendered Markdown pages to static HTML in build/site.
# Chunks are not executed here; README.md files are outputs of README.Rmd.
local({
  repository <- "https://github.com/RGenomicsETL/ducksemantics/blob/main/"
  destination <- file.path("build", "site")
  pages <- c(
    "README.md" = "index.html",
    "r/ducksemantics/README.md" = "r-package.html",
    "docs/semantic-relations.md" = "semantic-relations.html",
    "docs/native-kernels.md" = "native-kernels.html",
    "docs/roadmap.md" = "roadmap.html"
  )
  navigation <- c(
    "README.md" = "Overview",
    "r/ducksemantics/README.md" = "R package",
    "docs/semantic-relations.md" = "Relations",
    "docs/native-kernels.md" = "Kernels"
  )
  generated <- c(
    "README.Rmd" = "README.md",
    "r/ducksemantics/README.Rmd" = "r/ducksemantics/README.md"
  )
  required_claims <- list(
    "README.md" = c("readme-hypothesis-1", "semantic_embed"),
    "r/ducksemantics/README.md" = c("walkthrough-h1", "state-abstained"),
    "docs/semantic-relations.md" = c("ducksemantics_hypothesis_source", "Rejection behavior"),
    "docs/native-kernels.md" = c("semantic_generate_stats", "Verified native prefill profile"),
    "docs/roadmap.md" = c("Explicit shared-prefix generation", "physical prompt work")
  )

  missing <- names(pages)[!file.exists(names(pages))]
  if (length(missing) > 0L) {
    stop("Site source files are missing: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  for (source in names(generated)) {
    output <- generated[[source]]
    if (file.info(source)$mtime > file.info(output)$mtime) {
      stop(output, " is older than ", source, "; render the README first.",
           call. = FALSE)
    }
  }

  dir.create(destination, recursive = TRUE, showWarnings = FALSE)
  destination <- normalizePath(destination)
  file.copy(file.path("docs", "site.css"), destination, overwrite = TRUE)

  path_parts <- function(path) {
    path <- gsub("^\\./|/$", "", path)
    if (!nzchar(path) || identical(path, ".")) character() else strsplit(path, "/", fixed = TRUE)[[1L]]
  }
  relative_path <- function(source, target) {
    from <- path_parts(dirname(source))
    to <- path_parts(target)
    common <- 0L
    limit <- min(length(from), length(to))
    while (common < limit && identical(from[[common + 1L]], to[[common + 1L]])) {
      common <- common + 1L
    }
    remainder <- if (common == 0L) to else to[-seq_len(common)]
    paste(c(rep("..", length(from) - common), remainder), collapse = "/")
  }
  relink <- function(text, source) {
    for (target in names(pages)) {
      relative <- relative_path(source, target)
      text <- gsub(
        paste0("](", relative, ")"),
        paste0("](", pages[[target]], ")"),
        text,
        fixed = TRUE
      )
    }
    base <- dirname(source)
    prefix <- if (base == ".") repository else paste0(repository, base, "/")
    gsub(
      "\\]\\((?![[:alpha:]][[:alnum:]+.-]*:|#|[^)]*\\.html)([^)]+)\\)",
      paste0("](", prefix, "\\1)"), text, perl = TRUE
    )
  }

  nav <- tempfile(fileext = ".html")
  on.exit(unlink(nav), add = TRUE)
  writeLines(c(
    '<header class="site-header"><a class="site-name" href="index.html">DuckSemantics</a><nav>',
    sprintf('<a href="%s">%s</a>', pages[names(navigation)], navigation),
    sprintf('<a href="%s">Source</a>', sub("/blob/main/$", "", repository)),
    "</nav></header>"
  ), nav)

  for (source in names(pages)) {
    text <- readLines(source, warn = FALSE, encoding = "UTF-8")
    title <- sub("^# ", "", text[startsWith(text, "# ")][1L])
    input <- tempfile(fileext = ".md")
    output <- file.path(destination, pages[[source]])
    writeLines(relink(text, source), input, useBytes = TRUE)
    rmarkdown::pandoc_convert(
      input, to = "html5", from = "gfm", output = output,
      options = c(
        "--standalone", "--toc", "--toc-depth=2", "--css=site.css",
        "--highlight-style=tango",
        paste0("--metadata=pagetitle:", title, " | DuckSemantics"),
        paste0("--include-before-body=", nav)
      )
    )
    unlink(input)

    html <- gsub(
      "[[:space:]]+", " ",
      paste(readLines(output, warn = FALSE, encoding = "UTF-8"), collapse = " ")
    )
    absent <- required_claims[[source]][
      !vapply(required_claims[[source]], grepl, logical(1), x = html, fixed = TRUE)
    ]
    if (length(absent) > 0L) {
      stop("Rendered site page ", pages[[source]], " is missing claims: ",
           paste(absent, collapse = ", "), call. = FALSE)
    }
  }
  message("Site written to ", destination)
})
