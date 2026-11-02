local({
  input <- "README.Rmd"
  headers <- grep(
    "^```\\{r",
    readLines(input, warn = FALSE, encoding = "UTF-8"),
    value = TRUE
  )
  checked <- headers[!grepl("^```\\{r setup([,}])", headers)]
  disabled <- grepl(
    "(eval|include)\\s*=\\s*FALSE|error\\s*=\\s*TRUE",
    checked
  )
  if (any(disabled)) {
    stop("README relation examples must execute: ",
         paste(checked[disabled], collapse = "; "), call. = FALSE)
  }
  rmarkdown::render(input, output_file = "README.md", quiet = TRUE)
  rendered <- readLines("README.md", warn = FALSE, encoding = "UTF-8")
  required <- c("walkthrough-h1", "state-abstained")
  absent <- required[!vapply(required, function(x) any(grepl(x, rendered, fixed = TRUE)), logical(1))]
  if (length(absent) > 0L) {
    stop("Rendered package README is missing required results: ",
         paste(absent, collapse = ", "), call. = FALSE)
  }
})
