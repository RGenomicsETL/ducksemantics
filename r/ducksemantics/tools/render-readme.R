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
})
