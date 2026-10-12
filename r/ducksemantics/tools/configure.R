# Build solely from the bundled canonical extension tree.
run <- function(command, args, capture = FALSE) {
  value <- system2(command, shQuote(args), stdout = if (capture) TRUE else "",
                   stderr = if (capture) TRUE else "")
  status <- if (capture) attr(value, "status") else value
  if (!is.null(status) && status != 0L) stop(command, " failed.", call. = FALSE)
  value
}
r_config <- function(name) {
  paste(run(file.path(R.home("bin"), "R"), c("CMD", "config", name), TRUE),
        collapse = " ")
}
platform <- Sys.getenv("DUCKDB_PLATFORM")
os <- Sys.info()[["sysname"]]
if (!os %in% c("Linux", "Darwin")) {
  stop("The native mapped-file reader currently requires Linux or macOS.", call. = FALSE)
}
if (!nzchar(platform)) {
  cpu <- switch(R.version$arch, x86_64 = "amd64", aarch64 = "arm64", arm64 = "arm64")
  if (is.null(cpu)) stop("Set DUCKDB_PLATFORM for this target.", call. = FALSE)
  platform <- paste(if (os == "Darwin") "osx" else "linux", cpu, sep = "_")
}
compiler <- Sys.getenv("CC", r_config("CC"))
make <- Sys.getenv("MAKE", "make")
if (!nzchar(Sys.which(make))) stop("GNU make is required.", call. = FALSE)
build <- file.path(tempdir(), "ducksemantics-extension")
dir.create(build, recursive = TRUE, showWarnings = FALSE)
if (!file.copy("inst/extension", build, recursive = TRUE)) {
  stop("Could not stage the bundled extension sources.", call. = FALSE)
}
build <- file.path(build, "extension")
run(make, c("-C", build, "all", paste0("CC=", compiler),
            paste0("CFLAGS=", Sys.getenv("CFLAGS", r_config("CFLAGS"))),
            paste0("CPPFLAGS=", Sys.getenv("CPPFLAGS", r_config("CPPFLAGS"))),
            paste0("LDFLAGS=", Sys.getenv("LDFLAGS", r_config("LDFLAGS"))),
            paste0("SHLIB_LDFLAGS=", if (os == "Darwin") "-dynamiclib" else "-shared"),
            paste0("PLATFORM=", platform)))
dir.create("inst/duckdb", recursive = TRUE, showWarnings = FALSE)
if (!file.copy(file.path(build, "build", "ducksemantics.duckdb_extension"),
               "inst/duckdb", overwrite = TRUE)) {
  stop("Could not stage the built extension.", call. = FALSE)
}
