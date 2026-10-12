local({
  fixture <- function(name) system.file("extdata", name, package = "ducksemantics", mustWork = TRUE)
  receipts <- utils::read.csv(fixture("source-releases.csv"), stringsAsFactors = FALSE)
  receipts$available_at <- as.POSIXct(receipts$available_at, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  catalog <- ducksemantics::ducksemantics_source_release_catalog(receipts)
  expect_equal(catalog$receipt_id, receipts$receipt_id)
  expect_equal(catalog$upstream_sha256, receipts$upstream_sha256)
  mondo_release <- ducksemantics::ducksemantics_source_release_select(catalog, "Mondo",
    as_of = as.POSIXct("2026-10-07", tz = "UTC"))
  expect_equal(mondo_release$release_id, "v2026-10-06")
  expect_error(ducksemantics::ducksemantics_source_release_select(catalog, "Mondo",
    as_of = as.POSIXct("2026-10-06", tz = "UTC")), "visible")
  expect_error(ducksemantics::ducksemantics_source_release_select(catalog, "DisMech",
    as_of = as.POSIXct("2026-10-07", tz = "UTC")), "visible")
  expect_error(ducksemantics::ducksemantics_source_release_select(catalog, "Mondo",
    as_of = "2026-10-07"), "POSIXct")
  expect_error(ducksemantics::ducksemantics_source_release_select(catalog, "Mondo"), "exactly one")
  later <- mondo_release
  later$release_id <- "v2026-01-01"
  later$available_at <- as.POSIXct("2026-10-08", tz = "UTC")
  later$receipt_id <- "later-observed-artifact"
  epochs <- rbind(mondo_release, later)
  expect_equal(ducksemantics::ducksemantics_source_release_select(epochs, "Mondo",
    as_of = as.POSIXct("2026-10-09", tz = "UTC"))$release_id, "v2026-01-01")
  expect_equal(ducksemantics::ducksemantics_source_release_select(epochs, "Mondo",
    as_of = as.POSIXct("2026-10-07", tz = "UTC"))$release_id, "v2026-10-06")
  later$available_at <- mondo_release$available_at
  expect_error(ducksemantics::ducksemantics_source_release_catalog(rbind(mondo_release, later)),
    "unambiguous")
  invalid <- catalog
  invalid$available_at <- as.character(invalid$available_at)
  expect_error(ducksemantics::ducksemantics_source_release_catalog(invalid), "POSIXct")

  mondo <- ducksemantics::ducksemantics_mondo_import(fixture("mondo-subset.obo"),
    mondo_release, mondo_release$source_id, mondo_release$source_uri, mondo_release$sha256)
  expect_equal(mondo$terms$node_id, c("MONDO:0007947", "MONDO:0000002", "MONDO:0009299"))
  expect_equal(mondo$terms$is_obsolete, c(FALSE, TRUE, FALSE))
  expect_equal(mondo$terms$label[[1L]], "Marfan syndrome")
  expect_true(any(mondo$synonyms$alias == "Marfan's syndrome"))
  expect_true(any(mondo$synonyms$alias_kind == "synonym:exact"))
  expect_true(any(mondo$mappings$value == "skos:exactMatch OMIM:154700"))
  expect_true(any(grepl('^OMIM:154700 .*MONDO:equivalentTo', mondo$mappings$value)))
  expect_equal(mondo$obsolescence$value[mondo$obsolescence$tag == "replaced_by"], "MONDO:0009299")
  expect_true(all(mondo$tags$release_id == mondo_release$release_id))
  expect_true(all(mondo$tags$receipt_id == mondo_release$receipt_id))
  expect_false(anyDuplicated(mondo$tags$source_row_id) > 0L)
  expect_equal(mondo$tags$source_text[mondo$tags$tag == "synonym"][[1L]],
    'synonym: "Marfan syndrome" EXACT [DOID:14323, ICD10CM:Q87.4, icd11.foundation:236564145, NCIT:C34807, OMIM:154700, Orphanet:558] {OMO:0002001="https://w3id.org/information-resource-registry/clingen"}')
  expect_error(ducksemantics::ducksemantics_mondo_import(fixture("mondo-subset.obo"),
    mondo_release, mondo_release$source_id, mondo_release$source_uri, paste(rep("0", 64), collapse = "")),
    "SHA-256")

  whitespace <- tempfile(fileext = ".obo")
  on.exit(unlink(whitespace), add = TRUE)
  writeLines(c("[Term]", "  id: MONDO:1234567  ", "  name: Example disease  ",
    "alt_id: MONDO:7654321", "consider: MONDO:0007947", "", "[Term]",
    "id: HP:0000001", "name: External phenotype"), whitespace)
  projected <- ducksemantics::ducksemantics_mondo_import(whitespace, mondo_release,
    "whitespace.obo", "fixture://whitespace.obo", digest::digest(file = whitespace, algo = "sha256"))
  expect_equal(projected$terms$node_id, "MONDO:1234567")
  expect_equal(projected$terms$label, "Example disease")
  expect_true("MONDO:7654321" %in% projected$synonyms$alias)
  expect_equal(projected$obsolescence$value, "MONDO:0007947")
  expect_equal(projected$tags$source_text[projected$tags$tag == "name"],
    "  name: Example disease  ")

  dismech_release <- ducksemantics::ducksemantics_source_release_select(catalog, "DisMech",
    release_id = "fac61860115b1fdb194de4f4cacef3a8f6a398af")
  dismech <- ducksemantics::ducksemantics_dismech_import(fixture("dismech-marfan.yaml"),
    dismech_release, dismech_release$source_id, dismech_release$source_uri, dismech_release$sha256)
  expect_equal(dismech$records$name, "Marfan Syndrome")
  expect_equal(dismech$records$yaml_text, rawToChar(readBin(fixture("dismech-marfan.yaml"),
    "raw", n = file.info(fixture("dismech-marfan.yaml"))$size)))
  field <- function(path) dismech$fields$value[dismech$fields$field_path == path]
  expect_equal(field("/disease_term/term/id"), "MONDO:0007947")
  expect_equal(field("/pathophysiology/0/name"), "FBN1 Gene Mutation")
  expect_equal(field("/pathophysiology/0/evidence/0/reference"), "PMID:8180508")
  expect_equal(field("/pathophysiology/0/evidence/0/snippet"),
    paste("The gene encoding for fibrillin, the FBN1 gene on chromosome 15, has been identified",
      "as the defective gene causing the Marfan syndrome. Fibrillin is the large glycoprotein",
      "with a repetitive domain structure and is a major protein component of microfibrils,",
      "a fibrillar system closely associated with elastin in connective tissue."))
  expect_true("/pathophysiology/0/cell_types/0/term/id" %in% dismech$fields$field_path)
  expect_equal(dismech$fields$source_row_id[dismech$fields$field_path ==
    "/pathophysiology/0/evidence/0/reference"],
    "kb/disorders/Marfan_Syndrome.yaml#/pathophysiology/0/evidence/0/reference")
  expect_false(anyDuplicated(dismech$fields$source_row_id) > 0L)
  expect_true(all(dismech$fields$release_id == dismech_release$release_id))
  expect_false("proposals" %in% names(dismech))
  expect_error(ducksemantics::ducksemantics_dismech_import(fixture("dismech-marfan.yaml"),
    catalog, "record", "uri", dismech_release$sha256), "exactly one")

  if (requireNamespace("duckdb", quietly = TRUE)) {
    conn <- ducksemantics::ducksemantics_connect()
    on.exit(DBI::dbDisconnect(conn, shutdown = TRUE), add = TRUE)
    DBI::dbWriteTable(conn, "mondo_terms", mondo$terms)
    DBI::dbWriteTable(conn, "dismech_fields", dismech$fields)
    DBI::dbWriteTable(conn, "source_releases", catalog)
    expect_equal(DBI::dbGetQuery(conn, "SELECT count(*) AS n FROM mondo_terms")$n, 3)
    expect_equal(ducksemantics::ducksemantics_source_release_select("source_releases", "Mondo",
      as_of = as.POSIXct("2026-10-07", tz = "UTC"), conn = conn)$release_id, "v2026-10-06")
    expect_equal(DBI::dbGetQuery(conn,
      "SELECT value FROM dismech_fields WHERE field_path = '/pathophysiology/0/evidence/0/reference'")$value,
      "PMID:8180508")
  }
})
