snapshots <- data.frame(
  provider_id = "pubmed",
  snapshot_id = c("baseline", "update", "delete"),
  high_water_ordinal = c(1, 2, 3)
)
articles <- data.frame(
  provider_id = "pubmed",
  article_id = "1001",
  pmid = "1001",
  version_id = snapshots$snapshot_id,
  source_ordinal = snapshots$high_water_ordinal,
  is_deleted = c(FALSE, FALSE, TRUE)
)
sections <- data.frame(
  provider_id = "pubmed",
  article_id = "1001",
  pmid = "1001",
  version_id = c(rep("baseline", 4), rep("update", 2)),
  source_ordinal = c(rep(1, 4), rep(2, 2)),
  section = c("title", "abstract", "abstract", "abstract", "title", "abstract"),
  text = c(
    "Unrelated title",
    "Seizure; seizure",
    "Seizure; seizure",
    "No neurological features",
    "Corrected title",
    "Updated seizure"
  ),
  subsection = c(NA, "BACKGROUND", "BACKGROUND", "RESULTS", NA, "METHODS"),
  source_entity_ordinal = c(NA, 3, 5, 8, NA, 3)
)

hits <- ducksemantics_literature_retrieve(
  articles,
  sections,
  snapshots,
  "seizure",
  "pubmed",
  "baseline"
)
expect_equal(nrow(hits), 4L)
expect_equal(hits$source_text, rep(c("Seizure", "seizure"), 2))
expect_equal(hits$start_offset, rep(c(0L, 9L), 2))
expect_equal(hits$end_offset, rep(c(7L, 16L), 2))
expect_equal(hits$subsection, rep("BACKGROUND", 4))
expect_equal(hits$source_entity_ordinal, c(3, 3, 5, 5))
expect_equal(anyDuplicated(hits), 0L)
expect_true(all(hits$version_id == "baseline"))

updated <- ducksemantics_literature_retrieve(
  articles,
  sections,
  snapshots,
  "seizure",
  "pubmed",
  "update"
)
expect_equal(updated$version_id, "update")
expect_equal(updated$source_entity_ordinal, 3)
expect_equal(updated$subsection, "METHODS")
expect_equal(updated$source_text, "seizure")

absent <- ducksemantics_literature_retrieve(
  articles,
  sections,
  snapshots,
  "absent phrase",
  "pubmed",
  "baseline"
)
expect_equal(nrow(absent), 0L)
expect_equal(names(absent), names(hits))
expect_true(is.character(absent$subsection))
expect_true(is.numeric(absent$source_entity_ordinal))
title_only <- ducksemantics_literature_retrieve(
  articles,
  sections,
  snapshots,
  "seizure",
  "pubmed",
  "baseline",
  section_names = "title"
)
expect_equal(nrow(title_only), 0L)
deleted <- ducksemantics_literature_retrieve(
  articles,
  sections,
  snapshots,
  "seizure",
  "pubmed",
  "delete"
)
expect_equal(nrow(deleted), 0L)
expect_equal(names(deleted), names(hits))

legacy_sections <- sections[,
  !names(sections) %in% c("subsection", "source_entity_ordinal")
]
legacy_hits <- ducksemantics_literature_retrieve(
  articles,
  legacy_sections,
  snapshots,
  "seizure",
  "pubmed",
  "baseline"
)
expect_equal(nrow(legacy_hits), 4L)
expect_true(all(is.na(legacy_hits$subsection)))
expect_true(all(is.na(legacy_hits$source_entity_ordinal)))
expect_error(
  ducksemantics_literature_retrieve(
    articles,
    transform(sections, subsection = 1),
    snapshots,
    "seizure",
    "pubmed",
    "baseline"
  ),
  "subsection.*character"
)
expect_error(
  ducksemantics_literature_retrieve(
    articles,
    transform(sections, source_entity_ordinal = 1.5),
    snapshots,
    "seizure",
    "pubmed",
    "baseline"
  ),
  "source_entity_ordinal.*finite integer"
)
expect_error(
  ducksemantics_literature_retrieve(
    articles,
    transform(
      sections,
      source_entity_ordinal = as.character(source_entity_ordinal)
    ),
    snapshots,
    "seizure",
    "pubmed",
    "baseline"
  ),
  "source_entity_ordinal.*finite integer"
)

if (requireNamespace("duckdb", quietly = TRUE)) {
  con <- DBI::dbConnect(duckdb::duckdb(shared_home = FALSE))
  tryCatch(
    {
      DBI::dbWriteTable(con, "articles", articles)
      DBI::dbWriteTable(con, "sections", sections)
      DBI::dbWriteTable(con, "snapshots", snapshots)
      tables <- sort(DBI::dbListTables(con))
      db_hits <- ducksemantics_literature_retrieve(
        "articles",
        "sections",
        "snapshots",
        "seizure",
        "pubmed",
        "baseline",
        conn = con
      )
      expect_equal(db_hits, hits)
      expect_equal(sort(DBI::dbListTables(con)), tables)
    },
    finally = DBI::dbDisconnect(con, shutdown = TRUE)
  )
}
