# Package index

## Ontology Graphs

- [`ducksemantics_connect()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_connect.md)
  : Connect to a DuckDB semantic store
- [`ducksemantics_init()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_init.md)
  : Initialize semantic graph tables
- [`ducksemantics_schema_sql()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_schema_sql.md)
  : DuckDB semantic graph schema
- [`ducksemantics_tables()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_tables.md)
  : Semantic table names
- [`ducksemantics_read_obo()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_read_obo.md)
  : Read an OBO ontology into graph relations
- [`ducksemantics_write_obo()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_write_obo.md)
  : Write an OBO ontology into the graph
- [`ducksemantics_write_graph()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_write_graph.md)
  : Write ontology graph rows
- [`ducksemantics_projection_sql()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_projection_sql.md)
  : Project edge-shaped data to graph SQL
- [`ducksemantics_closure_sql()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_closure_sql.md)
  : Materialize transitive closure SQL
- [`ducksemantics_normalize()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_normalize.md)
  : Normalize semantic text
- [`ducksemantics_tokens()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_tokens.md)
  : Tokenize source text
- [`ducksemantics_index_aliases()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_index_aliases.md)
  : Build the lexical alias index
- [`ducksemantics_annotate()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_annotate.md)
  : Generate lexical HPO/ontology candidates

## HPO Observations

- [`ducksemantics_hpo_observation_contract()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_hpo_observation_contract.md)
  : HPO observation contract
- [`ducksemantics_hpo_observations()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_hpo_observations.md)
  : Validate accepted HPO observations against source documents

## Monarch and Literature Relations

- [`ducksemantics_monarch_release_catalog()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_monarch_release_catalog.md)
  : Validate a typed Monarch release catalog
- [`ducksemantics_monarch_project_pack()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_monarch_project_pack.md)
  : Project one attached, dated Monarch annotation pack
- [`ducksemantics_monarch_import()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_monarch_import.md)
  : Import caller-supplied Monarch relations
- [`ducksemantics_monarch_query()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_monarch_query.md)
  : Select one cataloged Monarch release
- [`ducksemantics_monarch_gene_disease_holdout_audit()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_monarch_gene_disease_holdout_audit.md)
  : Audit a historical Monarch gene-disease holdout
- [`ducksemantics_literature_snapshot_catalog()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_literature_snapshot_catalog.md)
  : Validate an append-only literature snapshot catalog
- [`ducksemantics_literature_retrieve()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_literature_retrieve.md)
  : Retrieve exact literature spans from append-only caller-owned
  relations

## Semantic Retrieval

- [`ducksemantics_embedding_batch()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_embedding_batch.md)
  : Construct embedding rows
- [`ducksemantics_write_embeddings()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_write_embeddings.md)
  : Store embedding rows in DuckDB
- [`ducksemantics_embedding_search()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_embedding_search.md)
  : Search dense embeddings
- [`ducksemantics_materialize_embedding_index()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_materialize_embedding_index.md)
  : Materialize a fixed-dimension embedding table
- [`ducksemantics_token_embedding_batch()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_token_embedding_batch.md)
  : Construct token embedding rows
- [`ducksemantics_token_embedding_batch_from_provider()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_token_embedding_batch_from_provider.md)
  : Construct token embedding rows from a provider
- [`ducksemantics_write_token_embeddings()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_write_token_embeddings.md)
  : Store token embedding rows in DuckDB
- [`ducksemantics_late_interaction_search()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_late_interaction_search.md)
  : Exact token-level late-interaction search

## Provider Protocols and Judgments

- [`DucksemanticsAnnotator`](https://RGenomicsETL.github.io/ducksemantics/reference/DucksemanticsAnnotator.md)
  : Grounding-provider protocol
- [`DucksemanticsPromptRunner`](https://RGenomicsETL.github.io/ducksemantics/reference/DucksemanticsPromptRunner.md)
  : Prompt-runner provider protocol
- [`DucksemanticsJudgmentParser`](https://RGenomicsETL.github.io/ducksemantics/reference/DucksemanticsJudgmentParser.md)
  : Judgment-parser provider protocol
- [`DucksemanticsEmbeddingProvider`](https://RGenomicsETL.github.io/ducksemantics/reference/DucksemanticsEmbeddingProvider.md)
  : Embedding-provider protocol
- [`DucksemanticsTokenEmbeddingProvider`](https://RGenomicsETL.github.io/ducksemantics/reference/DucksemanticsTokenEmbeddingProvider.md)
  : Token-embedding-provider protocol
- [`ducksemantics_run()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_provider_generics.md)
  [`ducksemantics_embed()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_provider_generics.md)
  [`ducksemantics_token_embed()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_provider_generics.md)
  [`ducksemantics_parse()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_provider_generics.md)
  [`ducksemantics_ground()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_provider_generics.md)
  : Provider protocol generics
- [`ducksemantics_lexical_annotator()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_lexical_annotator.md)
  : Create the default lexical grounding provider
- [`ducksemantics_prompt_runner()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_prompt_runner.md)
  : Wrap a prompt function as a prompt provider
- [`ducksemantics_embedding_provider()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_embedding_provider.md)
  : Wrap an embedding function as an embedding provider
- [`ducksemantics_token_embedding_provider()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_token_embedding_provider.md)
  : Wrap a token embedding function as a token provider
- [`ducksemantics_json_judgment_parser()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_json_judgment_parser.md)
  : Create a JSON judgment parser
- [`ducksemantics_bebel_runner()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_bebel_runner.md)
  : Create a BebeLM prompt provider
- [`ducksemantics_embeddinggemma_provider()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_embeddinggemma_provider.md)
  : Create an EmbeddingGemma provider
- [`ducksemantics_colbert_provider()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_colbert_provider.md)
  : Create a native ColBERT provider
- [`ducksemantics_bebel_tool_judgment_parser()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_bebel_tool_judgment_parser.md)
  : Create a BebeLM tool-call judgment parser
- [`ducksemantics_default_judgment_instructions()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_default_judgment_instructions.md)
  : Default source-grounded judgment instructions
- [`ducksemantics_judgment_prompt()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_judgment_prompt.md)
  : Build a source-grounded judgment prompt
- [`ducksemantics_judge()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_judge.md)
  : Judge lexical candidates with a provider
- [`ducksemantics_record_judgments()`](https://RGenomicsETL.github.io/ducksemantics/reference/ducksemantics_record_judgments.md)
  : Record source-grounded judgments
