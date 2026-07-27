# Architecture

**Current.** `ducksemantics` owns ontology graphs, HPO extraction observations,
release-cataloged Monarch gene/phenotype/disease relations, semantic and
literature retrieval, source-grounded judgments, and real model/provider
protocols. Bulk values are data frames or caller-owned DuckDB relations; S7 is
reserved for prompt, embedding, parser, and annotator provider protocols.

`RClinVarbitration` owns ClinVar and append-only PubMed/Europe PMC source
relations; ducksemantics consumes their caller-owned source-order literature
projection without writing a copy.
`VariantStoryBench` owns all evaluation and benchmark APIs.

The semantic authority is the documented relational contracts and their
validators: lexical HPO candidates are not observations; observations validate
their exact source spans; Monarch facts retain provider/release identity through
a typed catalog; and literature retrieval binds a provider plus exact cataloged
source-order cutoff, selects each article's latest event, drops deletion events,
and joins only that event's sections.
