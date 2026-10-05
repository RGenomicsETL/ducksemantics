# DuckSemantics: CPU-only DuckDB extension in C over the stable C API (v1.2.0),
# so one artifact loads in every DuckDB release that supports that ABI.
EXTENSION     = ducksemantics
VERSION       = $(shell sed -n 's/^version:[[:space:]]*//p' description.yml)
ABI_VERSION   = v1.2.0
DUCKDB       ?= duckdb
PLATFORM     ?= $(shell $(DUCKDB) -noheader -list -c 'PRAGMA platform' 2>/dev/null || echo linux_amd64)
PYTHON       ?= python3

CC           ?= cc
# -ffp-contract=off keeps float reductions bit-identical across ISAs.
CFLAGS       ?= -O3
CFLAGS       += -std=c11 -fPIC -fvisibility=hidden -ffp-contract=off \
                -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
CPPFLAGS     += -D_POSIX_C_SOURCE=200809L -Iduckdb_capi -DDUCKDB_EXTENSION_NAME=$(EXTENSION) \
                -DDUCKDB_EXTENSION_API_VERSION_MAJOR=1 -DDUCKDB_EXTENSION_API_VERSION_MINOR=2 \
                -DDUCKDB_EXTENSION_API_VERSION_PATCH=0
LDLIBS       += -lm

SOURCES = src/ducksemantics_extension.c src/gguf.c src/tokenizer.c src/tokenizer_sql.c
HEADERS = src/gguf.h src/tokenizer.h src/tokenizer_unicode.h duckdb_capi/duckdb_extension.h duckdb_capi/duckdb.h
LIBRARY = build/lib$(EXTENSION).so
ARTIFACT = build/$(EXTENSION).duckdb_extension

.PHONY: all test fixtures clean
all: $(ARTIFACT)

$(LIBRARY): $(SOURCES) $(HEADERS) Makefile
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -shared -o $@ $(SOURCES) $(LDLIBS)

$(ARTIFACT): $(LIBRARY) scripts/append_metadata.py description.yml
	$(PYTHON) scripts/append_metadata.py --library $< --output $@ \
	  --platform $(PLATFORM) --duckdb-version $(ABI_VERSION) --extension-version $(VERSION)

build/make_tokenizer_fixtures: test/make_tokenizer_fixtures.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $<

build/tokenizer_test: test/tokenizer_test.c src/tokenizer.c src/tokenizer.h src/tokenizer_unicode.h src/gguf.c src/gguf.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ test/tokenizer_test.c src/tokenizer.c src/gguf.c $(LDLIBS)

fixtures: build/make_tokenizer_fixtures build/tokenizer_test
	$(PYTHON) scripts/make_fixtures.py build/fixtures
	build/make_tokenizer_fixtures build/fixtures

test: $(ARTIFACT) fixtures
	DUCKDB=$(DUCKDB) sh test/run.sh $(ARTIFACT)

clean:
	rm -rf build
