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

SOURCES = src/ducksemantics_extension.c src/gguf.c src/quant.c src/quant_sql.c
HEADERS = src/gguf.h src/quant.h duckdb_capi/duckdb_extension.h duckdb_capi/duckdb.h
LIBRARY = build/lib$(EXTENSION).so
ARTIFACT = build/$(EXTENSION).duckdb_extension

.PHONY: all test fixtures quant-oracle clean
all: $(ARTIFACT)

$(LIBRARY): $(SOURCES) $(HEADERS) Makefile
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -shared -o $@ $(SOURCES) $(LDLIBS)

$(ARTIFACT): $(LIBRARY) scripts/append_metadata.py description.yml
	$(PYTHON) scripts/append_metadata.py --library $< --output $@ \
	  --platform $(PLATFORM) --duckdb-version $(ABI_VERSION) --extension-version $(VERSION)

build/make_quant_fixtures: test/make_quant_fixtures.c src/quant.c src/gguf.c src/quant.h src/gguf.h Makefile
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ test/make_quant_fixtures.c src/quant.c src/gguf.c $(LDLIBS)

fixtures: build/make_quant_fixtures
	$(PYTHON) scripts/make_fixtures.py build/fixtures
	build/make_quant_fixtures build/fixtures

build/bench_quant: scripts/bench_quant.c src/quant.c src/gguf.c src/quant.h src/gguf.h Makefile
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ scripts/bench_quant.c src/quant.c src/gguf.c $(LDLIBS)

build/test_quant: test/quant.c src/quant.c src/gguf.c src/quant.h src/gguf.h Makefile
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ test/quant.c src/quant.c src/gguf.c $(LDLIBS)

test: $(ARTIFACT) fixtures build/test_quant
	build/test_quant
	DUCKDB=$(DUCKDB) sh test/run.sh $(ARTIFACT)

# Developer oracle only; GGML is read from this external source tree.
GGML_DIR ?= /root/Rfmalloc/packages/Rggml/inst/ggml
ORACLE_FLAGS = -D_POSIX_C_SOURCE=200809L -O2 -std=c11 -ffp-contract=off -fPIC \
               -fvisibility=hidden -ffunction-sections -fdata-sections \
               -I$(GGML_DIR) -I$(GGML_DIR)/../include
ORACLE_SOURCES = test/ggml_oracle.c src/quant.c src/gguf.c $(GGML_DIR)/ggml-quants.c

build/oracle/ggml.so: $(ORACLE_SOURCES) src/quant.h src/gguf.h Makefile
	@mkdir -p build/oracle
	$(CC) $(ORACLE_FLAGS) -shared -Wl,--gc-sections -Wl,-z,defs -o $@ $(ORACLE_SOURCES) -lm

build/oracle/test_ggml: $(ORACLE_SOURCES) src/quant.h src/gguf.h Makefile
	@mkdir -p build/oracle
	$(CC) $(ORACLE_FLAGS) -Wl,--gc-sections -o $@ $(ORACLE_SOURCES) -lm

quant-oracle: build/oracle/ggml.so build/oracle/test_ggml
	build/oracle/test_ggml

clean:
	rm -rf build
