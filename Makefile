# DuckSemantics: CPU-only DuckDB extension in C over the stable C API (v1.2.0),
# so one artifact loads in every DuckDB release that supports that ABI.
# Building needs only a C compiler and a POSIX shell.
EXTENSION     = ducksemantics
VERSION       = $(shell sed -n 's/^version:[[:space:]]*//p' description.yml)
ABI_VERSION   = v1.2.0
DUCKDB       ?= duckdb
PLATFORM     ?= $(shell $(DUCKDB) -noheader -list -c 'PRAGMA platform' 2>/dev/null || echo linux_amd64)

CC           ?= cc
# -ffp-contract=off keeps float reductions bit-identical across ISAs.
CFLAGS       ?= -O3
CFLAGS       += -std=c11 -fPIC -fvisibility=hidden -ffp-contract=off \
                -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
CPPFLAGS     += -D_POSIX_C_SOURCE=200809L -Iduckdb_capi -DDUCKDB_EXTENSION_NAME=$(EXTENSION) \
                -DDUCKDB_EXTENSION_API_VERSION_MAJOR=1 -DDUCKDB_EXTENSION_API_VERSION_MINOR=2 \
                -DDUCKDB_EXTENSION_API_VERSION_PATCH=0
LDLIBS       += -lm
SANITIZE      = -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined

SOURCES = src/ducksemantics_extension.c src/gguf.c src/quant.c src/quant_sql.c \
          src/tokenizer.c src/tokenizer_sql.c src/lfm2.c src/lfm2_sql.c
HEADERS = src/gguf.h src/quant.h src/tokenizer.h src/lfm2.h src/tokenizer_unicode.h \
          duckdb_capi/duckdb_extension.h duckdb_capi/duckdb.h
LIBRARY = build/lib$(EXTENSION).so
ARTIFACT = build/$(EXTENSION).duckdb_extension
ASAN_ARTIFACT = build/asan/$(EXTENSION).duckdb_extension
# Real model files for `make audit` (Rscript with duckdb and Rgguf installed).
GGUF_FILES ?= $(wildcard /root/bebelm/*.gguf)

.PHONY: all test fixtures sanitize fuzz audit quant-oracle clean
all: $(ARTIFACT)

$(LIBRARY): $(SOURCES) $(HEADERS) Makefile
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -shared -o $@ $(SOURCES) $(LDLIBS)

$(ARTIFACT): $(LIBRARY) scripts/append_metadata.sh description.yml
	sh scripts/append_metadata.sh $< $@ $(PLATFORM) $(ABI_VERSION) $(VERSION)

build/make_fixtures: test/make_fixtures.c
	@mkdir -p build
	$(CC) -std=c11 -Wall -Wextra -Wpedantic -O1 -o $@ $<

build/make_quant_fixtures: test/make_quant_fixtures.c src/quant.c src/gguf.c src/quant.h src/gguf.h Makefile
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ test/make_quant_fixtures.c src/quant.c src/gguf.c $(LDLIBS)

build/make_tokenizer_fixtures: test/make_tokenizer_fixtures.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $<

build/tokenizer_test: test/tokenizer_test.c src/tokenizer.c src/tokenizer.h src/tokenizer_unicode.h src/gguf.c src/gguf.h
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ test/tokenizer_test.c src/tokenizer.c src/gguf.c $(LDLIBS)

fixtures: build/make_fixtures build/make_quant_fixtures build/make_tokenizer_fixtures build/tokenizer_test
	@mkdir -p build/fixtures
	build/make_fixtures build/fixtures
	build/make_quant_fixtures build/fixtures
	build/make_tokenizer_fixtures build/fixtures

build/test_quant: test/quant.c src/quant.c src/gguf.c src/quant.h src/gguf.h Makefile
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ test/quant.c src/quant.c src/gguf.c $(LDLIBS)

build/bench_quant: scripts/bench_quant.c src/quant.c src/gguf.c src/quant.h src/gguf.h Makefile
	@mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ scripts/bench_quant.c src/quant.c src/gguf.c $(LDLIBS)

build/lfm2_test: test/lfm2_test.c src/lfm2.c src/quant.c src/tokenizer.c src/gguf.c $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ test/lfm2_test.c src/lfm2.c src/quant.c src/tokenizer.c src/gguf.c $(LDLIBS) -lpthread

build/lfm2_oracle: test/lfm2_oracle.c src/lfm2.c src/quant.c src/tokenizer.c src/gguf.c $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ test/lfm2_oracle.c src/lfm2.c src/quant.c src/tokenizer.c src/gguf.c $(LDLIBS) -lpthread

build/asan/lfm2_test: test/lfm2_test.c src/lfm2.c src/quant.c src/tokenizer.c src/gguf.c $(HEADERS)
	@mkdir -p build/asan
	$(CC) $(CPPFLAGS) $(SANITIZE) -std=c11 -ffp-contract=off -o $@ test/lfm2_test.c src/lfm2.c src/quant.c src/tokenizer.c src/gguf.c $(LDLIBS) -lpthread

test: $(ARTIFACT) fixtures build/test_quant build/lfm2_test
	build/test_quant
	build/lfm2_test
	DUCKDB=$(DUCKDB) sh test/run.sh $(ARTIFACT)

# The same suite against an ASan/UBSan build, run inside the DuckDB CLI.
$(ASAN_ARTIFACT): $(SOURCES) $(HEADERS) Makefile
	@mkdir -p build/asan
	$(CC) $(CPPFLAGS) $(SANITIZE) -std=c11 -fPIC -fvisibility=hidden -shared \
	  -o build/asan/lib$(EXTENSION).so $(SOURCES) $(LDLIBS)
	sh scripts/append_metadata.sh build/asan/lib$(EXTENSION).so $@ $(PLATFORM) $(ABI_VERSION) $(VERSION)
	printf '#!/bin/sh\nLD_PRELOAD="%s:%s" ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 exec %s "$$@"\n' \
	  "$$($(CC) -print-file-name=libasan.so)" "$$($(CC) -print-file-name=libubsan.so)" "$$(command -v $(DUCKDB))" \
	  > build/asan/duckdb && chmod +x build/asan/duckdb

build/asan/tokenizer_test: test/tokenizer_test.c src/tokenizer.c src/tokenizer.h src/tokenizer_unicode.h src/gguf.c src/gguf.h
	@mkdir -p build/asan
	$(CC) $(CPPFLAGS) $(SANITIZE) -std=c11 -o $@ test/tokenizer_test.c src/tokenizer.c src/gguf.c $(LDLIBS) -lpthread

sanitize: $(ASAN_ARTIFACT) build/asan/tokenizer_test build/asan/lfm2_test fixtures
	ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 build/asan/lfm2_test
	TOKENIZER_TEST=build/asan/tokenizer_test DUCKDB=build/asan/duckdb sh test/run.sh $(ASAN_ARTIFACT)

build/asan/fuzz_gguf: test/fuzz_gguf.c src/gguf.c src/gguf.h
	@mkdir -p build/asan
	$(CC) $(CPPFLAGS) $(SANITIZE) -std=c11 -o $@ test/fuzz_gguf.c src/gguf.c

fuzz: build/asan/fuzz_gguf fixtures
	build/asan/fuzz_gguf 20000 build/fixtures/valid.gguf build/fixtures/many.gguf

audit: $(ARTIFACT)
	Rscript scripts/audit_gguf.R $(ARTIFACT) $(GGUF_FILES)

# Developer oracle only: compiles ggml's reference quant code from an external
# source tree (Rfmalloc's pinned vendored ggml) next to ours.
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
