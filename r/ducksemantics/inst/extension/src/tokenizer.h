/* GGUF-driven LFM2 and EmbeddingGemma tokenizers. SPDX-License-Identifier: MIT */
#ifndef DUCKSEMANTICS_TOKENIZER_H
#define DUCKSEMANTICS_TOKENIZER_H

#include "gguf.h"

#include <stddef.h>

typedef struct tokenizer tokenizer;

/* The validated GGUF mapping must outlive this immutable object. */
tokenizer *tokenizer_create(const gguf_file *file, char error[GGUF_ERROR_SIZE]);
void tokenizer_destroy(tokenizer *t);
bool tokenizer_encode(const tokenizer *t, const char *text, size_t len,
                      bool add_special, int32_t **ids, size_t *count,
                      char error[GGUF_ERROR_SIZE]);
bool tokenizer_decode(const tokenizer *t, const int32_t *ids, size_t count,
                      char **text, size_t *len, char error[GGUF_ERROR_SIZE]);

/* Borrowed for the process lifetime; the cache owns the GGUF mapping. */
const tokenizer *tokenizer_cached(const char *path, char error[GGUF_ERROR_SIZE]);

#endif
