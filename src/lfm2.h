/* SPDX-License-Identifier: MIT */
#ifndef DUCKSEMANTICS_LFM2_H
#define DUCKSEMANTICS_LFM2_H

#include "gguf.h"
#include <stddef.h>
#include <stdint.h>

#define LFM2_MAX_INPUT_BYTES (1024 * 1024)
#define LFM2_COLBERT_DIM 128
#define LFM2_COLBERT_MAX_TOKENS 512

typedef struct lfm2_model lfm2_model;

typedef struct lfm2_embeddings {
    float *values;
    uint32_t *ids;
    size_t count;
} lfm2_embeddings;

/* Immutable, borrowed process-lifetime handle. Scratch belongs to each call. */
const lfm2_model *lfm2_model_get(const char *path, char *error);
bool lfm2_colbert_encode(const lfm2_model *model, const char *text, size_t bytes,
                         bool query, lfm2_embeddings *out, char *error);
void lfm2_embeddings_free(lfm2_embeddings *out);

void lfm2_rmsnorm(const float *x, const float *gain, size_t n, float epsilon, float *out);
float lfm2_dot(const float *a, const float *b, size_t n);
void lfm2_rope(float *x, size_t n, size_t position, float theta);
void lfm2_conv_centered(const float *bx, const float *c, const float *weights,
                        size_t tokens, size_t hidden, size_t width, float *out);
void lfm2_conv_step(const float *bx, const float *c, const float *weights,
                    size_t hidden, size_t width, float *state, float *out);
void lfm2_attention(const float *q, const float *k, const float *v,
                    size_t tokens, size_t heads, size_t kv_heads, size_t dim,
                    float *scores, float *out);

#endif
