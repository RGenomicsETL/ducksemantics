/* SPDX-License-Identifier: MIT */
#ifndef DUCKSEMANTICS_LFM2_H
#define DUCKSEMANTICS_LFM2_H

#include "gguf.h"
#include <stddef.h>
#include <stdint.h>

#define LFM2_MAX_INPUT_BYTES (1024 * 1024)
#define LFM2_COLBERT_DIM 128
#define LFM2_COLBERT_MAX_TOKENS 512
#define LFM2_MAX_CONTEXT 4096
#define LFM2_MAX_GENERATED 1024
#define LFM2_MOE_LAYERS 24
#define LFM2_MOE_HIDDEN 2048
#define LFM2_MOE_KV_DIM 512
#define LFM2_MOE_VOCAB 128000

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

/* Generator profile. Initialize/fork into an empty destination; free once. */
typedef struct lfm2_cache {
    size_t position, capacity;
    float *k[LFM2_MOE_LAYERS], *v[LFM2_MOE_LAYERS], *conv[LFM2_MOE_LAYERS];
    float norm[LFM2_MOE_HIDDEN];
} lfm2_cache;
bool lfm2_cache_init(lfm2_cache *cache, size_t capacity, char *error);
bool lfm2_cache_fork(lfm2_cache *out, const lfm2_cache *source, char *error);
void lfm2_cache_free(lfm2_cache *cache);

typedef struct lfm2_state lfm2_state;
typedef struct lfm2_generation {
    int32_t *ids;
    float *margins;
    char *text;
    size_t count, bytes, prompt_tokens;
    double prefill_seconds, decode_seconds;
    int32_t terminal_id;
    float terminal_margin;
    bool stopped;
} lfm2_generation;
lfm2_state *lfm2_state_new(const lfm2_model *model, size_t capacity, char *error);
lfm2_state *lfm2_state_fork(const lfm2_state *source, char *error);
void lfm2_state_free(lfm2_state *state);
bool lfm2_prefill(lfm2_state *state, const int32_t *ids, size_t count, char *error);
bool lfm2_prompt_ids(const lfm2_model *model, const char *text, size_t bytes, bool bos,
                     int32_t **ids, size_t *count, char *error);
bool lfm2_state_logits(lfm2_state *state, const int32_t *ids, size_t count, float *out, char *error);
/* The final emitted id is pending, as in bebelm; prefill it before continuing. */
bool lfm2_decode(lfm2_state *state, size_t max_tokens, lfm2_generation *out, char *error);
bool lfm2_generate(const lfm2_model *model, const char *prompt, size_t bytes,
                   size_t max_tokens, lfm2_generation *out, char *error);
void lfm2_generation_free(lfm2_generation *out);

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
