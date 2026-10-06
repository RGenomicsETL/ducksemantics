/* EmbeddingGemma-300M CPU encoder. SPDX-License-Identifier: MIT */
#ifndef DUCKSEMANTICS_EMBEDDINGGEMMA_H
#define DUCKSEMANTICS_EMBEDDINGGEMMA_H

#include "quant.h"
#include "tokenizer.h"

#define GEMMA_CONTEXT 2048
#define GEMMA_DIM 768
#define GEMMA_TASK_COUNT 10

typedef struct gemma_model gemma_model;
typedef struct gemma_workspace gemma_workspace;
typedef struct gemma_sequence {
    uint32_t first, count;
} gemma_sequence;

/* Models are immutable borrowed cache entries, retained until process exit. */
bool gemma_model_cached(const char *path, const gemma_model **out, char *error);
/* Task strings are length-delimited; -1 means unsupported. */
int gemma_task(const char *text, size_t len);
bool gemma_dimensions(int dimensions);
/* Owns *ids on success. Truncates to 2048, preserving BOS/EOS. */
bool gemma_tokenize(const gemma_model *model, const char *text, size_t len,
                    int task, int32_t **ids, size_t *count, char *error);
/* Caller-owned scratch, bounded to GEMMA_CONTEXT packed tokens. */
gemma_workspace *gemma_workspace_new(size_t capacity, char *error);
void gemma_workspace_free(gemma_workspace *work);
/* Sequences partition ids in order; each has independent attention/positions.
 * out contains n_sequences unnormalized 768-vectors, in sequence order.
 */
bool gemma_forward(const gemma_model *model, gemma_workspace *work,
                   const int32_t *ids, size_t count,
                   const gemma_sequence *sequences, size_t n_sequences,
                   float *out, char *error);
void gemma_select(const float *full, int dimensions, bool normalize, float *out);

#endif
