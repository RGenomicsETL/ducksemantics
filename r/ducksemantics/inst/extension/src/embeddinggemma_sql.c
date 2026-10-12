/* DuckDB adapter for EmbeddingGemma. SPDX-License-Identifier: MIT */
#include "duckdb_extension.h"
DUCKDB_EXTENSION_EXTERN

#include "embeddinggemma.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct embed_batch {
    const gemma_model *model;
    int32_t ids[GEMMA_CONTEXT];
    gemma_sequence sequences[GEMMA_CONTEXT];
    size_t tokens, count;
    float *full;
    gemma_workspace *work;
} embed_batch;

typedef struct embed_row {
    int task, dimensions;
    bool normalize;
    size_t sequence;
} embed_row;

static bool valid(duckdb_vector v, idx_t row) {
    uint64_t *mask = duckdb_vector_get_validity(v);
    return !mask || duckdb_validity_row_is_valid(mask, row);
}

static bool flush(embed_batch *b, embed_row *rows, idx_t first, idx_t end,
                  const duckdb_list_entry *entries, float *child, char *error) {
    if (!b->count) return true;
    if (!gemma_forward(b->model, b->work, b->ids, b->tokens, b->sequences, b->count, b->full, error)) return false;
    for (idx_t r = first; r < end; r++) {
        if (rows[r].sequence == SIZE_MAX) continue;
        gemma_select(b->full + rows[r].sequence * GEMMA_DIM, rows[r].dimensions,
                     rows[r].normalize, child + entries[r].offset);
    }
    b->count = 0;
    b->tokens = 0;
    return true;
}

static void embed_sql(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    idx_t n = duckdb_data_chunk_get_size(input), used = 0;
    duckdb_vector vectors[5];
    for (unsigned i = 0; i < 5; i++) vectors[i] = duckdb_data_chunk_get_vector(input, i);
    duckdb_string_t *paths = duckdb_vector_get_data(vectors[0]);
    duckdb_string_t *texts = duckdb_vector_get_data(vectors[1]);
    duckdb_string_t *tasks = duckdb_vector_get_data(vectors[2]);
    int32_t *dimensions = duckdb_vector_get_data(vectors[3]);
    bool *normalize = duckdb_vector_get_data(vectors[4]);
    duckdb_list_entry *entries = duckdb_vector_get_data(output);
    duckdb_vector_ensure_validity_writable(output);
    uint64_t *mask = duckdb_vector_get_validity(output);
    char error[GGUF_ERROR_SIZE] = {0};
    embed_row *rows = calloc(n ? n : 1, sizeof(*rows));
    embed_batch *b = NULL;
    if (!rows) { duckdb_scalar_function_set_error(info, "out of memory allocating embedding rows"); return; }
    for (idx_t r = 0; r < n; r++) {
        rows[r].sequence = SIZE_MAX;
        bool null = false;
        for (unsigned i = 0; i < 5; i++) if (!valid(vectors[i], r)) null = true;
        entries[r] = (duckdb_list_entry){used, 0};
        if (null) { duckdb_validity_set_row_invalid(mask, r); continue; }
        rows[r].task = gemma_task(duckdb_string_t_data(&tasks[r]), duckdb_string_t_length(tasks[r]));
        if (rows[r].task < 0) { snprintf(error, sizeof(error), "unknown EmbeddingGemma task"); goto done; }
        if (!gemma_dimensions(dimensions[r])) {
            snprintf(error, sizeof(error), "unsupported embedding dimension; use 768, 512, 256, or 128"); goto done;
        }
        rows[r].dimensions = dimensions[r];
        rows[r].normalize = normalize[r];
        if (used > UINT64_MAX - (idx_t)dimensions[r]) {
            snprintf(error, sizeof(error), "embedding list size overflow"); goto done;
        }
        entries[r].length = (idx_t)dimensions[r];
        used += (idx_t)dimensions[r];
    }
    if (duckdb_list_vector_reserve(output, used) != DuckDBSuccess ||
        duckdb_list_vector_set_size(output, used) != DuckDBSuccess) {
        snprintf(error, sizeof(error), "cannot allocate embedding output list"); goto done;
    }
    if (!used) goto done;
    float *child = duckdb_vector_get_data(duckdb_list_vector_get_child(output));
    b = calloc(1, sizeof(*b));
    if (!b) { snprintf(error, sizeof(error), "out of memory allocating embedding batch"); goto done; }
    b->work = gemma_workspace_new(GEMMA_CONTEXT, error);
    b->full = malloc(GEMMA_CONTEXT * GEMMA_DIM * sizeof(float));
    if (!b->work) goto done;
    if (!b->full) { snprintf(error, sizeof(error), "out of memory allocating embedding projection buffer"); goto done; }
    idx_t first = 0;
    for (idx_t r = 0; r < n; r++) {
        if (!entries[r].length) continue;
        uint32_t len = duckdb_string_t_length(paths[r]);
        const char *data = duckdb_string_t_data(&paths[r]);
        if (memchr(data, '\0', len)) { snprintf(error, sizeof(error), "EmbeddingGemma path contains NUL"); goto done; }
        char *path = malloc((size_t)len + 1);
        if (!path) { snprintf(error, sizeof(error), "out of memory copying embedding path"); goto done; }
        memcpy(path, data, len); path[len] = '\0';
        const gemma_model *m;
        bool ok = gemma_model_cached(path, &m, error);
        free(path);
        if (!ok) goto done;
        int32_t *ids;
        size_t count;
        if (!gemma_tokenize(m, duckdb_string_t_data(&texts[r]), duckdb_string_t_length(texts[r]),
                            rows[r].task, &ids, &count, error)) goto done;
        /* Consecutive identical tokenized inputs share the full projection. */
        if (b->count && b->model == m && b->sequences[b->count - 1].count == count &&
            !memcmp(ids, b->ids + b->sequences[b->count - 1].first, count * sizeof(*ids))) {
            rows[r].sequence = b->count - 1;
            free(ids);
            continue;
        }
        if (b->count && (b->model != m || count > GEMMA_CONTEXT - b->tokens)) {
            ok = flush(b, rows, first, r, entries, child, error);
            if (!ok) { free(ids); goto done; }
            first = r;
        }
        b->model = m;
        rows[r].sequence = b->count;
        b->sequences[b->count++] = (gemma_sequence){(uint32_t)b->tokens, (uint32_t)count};
        memcpy(b->ids + b->tokens, ids, count * sizeof(*ids));
        b->tokens += count;
        free(ids);
    }
    flush(b, rows, first, n, entries, child, error);
done:
    if (error[0]) duckdb_scalar_function_set_error(info, error);
    if (b) { gemma_workspace_free(b->work); free(b->full); free(b); }
    free(rows);
}

duckdb_state semantic_register_embeddinggemma(duckdb_connection con) {
    duckdb_logical_type string = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_logical_type integer = duckdb_create_logical_type(DUCKDB_TYPE_INTEGER);
    duckdb_logical_type boolean = duckdb_create_logical_type(DUCKDB_TYPE_BOOLEAN);
    duckdb_logical_type real = duckdb_create_logical_type(DUCKDB_TYPE_FLOAT);
    duckdb_logical_type list = duckdb_create_list_type(real);
    duckdb_scalar_function f = duckdb_create_scalar_function();
    duckdb_scalar_function_set_name(f, "semantic_embed");
    for (unsigned i = 0; i < 3; i++) duckdb_scalar_function_add_parameter(f, string);
    duckdb_scalar_function_add_parameter(f, integer);
    duckdb_scalar_function_add_parameter(f, boolean);
    duckdb_scalar_function_set_return_type(f, list);
    duckdb_scalar_function_set_function(f, embed_sql);
    duckdb_scalar_function_set_volatile(f);
    duckdb_state status = duckdb_register_scalar_function(con, f);
    duckdb_destroy_scalar_function(&f);
    duckdb_destroy_logical_type(&list);
    duckdb_destroy_logical_type(&real);
    duckdb_destroy_logical_type(&boolean);
    duckdb_destroy_logical_type(&integer);
    duckdb_destroy_logical_type(&string);
    return status;
}
