/* SPDX-License-Identifier: MIT */
#include "duckdb_extension.h"
#include "lfm2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DUCKDB_EXTENSION_EXTERN

static void set_null(duckdb_vector output, idx_t row) {
    duckdb_vector_ensure_validity_writable(output);
    duckdb_validity_set_row_invalid(duckdb_vector_get_validity(output), row);
}

static char *path_copy(duckdb_string_t value, char *error) {
    size_t n = duckdb_string_t_length(value);
    const char *p = duckdb_string_t_data(&value);
    if (!n || n > 4096 || memchr(p, 0, n)) {
        snprintf(error, GGUF_ERROR_SIZE, "invalid LFM2 model path");
        return NULL;
    }
    char *path = malloc(n + 1);
    if (!path) { snprintf(error, GGUF_ERROR_SIZE, "out of memory for LFM2 path"); return NULL; }
    memcpy(path, p, n); path[n] = 0;
    return path;
}

static void colbert_encode(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector args[3];
    duckdb_string_t *strings[3];
    uint64_t *valid[3];
    for (unsigned j = 0; j < 3; j++) {
        args[j] = duckdb_data_chunk_get_vector(input, j);
        strings[j] = duckdb_vector_get_data(args[j]);
        valid[j] = duckdb_vector_get_validity(args[j]);
    }
    duckdb_vector tokens = duckdb_list_vector_get_child(output);
    duckdb_vector values = duckdb_list_vector_get_child(tokens);
    idx_t token_count = 0, value_count = 0;
    for (idx_t row = 0; row < duckdb_data_chunk_get_size(input); row++) {
        if (!duckdb_validity_row_is_valid(valid[0], row) || !duckdb_validity_row_is_valid(valid[1], row) ||
            !duckdb_validity_row_is_valid(valid[2], row)) { set_null(output, row); continue; }
        char error[GGUF_ERROR_SIZE];
        duckdb_string_t role = strings[2][row];
        size_t role_n = duckdb_string_t_length(role);
        const char *role_p = duckdb_string_t_data(&role);
        bool query = role_n == 5 && !memcmp(role_p, "query", 5);
        if (!query && !(role_n == 8 && !memcmp(role_p, "document", 8))) {
            duckdb_scalar_function_set_error(info, "ColBERT role must be 'query' or 'document'"); return;
        }
        char *path = path_copy(strings[0][row], error);
        if (!path) { duckdb_scalar_function_set_error(info, error); return; }
        const lfm2_model *model = lfm2_model_get(path, error);
        free(path);
        if (!model) { duckdb_scalar_function_set_error(info, error); return; }
        duckdb_string_t text = strings[1][row];
        lfm2_embeddings encoded;
        if (!lfm2_colbert_encode(model, duckdb_string_t_data(&text), duckdb_string_t_length(text),
                                query, &encoded, error)) {
            duckdb_scalar_function_set_error(info, error); return;
        }
        idx_t add_values = encoded.count * LFM2_COLBERT_DIM;
        if (duckdb_list_vector_reserve(output, token_count + encoded.count) != DuckDBSuccess ||
            duckdb_list_vector_reserve(tokens, value_count + add_values) != DuckDBSuccess) {
            lfm2_embeddings_free(&encoded);
            duckdb_scalar_function_set_error(info, "out of memory for ColBERT result"); return;
        }
        duckdb_list_entry *outer = duckdb_vector_get_data(output);
        duckdb_list_entry *inner = duckdb_vector_get_data(tokens);
        float *data = duckdb_vector_get_data(values);
        outer[row] = (duckdb_list_entry){token_count, encoded.count};
        for (size_t t = 0; t < encoded.count; t++)
            inner[token_count + t] = (duckdb_list_entry){value_count + t * LFM2_COLBERT_DIM, LFM2_COLBERT_DIM};
        memcpy(data + value_count, encoded.values, add_values * sizeof(float));
        token_count += encoded.count; value_count += add_values;
        duckdb_list_vector_set_size(output, token_count);
        duckdb_list_vector_set_size(tokens, value_count);
        lfm2_embeddings_free(&encoded);
    }
}

static void generate(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector p = duckdb_data_chunk_get_vector(input, 0), t = duckdb_data_chunk_get_vector(input, 1);
    duckdb_vector n = duckdb_data_chunk_get_vector(input, 2);
    duckdb_string_t *paths = duckdb_vector_get_data(p), *texts = duckdb_vector_get_data(t);
    int32_t *limits = duckdb_vector_get_data(n);
    for (idx_t row = 0; row < duckdb_data_chunk_get_size(input); row++) {
        if (!duckdb_validity_row_is_valid(duckdb_vector_get_validity(p), row) ||
            !duckdb_validity_row_is_valid(duckdb_vector_get_validity(t), row) ||
            !duckdb_validity_row_is_valid(duckdb_vector_get_validity(n), row)) { set_null(output, row); continue; }
        if (limits[row] <= 0 || limits[row] > LFM2_MAX_GENERATED) {
            duckdb_scalar_function_set_error(info, "LFM2 generation length out of bounds"); return;
        }
        char error[GGUF_ERROR_SIZE]; char *path = path_copy(paths[row], error);
        if (!path) { duckdb_scalar_function_set_error(info, error); return; }
        const lfm2_model *m = lfm2_model_get(path, error); free(path);
        if (!m) { duckdb_scalar_function_set_error(info, error); return; }
        duckdb_string_t text = texts[row]; lfm2_generation result;
        if (!lfm2_generate(m, duckdb_string_t_data(&text), duckdb_string_t_length(text), limits[row], &result, error)) {
            duckdb_scalar_function_set_error(info, error); return;
        }
        duckdb_vector_assign_string_element_len(output, row, result.text, result.bytes);
        lfm2_generation_free(&result);
    }
}

static void next_token_logits(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector p = duckdb_data_chunk_get_vector(input, 0), t = duckdb_data_chunk_get_vector(input, 1);
    duckdb_vector options = duckdb_data_chunk_get_vector(input, 2), child = duckdb_list_vector_get_child(options);
    duckdb_string_t *paths = duckdb_vector_get_data(p), *texts = duckdb_vector_get_data(t);
    duckdb_list_entry *entries = duckdb_vector_get_data(options);
    int32_t *ids = duckdb_vector_get_data(child); idx_t total = duckdb_list_vector_get_size(options), used = 0;
    for (idx_t row = 0; row < duckdb_data_chunk_get_size(input); row++) {
        if (!duckdb_validity_row_is_valid(duckdb_vector_get_validity(p), row) ||
            !duckdb_validity_row_is_valid(duckdb_vector_get_validity(t), row) ||
            !duckdb_validity_row_is_valid(duckdb_vector_get_validity(options), row)) { set_null(output, row); continue; }
        duckdb_list_entry e = entries[row];
        if (e.offset > total || e.length > total - e.offset || e.length > LFM2_MOE_VOCAB) {
            duckdb_scalar_function_set_error(info, "LFM2 requested logits exceed vocabulary size"); return;
        }
        for (idx_t i = 0; i < e.length; i++) {
            if (!duckdb_validity_row_is_valid(duckdb_vector_get_validity(child), e.offset + i) ||
                ids[e.offset + i] < 0 || ids[e.offset + i] >= LFM2_MOE_VOCAB) {
                duckdb_scalar_function_set_error(info, "LFM2 requested token id must be non-NULL and within vocabulary"); return;
            }
        }
        char error[GGUF_ERROR_SIZE]; char *path = path_copy(paths[row], error);
        if (!path) { duckdb_scalar_function_set_error(info, error); return; }
        const lfm2_model *m = lfm2_model_get(path, error); free(path);
        if (!m) { duckdb_scalar_function_set_error(info, error); return; }
        duckdb_string_t text = texts[row]; int32_t *prompt_ids = NULL; size_t count = 0;
        if (!lfm2_prompt_ids(m, duckdb_string_t_data(&text), duckdb_string_t_length(text), true, &prompt_ids, &count, error)) {
            duckdb_scalar_function_set_error(info, error); return;
        }
        if (!count || count > LFM2_MAX_CONTEXT) {
            free(prompt_ids); duckdb_scalar_function_set_error(info, "LFM2 context exceeds workspace limit"); return;
        }
        if (!e.length) {
            free(prompt_ids);
            duckdb_list_entry *out = duckdb_vector_get_data(output);
            out[row] = (duckdb_list_entry){used, 0}; continue;
        }
        lfm2_state *s = lfm2_state_new(m, count, error);
        if (!s) { free(prompt_ids); duckdb_scalar_function_set_error(info, error); return; }
        bool ok = lfm2_prefill(s, prompt_ids, count, error); free(prompt_ids);
        if (ok && duckdb_list_vector_reserve(output, used + e.length) != DuckDBSuccess) {
            snprintf(error, sizeof(error), "out of memory for LFM2 logits"); ok = false;
        }
        if (ok) {
            float *values = duckdb_vector_get_data(duckdb_list_vector_get_child(output));
            ok = lfm2_state_logits(s, ids + e.offset, e.length, values + used, error);
        }
        lfm2_state_free(s);
        if (!ok) { duckdb_scalar_function_set_error(info, error); return; }
        duckdb_list_entry *out = duckdb_vector_get_data(output);
        out[row] = (duckdb_list_entry){used, e.length}; used += e.length;
    }
    duckdb_list_vector_set_size(output, used);
}

duckdb_state semantic_register_lfm2(duckdb_connection connection) {
    duckdb_scalar_function fn = duckdb_create_scalar_function();
    duckdb_logical_type str = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_logical_type real = duckdb_create_logical_type(DUCKDB_TYPE_FLOAT);
    duckdb_logical_type token = real ? duckdb_create_list_type(real) : NULL;
    duckdb_logical_type matrix = token ? duckdb_create_list_type(token) : NULL;
    duckdb_logical_type integer = duckdb_create_logical_type(DUCKDB_TYPE_INTEGER);
    duckdb_logical_type options = integer ? duckdb_create_list_type(integer) : NULL;
    duckdb_state status = DuckDBError;
    if (fn && str && matrix && integer && options) {
        duckdb_scalar_function_set_name(fn, "semantic_colbert_encode");
        for (unsigned j = 0; j < 3; j++) duckdb_scalar_function_add_parameter(fn, str);
        duckdb_scalar_function_set_return_type(fn, matrix);
        duckdb_scalar_function_set_function(fn, colbert_encode);
        duckdb_scalar_function_set_special_handling(fn);
        status = duckdb_register_scalar_function(connection, fn);
    }
    if (fn) duckdb_destroy_scalar_function(&fn);
    for (unsigned i = 0; i < 2 && status == DuckDBSuccess; i++) {
        fn = duckdb_create_scalar_function();
        if (!fn) { status = DuckDBError; break; }
        duckdb_scalar_function_set_name(fn, i ? "semantic_next_token_logits" : "semantic_generate");
        duckdb_scalar_function_add_parameter(fn, str); duckdb_scalar_function_add_parameter(fn, str);
        duckdb_scalar_function_add_parameter(fn, i ? options : integer);
        duckdb_scalar_function_set_return_type(fn, i ? token : str);
        duckdb_scalar_function_set_function(fn, i ? next_token_logits : generate);
        duckdb_scalar_function_set_special_handling(fn);
        status = duckdb_register_scalar_function(connection, fn);
        duckdb_destroy_scalar_function(&fn);
    }
    if (options) duckdb_destroy_logical_type(&options);
    if (integer) duckdb_destroy_logical_type(&integer);
    if (matrix) duckdb_destroy_logical_type(&matrix);
    if (token) duckdb_destroy_logical_type(&token);
    if (real) duckdb_destroy_logical_type(&real);
    if (str) duckdb_destroy_logical_type(&str);
    return status;
}
