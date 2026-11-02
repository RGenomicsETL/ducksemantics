/* SPDX-License-Identifier: MIT */
#include "duckdb_extension.h"
#include "lfm2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

DUCKDB_EXTENSION_EXTERN

enum generation_field {
    GEN_TEXT, GEN_PROMPT_TOKENS, GEN_GENERATED_TOKENS, GEN_DECODE_TOKENS,
    GEN_PREFILL_BATCHES, GEN_PREFILL_BATCH_SIZE, GEN_MODEL_LOAD_SECONDS,
    GEN_PREFILL_SECONDS, GEN_DECODE_SECONDS, GEN_GENERATION_SECONDS,
    GEN_TOTAL_SECONDS, GEN_PREFILL_TPS, GEN_DECODE_TPS, GEN_STOPPED,
    GEN_FIELD_COUNT
};

static char generation_profile_tag;

static double profile_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

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
    bool profiled = duckdb_scalar_function_get_extra_info(info) != NULL;
    duckdb_vector p = duckdb_data_chunk_get_vector(input, 0), t = duckdb_data_chunk_get_vector(input, 1);
    duckdb_vector n = duckdb_data_chunk_get_vector(input, 2);
    duckdb_vector batch = duckdb_data_chunk_get_column_count(input) == 4 ? duckdb_data_chunk_get_vector(input, 3) : NULL;
    duckdb_string_t *paths = duckdb_vector_get_data(p), *texts = duckdb_vector_get_data(t);
    int32_t *limits = duckdb_vector_get_data(n), *batch_sizes = batch ? duckdb_vector_get_data(batch) : NULL;
    duckdb_vector fields[GEN_FIELD_COUNT];
    if (profiled)
        for (unsigned j = 0; j < GEN_FIELD_COUNT; j++) fields[j] = duckdb_struct_vector_get_child(output, j);
    for (idx_t row = 0; row < duckdb_data_chunk_get_size(input); row++) {
        if (!duckdb_validity_row_is_valid(duckdb_vector_get_validity(p), row) ||
            !duckdb_validity_row_is_valid(duckdb_vector_get_validity(t), row) ||
            !duckdb_validity_row_is_valid(duckdb_vector_get_validity(n), row) ||
            (batch && !duckdb_validity_row_is_valid(duckdb_vector_get_validity(batch), row))) {
            set_null(output, row);
            if (profiled)
                for (unsigned j = 0; j < GEN_FIELD_COUNT; j++) set_null(fields[j], row);
            continue;
        }
        if (limits[row] <= 0 || limits[row] > LFM2_MAX_GENERATED) {
            duckdb_scalar_function_set_error(info, "LFM2 generation length out of bounds"); return;
        }
        size_t batch_size = LFM2_PREFILL_BATCH;
        if (batch) {
            if (batch_sizes[row] <= 0 || batch_sizes[row] > LFM2_PREFILL_MAX_BATCH) {
                duckdb_scalar_function_set_error(info, "LFM2 prefill batch size out of bounds"); return;
            }
            batch_size = (size_t)batch_sizes[row];
        }
        double start = profile_seconds();
        char error[GGUF_ERROR_SIZE]; char *path = path_copy(paths[row], error);
        if (!path) { duckdb_scalar_function_set_error(info, error); return; }
        double load_start = profile_seconds();
        const lfm2_model *m = lfm2_model_get(path, error);
        double load_seconds = profile_seconds() - load_start;
        free(path);
        if (!m) { duckdb_scalar_function_set_error(info, error); return; }
        duckdb_string_t text = texts[row]; lfm2_generation result;
        if (!lfm2_generate_batch(m, duckdb_string_t_data(&text), duckdb_string_t_length(text),
                                 (size_t)limits[row], batch_size, &result, error)) {
            duckdb_scalar_function_set_error(info, error); return;
        }
        double total_seconds = profile_seconds() - start;
        if (profiled) {
            duckdb_vector_assign_string_element_len(fields[GEN_TEXT], row, result.text, result.bytes);
            ((uint64_t *)duckdb_vector_get_data(fields[GEN_PROMPT_TOKENS]))[row] = result.prompt_tokens;
            ((uint64_t *)duckdb_vector_get_data(fields[GEN_GENERATED_TOKENS]))[row] = result.count;
            ((uint64_t *)duckdb_vector_get_data(fields[GEN_DECODE_TOKENS]))[row] = result.decode_tokens;
            ((uint64_t *)duckdb_vector_get_data(fields[GEN_PREFILL_BATCHES]))[row] = result.prefill_batches;
            ((uint64_t *)duckdb_vector_get_data(fields[GEN_PREFILL_BATCH_SIZE]))[row] = result.prefill_batch_size;
            ((double *)duckdb_vector_get_data(fields[GEN_MODEL_LOAD_SECONDS]))[row] = load_seconds;
            ((double *)duckdb_vector_get_data(fields[GEN_PREFILL_SECONDS]))[row] = result.prefill_seconds;
            ((double *)duckdb_vector_get_data(fields[GEN_DECODE_SECONDS]))[row] = result.decode_seconds;
            ((double *)duckdb_vector_get_data(fields[GEN_GENERATION_SECONDS]))[row] = result.total_seconds;
            ((double *)duckdb_vector_get_data(fields[GEN_TOTAL_SECONDS]))[row] = total_seconds;
            if (result.prompt_tokens && result.prefill_seconds > 0)
                ((double *)duckdb_vector_get_data(fields[GEN_PREFILL_TPS]))[row] = (double)result.prompt_tokens / result.prefill_seconds;
            else set_null(fields[GEN_PREFILL_TPS], row);
            if (result.decode_tokens && result.decode_seconds > 0)
                ((double *)duckdb_vector_get_data(fields[GEN_DECODE_TPS]))[row] = (double)result.decode_tokens / result.decode_seconds;
            else set_null(fields[GEN_DECODE_TPS], row);
            ((bool *)duckdb_vector_get_data(fields[GEN_STOPPED]))[row] = result.stopped;
        } else duckdb_vector_assign_string_element_len(output, row, result.text, result.bytes);
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
    duckdb_logical_type count_type = duckdb_create_logical_type(DUCKDB_TYPE_UBIGINT);
    duckdb_logical_type seconds = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
    duckdb_logical_type boolean = duckdb_create_logical_type(DUCKDB_TYPE_BOOLEAN);
    duckdb_logical_type profile = NULL;
    if (str && count_type && seconds && boolean) {
        const char *names[GEN_FIELD_COUNT] = {
            "text", "prompt_tokens", "generated_tokens", "decode_tokens", "prefill_batches", "prefill_batch_size",
            "model_load_seconds", "prefill_seconds", "decode_seconds", "generation_seconds", "total_seconds",
            "prefill_tokens_per_second", "decode_tokens_per_second", "stopped"
        };
        duckdb_logical_type types[GEN_FIELD_COUNT] = {
            str, count_type, count_type, count_type, count_type, count_type,
            seconds, seconds, seconds, seconds, seconds, seconds, seconds, boolean
        };
        profile = duckdb_create_struct_type(types, names, GEN_FIELD_COUNT);
    }
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
        if (!i) duckdb_scalar_function_set_volatile(fn);
        status = duckdb_register_scalar_function(connection, fn);
        duckdb_destroy_scalar_function(&fn);
    }
    if (status == DuckDBSuccess && profile) {
        duckdb_scalar_function_set set = duckdb_create_scalar_function_set("semantic_generate_stats");
        if (!set) status = DuckDBError;
        for (unsigned parameters = 3; parameters <= 4 && status == DuckDBSuccess; parameters++) {
            fn = duckdb_create_scalar_function();
            if (!fn) { status = DuckDBError; break; }
            duckdb_scalar_function_set_name(fn, "semantic_generate_stats");
            duckdb_scalar_function_add_parameter(fn, str);
            duckdb_scalar_function_add_parameter(fn, str);
            for (unsigned j = 2; j < parameters; j++) duckdb_scalar_function_add_parameter(fn, integer);
            duckdb_scalar_function_set_return_type(fn, profile);
            duckdb_scalar_function_set_function(fn, generate);
            duckdb_scalar_function_set_extra_info(fn, &generation_profile_tag, NULL);
            duckdb_scalar_function_set_special_handling(fn);
            duckdb_scalar_function_set_volatile(fn);
            status = duckdb_add_scalar_function_to_set(set, fn);
            duckdb_destroy_scalar_function(&fn);
        }
        if (status == DuckDBSuccess) status = duckdb_register_scalar_function_set(connection, set);
        if (set) duckdb_destroy_scalar_function_set(&set);
    } else if (!profile) status = DuckDBError;
    if (profile) duckdb_destroy_logical_type(&profile);
    if (boolean) duckdb_destroy_logical_type(&boolean);
    if (seconds) duckdb_destroy_logical_type(&seconds);
    if (count_type) duckdb_destroy_logical_type(&count_type);
    if (options) duckdb_destroy_logical_type(&options);
    if (integer) duckdb_destroy_logical_type(&integer);
    if (matrix) duckdb_destroy_logical_type(&matrix);
    if (token) duckdb_destroy_logical_type(&token);
    if (real) duckdb_destroy_logical_type(&real);
    if (str) duckdb_destroy_logical_type(&str);
    return status;
}
