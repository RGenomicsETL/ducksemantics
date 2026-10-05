/* DuckDB scalar adapters for GGUF tokenizers. SPDX-License-Identifier: MIT */
#include "duckdb_extension.h"
DUCKDB_EXTENSION_EXTERN

#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool valid(duckdb_vector v, idx_t row) {
    uint64_t *mask = duckdb_vector_get_validity(v);
    return !mask || duckdb_validity_row_is_valid(mask, row);
}

static const tokenizer *row_tokenizer(duckdb_vector paths, idx_t row, char *error) {
    duckdb_string_t value = ((duckdb_string_t *)duckdb_vector_get_data(paths))[row];
    uint32_t len = duckdb_string_t_length(value);
    const char *data = duckdb_string_t_data(&value);
    if (memchr(data, '\0', len)) {
        snprintf(error, GGUF_ERROR_SIZE, "tokenizer path contains NUL");
        return NULL;
    }
    char *path = malloc((size_t)len + 1);
    if (!path) { snprintf(error, GGUF_ERROR_SIZE, "out of memory in tokenizer"); return NULL; }
    memcpy(path, data, len);
    path[len] = '\0';
    const tokenizer *t = tokenizer_cached(path, error);
    free(path);
    return t;
}

static void tokenize_sql(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector paths = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector texts = duckdb_data_chunk_get_vector(input, 1);
    duckdb_vector specials = duckdb_data_chunk_get_vector(input, 2);
    duckdb_string_t *strings = duckdb_vector_get_data(texts);
    bool *flags = duckdb_vector_get_data(specials);
    duckdb_list_entry *entries = duckdb_vector_get_data(output);
    idx_t used = 0;
    duckdb_vector_ensure_validity_writable(output);
    uint64_t *mask = duckdb_vector_get_validity(output);
    for (idx_t row = 0; row < duckdb_data_chunk_get_size(input); row++) {
        if (!valid(paths, row) || !valid(texts, row) || !valid(specials, row)) {
            duckdb_validity_set_row_invalid(mask, row);
            entries[row] = (duckdb_list_entry){used, 0};
            continue;
        }
        char error[GGUF_ERROR_SIZE];
        const tokenizer *t = row_tokenizer(paths, row, error);
        if (!t) { duckdb_scalar_function_set_error(info, error); return; }
        int32_t *ids;
        size_t count;
        if (!tokenizer_encode(t, duckdb_string_t_data(&strings[row]), duckdb_string_t_length(strings[row]),
                              flags[row], &ids, &count, error)) {
            duckdb_scalar_function_set_error(info, error); return;
        }
        if (count > UINT64_MAX - used || duckdb_list_vector_reserve(output, used + count) != DuckDBSuccess ||
            duckdb_list_vector_set_size(output, used + count) != DuckDBSuccess) {
            free(ids); duckdb_scalar_function_set_error(info, "cannot allocate token list"); return;
        }
        int32_t *child = duckdb_vector_get_data(duckdb_list_vector_get_child(output));
        if (count) memcpy(child + used, ids, count * sizeof(*ids));
        entries[row] = (duckdb_list_entry){used, count};
        used += count;
        free(ids);
    }
}

static void detokenize_sql(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector paths = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector lists = duckdb_data_chunk_get_vector(input, 1);
    duckdb_list_entry *entries = duckdb_vector_get_data(lists);
    duckdb_vector child = duckdb_list_vector_get_child(lists);
    int32_t *ids = duckdb_vector_get_data(child);
    idx_t size = duckdb_list_vector_get_size(lists);
    duckdb_vector_ensure_validity_writable(output);
    uint64_t *mask = duckdb_vector_get_validity(output);
    for (idx_t row = 0; row < duckdb_data_chunk_get_size(input); row++) {
        if (!valid(paths, row) || !valid(lists, row)) { duckdb_validity_set_row_invalid(mask, row); continue; }
        duckdb_list_entry e = entries[row];
        if (e.offset > size || e.length > size - e.offset || e.length > SIZE_MAX) {
            duckdb_scalar_function_set_error(info, "token list bounds invalid"); return;
        }
        for (idx_t i = 0; i < e.length; i++) {
            if (!valid(child, e.offset + i)) { duckdb_scalar_function_set_error(info, "NULL token id"); return; }
        }
        char error[GGUF_ERROR_SIZE];
        const tokenizer *t = row_tokenizer(paths, row, error);
        if (!t) { duckdb_scalar_function_set_error(info, error); return; }
        char *text;
        size_t len;
        if (!tokenizer_decode(t, e.length ? ids + e.offset : NULL, (size_t)e.length, &text, &len, error)) {
            duckdb_scalar_function_set_error(info, error); return;
        }
        duckdb_vector_assign_string_element_len(output, row, text, len);
        free(text);
    }
}

duckdb_state semantic_register_tokenizers(duckdb_connection con) {
    duckdb_logical_type string = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_logical_type boolean = duckdb_create_logical_type(DUCKDB_TYPE_BOOLEAN);
    duckdb_logical_type integer = duckdb_create_logical_type(DUCKDB_TYPE_INTEGER);
    duckdb_logical_type list = duckdb_create_list_type(integer);
    duckdb_scalar_function encode = duckdb_create_scalar_function();
    duckdb_scalar_function decode = duckdb_create_scalar_function();
    duckdb_scalar_function_set_name(encode, "semantic_tokenize");
    duckdb_scalar_function_add_parameter(encode, string);
    duckdb_scalar_function_add_parameter(encode, string);
    duckdb_scalar_function_add_parameter(encode, boolean);
    duckdb_scalar_function_set_return_type(encode, list);
    duckdb_scalar_function_set_function(encode, tokenize_sql);
    duckdb_scalar_function_set_volatile(encode);
    duckdb_scalar_function_set_name(decode, "semantic_detokenize");
    duckdb_scalar_function_add_parameter(decode, string);
    duckdb_scalar_function_add_parameter(decode, list);
    duckdb_scalar_function_set_return_type(decode, string);
    duckdb_scalar_function_set_function(decode, detokenize_sql);
    duckdb_scalar_function_set_volatile(decode);
    duckdb_state state = duckdb_register_scalar_function(con, encode);
    if (state == DuckDBSuccess) state = duckdb_register_scalar_function(con, decode);
    duckdb_destroy_scalar_function(&encode);
    duckdb_destroy_scalar_function(&decode);
    duckdb_destroy_logical_type(&list);
    duckdb_destroy_logical_type(&integer);
    duckdb_destroy_logical_type(&boolean);
    duckdb_destroy_logical_type(&string);
    return state;
}
