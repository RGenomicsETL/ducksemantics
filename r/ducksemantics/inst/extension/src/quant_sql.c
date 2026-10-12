/* Quant kernel SQL validation surface
 * SPDX-License-Identifier: MIT
 */
#include "duckdb_extension.h"
#include "quant.h"

#include <stdlib.h>
#include <string.h>

DUCKDB_EXTENSION_EXTERN

static bool valid(duckdb_vector vector, idx_t row) {
    uint64_t *mask = duckdb_vector_get_validity(vector);
    return !mask || duckdb_validity_row_is_valid(mask, row);
}

static char *string_copy(duckdb_vector vector, idx_t row, char *error) {
    duckdb_string_t s = ((duckdb_string_t *)duckdb_vector_get_data(vector))[row];
    uint32_t len = duckdb_string_t_length(s);
    const char *p = duckdb_string_t_data(&s);
    if (memchr(p, 0, len)) {
        strcpy(error, "path and tensor name must not contain NUL bytes");
        return NULL;
    }
    if ((uint64_t)len + 1 > SIZE_MAX) { strcpy(error, "string size out of bounds"); return NULL; }
    char *out = malloc((size_t)len + 1);
    if (!out) { strcpy(error, "out of memory"); return NULL; }
    memcpy(out, p, len);
    out[len] = 0;
    return out;
}

static bool open_view(duckdb_data_chunk input, idx_t row, gguf_file *f,
                       quant_view *v, char *error) {
    char *path = string_copy(duckdb_data_chunk_get_vector(input, 0), row, error);
    if (!path) return false;
    bool ok = gguf_open(f, path);
    free(path);
    if (!ok) { strcpy(error, f->error); return false; }
    char *name = string_copy(duckdb_data_chunk_get_vector(input, 1), row, error);
    if (!name) return false;
    size_t len = strlen(name);
    const gguf_tensor *t = NULL;
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        if (f->tensors[i].name.len == len && !memcmp(f->tensors[i].name.ptr, name, len)) {
            t = f->tensors + i;
            break;
        }
    }
    free(name);
    if (!t) { strcpy(error, "tensor not found"); return false; }
    return quant_view_init(v, f, t, error);
}

static float *list_output(duckdb_vector output, idx_t row, uint64_t count, char *error) {
    idx_t size = duckdb_list_vector_get_size(output);
    if (size > SIZE_MAX / sizeof(float) || count > SIZE_MAX / sizeof(float) - size ||
        duckdb_list_vector_reserve(output, size + count) != DuckDBSuccess ||
        duckdb_list_vector_set_size(output, size + count) != DuckDBSuccess) {
        strcpy(error, "result list size out of bounds or out of memory");
        return NULL;
    }
    duckdb_list_entry *entries = duckdb_vector_get_data(output);
    entries[row] = (duckdb_list_entry){.offset = size, .length = count};
    float *data = duckdb_vector_get_data(duckdb_list_vector_get_child(output));
    return count ? data + size : data;
}

static void quant_sql(duckdb_function_info info, duckdb_data_chunk input,
                       duckdb_vector output) {
    bool matvec = duckdb_data_chunk_get_column_count(input) == 3;
    idx_t n = duckdb_data_chunk_get_size(input);
    duckdb_vector arg = duckdb_data_chunk_get_vector(input, 2);
    for (idx_t row = 0; row < n; row++) {
        bool null = false;
        for (idx_t col = 0; col < duckdb_data_chunk_get_column_count(input); col++) {
            if (!valid(duckdb_data_chunk_get_vector(input, col), row)) null = true;
        }
        if (null) {
            duckdb_vector_ensure_validity_writable(output);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(output), row);
            continue;
        }
        char error[GGUF_ERROR_SIZE] = {0};
        gguf_file f = {0};
        quant_view v;
        void *scratch = NULL;
        bool ok = open_view(input, row, &f, &v, error);
        if (ok && matvec) {
            duckdb_list_entry entry = ((duckdb_list_entry *)duckdb_vector_get_data(arg))[row];
            idx_t size = duckdb_list_vector_get_size(arg);
            duckdb_vector child = duckdb_list_vector_get_child(arg);
            if (v.n_dims < 2 || entry.length != v.columns || entry.offset > size ||
                entry.length > size - entry.offset) {
                strcpy(error, "matvec requires rank >= 2 and x length equal to dim[0]");
                ok = false;
            }
            for (idx_t j = 0; ok && j < entry.length; j++) {
                if (!valid(child, entry.offset + j)) {
                    strcpy(error, "activation must not contain NULL elements");
                    ok = false;
                }
            }
            uint64_t bytes = 0;
            if (ok && (!quant_matvec_scratch(&v, &bytes) || (bytes && !(scratch = malloc((size_t)bytes))))) {
                strcpy(error, "activation scratch out of memory");
                ok = false;
            }
            if (ok) {
                float *y = list_output(output, row, v.rows, error);
                const float *x = duckdb_vector_get_data(child);
                ok = y && quant_matvec(&v, x + entry.offset, entry.length, 0, v.rows,
                                       y, v.rows, scratch, bytes, error);
            }
        } else if (ok) {
            uint64_t first = ((uint64_t *)duckdb_vector_get_data(arg))[row];
            uint64_t count = ((uint64_t *)duckdb_vector_get_data(duckdb_data_chunk_get_vector(input, 3)))[row];
            uint64_t total = v.rows * v.columns;
            if (first > total || count > total - first) {
                strcpy(error, "tensor element range out of bounds");
                ok = false;
            } else {
                float *y = list_output(output, row, count, error);
                ok = !error[0] && quant_values(&v, first, count, y, error);
            }
        }
        free(scratch);
        gguf_close(&f);
        if (!ok) { duckdb_scalar_function_set_error(info, error); return; }
    }
}

bool ducksemantics_register_quant(duckdb_connection con) {
    duckdb_logical_type varchar = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_logical_type ubigint = duckdb_create_logical_type(DUCKDB_TYPE_UBIGINT);
    duckdb_logical_type real = duckdb_create_logical_type(DUCKDB_TYPE_FLOAT);
    duckdb_logical_type list = duckdb_create_list_type(real);
    bool ok = true;
    for (int i = 0; i < 2 && ok; i++) {
        duckdb_scalar_function fn = duckdb_create_scalar_function();
        duckdb_scalar_function_set_name(fn, i ? "gguf_tensor_matvec" : "gguf_tensor_values");
        duckdb_scalar_function_add_parameter(fn, varchar);
        duckdb_scalar_function_add_parameter(fn, varchar);
        duckdb_scalar_function_add_parameter(fn, i ? list : ubigint);
        if (!i) duckdb_scalar_function_add_parameter(fn, ubigint);
        duckdb_scalar_function_set_return_type(fn, list);
        duckdb_scalar_function_set_volatile(fn);
        duckdb_scalar_function_set_function(fn, quant_sql);
        ok = duckdb_register_scalar_function(con, fn) == DuckDBSuccess;
        duckdb_destroy_scalar_function(&fn);
    }
    duckdb_destroy_logical_type(&list);
    duckdb_destroy_logical_type(&real);
    duckdb_destroy_logical_type(&ubigint);
    duckdb_destroy_logical_type(&varchar);
    return ok;
}
