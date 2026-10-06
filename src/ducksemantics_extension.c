/* DuckSemantics DuckDB extension
 * SPDX-License-Identifier: MIT
 *
 * CPU-only semantic kernels over the stable DuckDB C extension API:
 *   gguf_header(path), gguf_metadata(path), gguf_metadata_array(path, key),
 *   gguf_tensors(path)        read-only views of one GGUF model file
 *   semantic_maxsim(q, d)     ColBERT late-interaction score
 * No model is loaded or executed here yet; the GGUF views are the receipt and
 * inspection surface the model kernels will bind to.
 */
#include "duckdb_extension.h"

#include "gguf.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DUCKDB_EXTENSION_EXTERN

bool ducksemantics_register_quant(duckdb_connection connection);

/* ------------------------------------------------------------------------ */
/* Shared helpers                                                           */

static bool utf8_valid(const char *s, uint64_t n) {
    const unsigned char *p = (const unsigned char *)s;
    uint64_t i = 0;
    while (i < n) {
        unsigned c = p[i];
        uint64_t extra;
        uint32_t cp;
        if (c < 0x80) {
            i++;
            continue;
        }
        if (c >= 0xC2 && c <= 0xDF) {
            extra = 1;
            cp = c & 0x1F;
        } else if (c >= 0xE0 && c <= 0xEF) {
            extra = 2;
            cp = c & 0x0F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            extra = 3;
            cp = c & 0x07;
        } else {
            return false;
        }
        if (n - i <= extra) return false;
        for (uint64_t k = 1; k <= extra; k++) {
            if ((p[i + k] & 0xC0) != 0x80) return false;
            cp = cp << 6 | (p[i + k] & 0x3F);
        }
        if ((extra == 2 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) ||
            (extra == 3 && (cp < 0x10000 || cp > 0x10FFFF))) {
            return false;
        }
        i += extra + 1;
    }
    return true;
}

static void set_null(duckdb_vector vector, idx_t row) {
    duckdb_vector_ensure_validity_writable(vector);
    duckdb_validity_set_row_invalid(duckdb_vector_get_validity(vector), row);
}

/* Assigns text, or NULL when s.ptr is NULL. Invalid UTF-8 is an error: DuckDB
 * VARCHAR must hold valid UTF-8, and the reference gguf-py reader also refuses
 * to decode it. */
static bool put_text(duckdb_function_info info, duckdb_vector vector, idx_t row, gguf_str s) {
    if (!s.ptr) {
        set_null(vector, row);
        return true;
    }
    if (!utf8_valid(s.ptr, s.len)) {
        duckdb_function_set_error(info, "GGUF string is not valid UTF-8");
        return false;
    }
    duckdb_vector_assign_string_element_len(vector, row, s.ptr, (idx_t)s.len);
    return true;
}

static gguf_str cstr(const char *s) {
    gguf_str out = {s, s ? strlen(s) : 0};
    return out;
}

static bool add_column(duckdb_bind_info info, const char *name, duckdb_type type) {
    duckdb_logical_type logical = duckdb_create_logical_type(type);
    if (!logical) return false;
    duckdb_bind_add_result_column(info, name, logical);
    duckdb_destroy_logical_type(&logical);
    return true;
}

/* ------------------------------------------------------------------------ */
/* GGUF table functions                                                     */

typedef enum {
    GGUF_VIEW_HEADER,
    GGUF_VIEW_METADATA,
    GGUF_VIEW_ARRAY,
    GGUF_VIEW_TENSORS
} gguf_view;

typedef struct {
    gguf_view view;
    gguf_file file;
    const gguf_kv *array;  /* GGUF_VIEW_ARRAY */
} gguf_bind_data;

typedef struct {
    uint64_t row;
    uint64_t pos;  /* next array element, GGUF_VIEW_ARRAY */
} gguf_scan_state;

static void gguf_bind_data_free(void *data) {
    gguf_bind_data *bind = data;
    if (!bind) return;
    gguf_close(&bind->file);
    free(bind);
}

static char *bind_text_parameter(duckdb_bind_info info, idx_t index, const char *what) {
    duckdb_value value = duckdb_bind_get_parameter(info, index);
    char *text = NULL;
    if (value && !duckdb_is_null_value(value)) text = duckdb_get_varchar(value);
    if (value) duckdb_destroy_value(&value);
    if (!text) {
        char message[96];
        snprintf(message, sizeof(message), "%s must be a non-NULL string", what);
        duckdb_bind_set_error(info, message);
    }
    return text;
}

static void gguf_bind(duckdb_bind_info info, gguf_view view) {
    gguf_bind_data *bind = calloc(1, sizeof(*bind));
    char *path;
    uint64_t rows = 0;
    bool ok = true;

    if (!bind) {
        duckdb_bind_set_error(info, "out of memory");
        return;
    }
    bind->view = view;
    path = bind_text_parameter(info, 0, "GGUF path");
    if (!path) {
        free(bind);
        return;
    }
    if (!gguf_open(&bind->file, path)) {
        duckdb_bind_set_error(info, bind->file.error);
        duckdb_free(path);
        gguf_bind_data_free(bind);
        return;
    }
    duckdb_free(path);

    switch (view) {
    case GGUF_VIEW_HEADER:
        ok = add_column(info, "version", DUCKDB_TYPE_UINTEGER) &&
             add_column(info, "tensor_count", DUCKDB_TYPE_UBIGINT) &&
             add_column(info, "metadata_count", DUCKDB_TYPE_UBIGINT) &&
             add_column(info, "alignment", DUCKDB_TYPE_UBIGINT) &&
             add_column(info, "data_offset", DUCKDB_TYPE_UBIGINT) &&
             add_column(info, "file_bytes", DUCKDB_TYPE_UBIGINT) &&
             add_column(info, "tensor_bytes", DUCKDB_TYPE_UBIGINT);
        rows = 1;
        break;
    case GGUF_VIEW_METADATA:
        ok = add_column(info, "key", DUCKDB_TYPE_VARCHAR) &&
             add_column(info, "value_type", DUCKDB_TYPE_VARCHAR) &&
             add_column(info, "array_type", DUCKDB_TYPE_VARCHAR) &&
             add_column(info, "array_length", DUCKDB_TYPE_UBIGINT) &&
             add_column(info, "value", DUCKDB_TYPE_VARCHAR);
        rows = bind->file.n_kv;
        break;
    case GGUF_VIEW_ARRAY: {
        char *key = bind_text_parameter(info, 1, "metadata key");
        if (!key) {
            gguf_bind_data_free(bind);
            return;
        }
        bind->array = gguf_find_kv(&bind->file, key);
        if (!bind->array || bind->array->type != GGUF_ARRAY) {
            char message[GGUF_ERROR_SIZE];
            snprintf(message, sizeof(message), "GGUF metadata key '%.120s' is not an array", key);
            duckdb_free(key);
            duckdb_bind_set_error(info, message);
            gguf_bind_data_free(bind);
            return;
        }
        duckdb_free(key);
        ok = add_column(info, "array_index", DUCKDB_TYPE_UBIGINT) &&
             add_column(info, "value", DUCKDB_TYPE_VARCHAR);
        rows = bind->array->array_len;
        break;
    }
    case GGUF_VIEW_TENSORS: {
        duckdb_logical_type ubigint = duckdb_create_logical_type(DUCKDB_TYPE_UBIGINT);
        duckdb_logical_type dims = ubigint ? duckdb_create_list_type(ubigint) : NULL;
        ok = dims && add_column(info, "tensor_index", DUCKDB_TYPE_UBIGINT) &&
             add_column(info, "name", DUCKDB_TYPE_VARCHAR) &&
             add_column(info, "ggml_type", DUCKDB_TYPE_VARCHAR) &&
             add_column(info, "ggml_type_id", DUCKDB_TYPE_UINTEGER);
        if (ok) duckdb_bind_add_result_column(info, "dims", dims);
        ok = ok && add_column(info, "n_elements", DUCKDB_TYPE_UBIGINT) &&
             add_column(info, "file_offset", DUCKDB_TYPE_UBIGINT) &&
             add_column(info, "n_bytes", DUCKDB_TYPE_UBIGINT);
        if (dims) duckdb_destroy_logical_type(&dims);
        if (ubigint) duckdb_destroy_logical_type(&ubigint);
        rows = bind->file.n_tensors;
        break;
    }
    }
    if (!ok) {
        duckdb_bind_set_error(info, "failed to declare GGUF result columns");
        gguf_bind_data_free(bind);
        return;
    }
    duckdb_bind_set_cardinality(info, (idx_t)rows, true);
    duckdb_bind_set_bind_data(info, bind, gguf_bind_data_free);
}

static void gguf_header_bind(duckdb_bind_info info) { gguf_bind(info, GGUF_VIEW_HEADER); }
static void gguf_metadata_bind(duckdb_bind_info info) { gguf_bind(info, GGUF_VIEW_METADATA); }
static void gguf_array_bind(duckdb_bind_info info) { gguf_bind(info, GGUF_VIEW_ARRAY); }
static void gguf_tensors_bind(duckdb_bind_info info) { gguf_bind(info, GGUF_VIEW_TENSORS); }

static void gguf_init(duckdb_init_info info) {
    gguf_bind_data *bind = duckdb_init_get_bind_data(info);
    gguf_scan_state *state = calloc(1, sizeof(*state));
    if (!state) {
        duckdb_init_set_error(info, "out of memory");
        return;
    }
    if (bind->view == GGUF_VIEW_ARRAY) state->pos = bind->array->value_pos;
    duckdb_init_set_init_data(info, state, free);
    duckdb_init_set_max_threads(info, 1);
}

static void put_u64(duckdb_vector vector, idx_t row, uint64_t value) {
    ((uint64_t *)duckdb_vector_get_data(vector))[row] = value;
}

static void gguf_scan(duckdb_function_info info, duckdb_data_chunk output) {
    gguf_bind_data *bind = duckdb_function_get_bind_data(info);
    gguf_scan_state *state = duckdb_function_get_init_data(info);
    const gguf_file *f = &bind->file;
    idx_t capacity = duckdb_vector_size();
    idx_t count = 0;
    char scratch[64];

    switch (bind->view) {
    case GGUF_VIEW_HEADER: {
        uint64_t tensor_bytes = 0;
        if (state->row > 0) break;
        for (uint64_t i = 0; i < f->n_tensors; i++) tensor_bytes += f->tensors[i].n_bytes;
        ((uint32_t *)duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, 0)))[0] = f->version;
        put_u64(duckdb_data_chunk_get_vector(output, 1), 0, f->n_tensors);
        put_u64(duckdb_data_chunk_get_vector(output, 2), 0, f->n_kv);
        put_u64(duckdb_data_chunk_get_vector(output, 3), 0, f->alignment);
        put_u64(duckdb_data_chunk_get_vector(output, 4), 0, f->data_offset);
        put_u64(duckdb_data_chunk_get_vector(output, 5), 0, f->size);
        put_u64(duckdb_data_chunk_get_vector(output, 6), 0, tensor_bytes);
        state->row = 1;
        count = 1;
        break;
    }
    case GGUF_VIEW_METADATA: {
        duckdb_vector key = duckdb_data_chunk_get_vector(output, 0);
        duckdb_vector value_type = duckdb_data_chunk_get_vector(output, 1);
        duckdb_vector array_type = duckdb_data_chunk_get_vector(output, 2);
        duckdb_vector array_length = duckdb_data_chunk_get_vector(output, 3);
        duckdb_vector value = duckdb_data_chunk_get_vector(output, 4);
        for (; count < capacity && state->row < f->n_kv; count++, state->row++) {
            const gguf_kv *kv = &f->kv[state->row];
            gguf_str text = {NULL, 0};
            if (!put_text(info, key, count, kv->key) ||
                !put_text(info, value_type, count, cstr(gguf_value_type_name(kv->type)))) {
                return;
            }
            if (kv->type == GGUF_ARRAY) {
                if (!put_text(info, array_type, count, cstr(gguf_value_type_name(kv->array_type)))) return;
                put_u64(array_length, count, kv->array_len);
            } else {
                set_null(array_type, count);
                set_null(array_length, count);
                gguf_scalar_text(f, kv->type, kv->value_pos, scratch, sizeof(scratch), &text);
            }
            if (!put_text(info, value, count, text)) return;
        }
        break;
    }
    case GGUF_VIEW_ARRAY: {
        duckdb_vector index = duckdb_data_chunk_get_vector(output, 0);
        duckdb_vector value = duckdb_data_chunk_get_vector(output, 1);
        const gguf_kv *kv = bind->array;
        for (; count < capacity && state->row < kv->array_len; count++, state->row++) {
            gguf_str text = {NULL, 0};
            put_u64(index, count, state->row);
            gguf_scalar_text(f, kv->array_type, state->pos, scratch, sizeof(scratch), &text);
            if (!put_text(info, value, count, text)) return;
            if (!gguf_value_skip(f, kv->array_type, &state->pos)) {
                duckdb_function_set_error(info, "GGUF array element overruns the file");
                return;
            }
        }
        break;
    }
    case GGUF_VIEW_TENSORS: {
        duckdb_vector index = duckdb_data_chunk_get_vector(output, 0);
        duckdb_vector name = duckdb_data_chunk_get_vector(output, 1);
        duckdb_vector type_name = duckdb_data_chunk_get_vector(output, 2);
        duckdb_vector type_id = duckdb_data_chunk_get_vector(output, 3);
        duckdb_vector dims = duckdb_data_chunk_get_vector(output, 4);
        duckdb_vector elements = duckdb_data_chunk_get_vector(output, 5);
        duckdb_vector offset = duckdb_data_chunk_get_vector(output, 6);
        duckdb_vector bytes = duckdb_data_chunk_get_vector(output, 7);
        duckdb_list_entry *entries = duckdb_vector_get_data(dims);
        idx_t child_size = 0;
        uint64_t end = state->row + capacity < f->n_tensors ? state->row + capacity : f->n_tensors;
        uint64_t *child;
        for (uint64_t i = state->row; i < end; i++) child_size += f->tensors[i].n_dims;
        if (duckdb_list_vector_reserve(dims, child_size) != DuckDBSuccess) {
            duckdb_function_set_error(info, "out of memory");
            return;
        }
        child = duckdb_vector_get_data(duckdb_list_vector_get_child(dims));
        child_size = 0;
        for (; state->row < end; count++, state->row++) {
            const gguf_tensor *t = &f->tensors[state->row];
            const gguf_type_info *type = gguf_ggml_type(t->type);
            put_u64(index, count, state->row);
            if (!put_text(info, name, count, t->name) ||
                !put_text(info, type_name, count, cstr(type ? type->name : NULL))) {
                return;
            }
            ((uint32_t *)duckdb_vector_get_data(type_id))[count] = t->type;
            entries[count].offset = child_size;
            entries[count].length = t->n_dims;
            for (uint32_t d = 0; d < t->n_dims; d++) child[child_size++] = t->dims[d];
            put_u64(elements, count, t->n_elements);
            put_u64(offset, count, t->offset);
            if (type) {
                put_u64(bytes, count, t->n_bytes);
            } else {
                set_null(bytes, count);
            }
        }
        duckdb_list_vector_set_size(dims, child_size);
        break;
    }
    }
    duckdb_data_chunk_set_size(output, count);
}

static bool register_gguf_view(duckdb_connection connection, const char *name,
                               duckdb_table_function_bind_t bind, idx_t text_parameters) {
    duckdb_table_function function = duckdb_create_table_function();
    duckdb_logical_type varchar;
    duckdb_state status;
    if (!function) return false;
    varchar = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    if (!varchar) {
        duckdb_destroy_table_function(&function);
        return false;
    }
    duckdb_table_function_set_name(function, name);
    for (idx_t i = 0; i < text_parameters; i++) duckdb_table_function_add_parameter(function, varchar);
    duckdb_destroy_logical_type(&varchar);
    duckdb_table_function_set_bind(function, bind);
    duckdb_table_function_set_init(function, gguf_init);
    duckdb_table_function_set_function(function, gguf_scan);
    status = duckdb_register_table_function(connection, function);
    duckdb_destroy_table_function(&function);
    return status == DuckDBSuccess;
}

/* ------------------------------------------------------------------------ */
/* semantic_maxsim                                                          */

/* Eight independent partial sums keep the reduction order fixed, so a score is
 * bit-identical across builds and ISAs (the build disables FMA contraction),
 * while still letting the compiler vectorize the loop. */
static float dot(const float *a, const float *b, uint64_t n) {
    float s[8] = {0};
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        for (int k = 0; k < 8; k++) s[k] += a[i + k] * b[i + k];
    }
    for (; i < n; i++) s[i & 7] += a[i] * b[i];
    return ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
}

typedef struct {
    duckdb_list_entry *outer;
    uint64_t *outer_validity;
    duckdb_list_entry *tokens;
    uint64_t *token_validity;
    float *values;
    uint64_t *value_validity;
} token_matrix;

static token_matrix token_matrix_view(duckdb_vector vector) {
    duckdb_vector tokens = duckdb_list_vector_get_child(vector);
    duckdb_vector values = duckdb_list_vector_get_child(tokens);
    token_matrix m = {
        duckdb_vector_get_data(vector), duckdb_vector_get_validity(vector),
        duckdb_vector_get_data(tokens), duckdb_vector_get_validity(tokens),
        duckdb_vector_get_data(values), duckdb_vector_get_validity(values),
    };
    return m;
}

/* Checks one row's token vectors: none NULL, no NULL element, one width. */
static const char *token_row_width(const token_matrix *m, duckdb_list_entry row, uint64_t *width) {
    *width = 0;
    for (uint64_t t = 0; t < row.length; t++) {
        idx_t token = (idx_t)(row.offset + t);
        duckdb_list_entry entry;
        if (!duckdb_validity_row_is_valid(m->token_validity, token)) return "token embedding is NULL";
        entry = m->tokens[token];
        if (t == 0) {
            *width = entry.length;
        } else if (entry.length != *width) {
            return "token embeddings in one matrix have different widths";
        }
        if (m->value_validity) {
            for (uint64_t k = 0; k < entry.length; k++) {
                if (!duckdb_validity_row_is_valid(m->value_validity, (idx_t)(entry.offset + k))) {
                    return "token embedding contains a NULL element";
                }
            }
        }
    }
    return NULL;
}

static void semantic_maxsim(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    idx_t n = duckdb_data_chunk_get_size(input);
    token_matrix q = token_matrix_view(duckdb_data_chunk_get_vector(input, 0));
    token_matrix d = token_matrix_view(duckdb_data_chunk_get_vector(input, 1));
    double *result = duckdb_vector_get_data(output);

    for (idx_t row = 0; row < n; row++) {
        duckdb_list_entry qr, dr;
        uint64_t qw, dw;
        const char *problem;
        double score = 0.0;
        if (!duckdb_validity_row_is_valid(q.outer_validity, row) ||
            !duckdb_validity_row_is_valid(d.outer_validity, row)) {
            set_null(output, row);
            continue;
        }
        qr = q.outer[row];
        dr = d.outer[row];
        if ((problem = token_row_width(&q, qr, &qw)) || (problem = token_row_width(&d, dr, &dw))) {
            duckdb_scalar_function_set_error(info, problem);
            return;
        }
        /* An empty side has no best match; zero-width tokens carry no signal. */
        if (qr.length == 0 || dr.length == 0 || qw == 0) {
            set_null(output, row);
            continue;
        }
        if (qw != dw) {
            duckdb_scalar_function_set_error(info, "query and document embeddings have different widths");
            return;
        }
        for (uint64_t i = 0; i < qr.length; i++) {
            const float *qv = q.values + q.tokens[qr.offset + i].offset;
            float best = -INFINITY;
            for (uint64_t j = 0; j < dr.length; j++) {
                float s = dot(qv, d.values + d.tokens[dr.offset + j].offset, qw);
                if (s > best) best = s;
            }
            score += best;
        }
        result[row] = score;
    }
}

static bool register_maxsim(duckdb_connection connection) {
    duckdb_scalar_function function = duckdb_create_scalar_function();
    duckdb_logical_type real = duckdb_create_logical_type(DUCKDB_TYPE_FLOAT);
    duckdb_logical_type token = real ? duckdb_create_list_type(real) : NULL;
    duckdb_logical_type matrix = token ? duckdb_create_list_type(token) : NULL;
    duckdb_logical_type score = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
    duckdb_state status = DuckDBError;
    if (function && matrix && score) {
        duckdb_scalar_function_set_name(function, "semantic_maxsim");
        duckdb_scalar_function_add_parameter(function, matrix);
        duckdb_scalar_function_add_parameter(function, matrix);
        duckdb_scalar_function_set_return_type(function, score);
        duckdb_scalar_function_set_function(function, semantic_maxsim);
        status = duckdb_register_scalar_function(connection, function);
    }
    if (score) duckdb_destroy_logical_type(&score);
    if (matrix) duckdb_destroy_logical_type(&matrix);
    if (token) duckdb_destroy_logical_type(&token);
    if (real) duckdb_destroy_logical_type(&real);
    if (function) duckdb_destroy_scalar_function(&function);
    return status == DuckDBSuccess;
}

/* ------------------------------------------------------------------------ */

duckdb_state semantic_register_tokenizers(duckdb_connection connection);
duckdb_state semantic_register_embeddinggemma(duckdb_connection connection);

DUCKDB_EXTENSION_ENTRYPOINT(duckdb_connection connection, duckdb_extension_info info,
                            struct duckdb_extension_access *access) {
    if (!register_gguf_view(connection, "gguf_header", gguf_header_bind, 1) ||
        !register_gguf_view(connection, "gguf_metadata", gguf_metadata_bind, 1) ||
        !register_gguf_view(connection, "gguf_metadata_array", gguf_array_bind, 2) ||
        !register_gguf_view(connection, "gguf_tensors", gguf_tensors_bind, 1)) {
        access->set_error(info, "failed to register GGUF table functions");
        return false;
    }
    if (!register_maxsim(connection)) {
        access->set_error(info, "failed to register semantic_maxsim()");
        return false;
    }
    if (!ducksemantics_register_quant(connection)) {
        access->set_error(info, "failed to register quant validation functions");
        return false;
    }
    if (semantic_register_tokenizers(connection) != DuckDBSuccess) {
        access->set_error(info, "failed to register tokenizer functions");
        return false;
    }
    if (semantic_register_embeddinggemma(connection) != DuckDBSuccess) {
        access->set_error(info, "failed to register semantic_embed()");
        return false;
    }
    return true;
}
