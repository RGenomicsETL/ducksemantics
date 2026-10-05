/* DuckSemantics GGUF reader
 * SPDX-License-Identifier: MIT
 *
 * A bounds-checked, read-only view of one memory-mapped GGUF v2/v3 file. It
 * records where metadata values and tensor payloads live; it decodes nothing
 * eagerly and never trusts a length, count, or offset read from the file.
 */
#ifndef DUCKSEMANTICS_GGUF_H
#define DUCKSEMANTICS_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define GGUF_MAX_DIMS 4
#define GGUF_ERROR_SIZE 256

enum gguf_value_type {
    GGUF_UINT8 = 0,
    GGUF_INT8 = 1,
    GGUF_UINT16 = 2,
    GGUF_INT16 = 3,
    GGUF_UINT32 = 4,
    GGUF_INT32 = 5,
    GGUF_FLOAT32 = 6,
    GGUF_BOOL = 7,
    GGUF_STRING = 8,
    GGUF_ARRAY = 9,
    GGUF_UINT64 = 10,
    GGUF_INT64 = 11,
    GGUF_FLOAT64 = 12,
    GGUF_VALUE_TYPE_COUNT
};

typedef struct gguf_str {
    const char *ptr;
    uint64_t len;
} gguf_str;

typedef struct gguf_kv {
    gguf_str key;
    uint32_t type;
    uint32_t array_type;  /* valid when type == GGUF_ARRAY */
    uint64_t array_len;   /* valid when type == GGUF_ARRAY */
    uint64_t value_pos;   /* scalar value, or first array element */
} gguf_kv;

typedef struct gguf_tensor {
    gguf_str name;
    uint32_t n_dims;
    uint64_t dims[GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t offset;      /* absolute file offset of the payload */
    uint64_t n_elements;
    uint64_t n_bytes;     /* 0 when the ggml type is unknown */
} gguf_tensor;

typedef struct gguf_type_info {
    const char *name;
    uint32_t block_elements;
    uint32_t block_bytes;
} gguf_type_info;

typedef struct gguf_file {
    const uint8_t *map;
    uint64_t size;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    gguf_kv *kv;
    gguf_tensor *tensors;
    uint64_t alignment;
    uint64_t data_offset;
    char error[GGUF_ERROR_SIZE];
} gguf_file;

/* Returns false and fills f->error on failure; gguf_close() is always safe. */
bool gguf_open(gguf_file *f, const char *path);
void gguf_close(gguf_file *f);

const gguf_type_info *gguf_ggml_type(uint32_t type);
const char *gguf_value_type_name(uint32_t type);

/* Renders a scalar metadata value as text. Strings are returned by reference
 * in *out_str; other scalars are printed into buf. Returns false for arrays. */
bool gguf_scalar_text(const gguf_file *f, uint32_t type, uint64_t pos,
                      char *buf, size_t cap, gguf_str *out_str);

/* Advances *pos past one value of `type`; used to walk string arrays. */
bool gguf_value_skip(const gguf_file *f, uint32_t type, uint64_t *pos);

const gguf_kv *gguf_find_kv(const gguf_file *f, const char *key);

#endif
