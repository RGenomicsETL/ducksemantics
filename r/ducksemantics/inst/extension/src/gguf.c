/* DuckSemantics GGUF reader
 * SPDX-License-Identifier: MIT
 *
 * Layout: magic "GGUF", u32 version, u64 tensor count, u64 metadata count,
 * metadata key/value pairs, tensor infos, padding to general.alignment, then
 * tensor payloads. All integers are little-endian. Every value is visited once
 * at open time so that later accessors only revisit validated positions.
 */
#include "gguf.h"

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define GGUF_DEFAULT_ALIGNMENT 32
#define GGUF_MAX_ARRAY_DEPTH 4

/* Block geometry from ggml's type_traits (ggml-common.h block structs). */
static const gguf_type_info ggml_types[] = {
    [0] = {"f32", 1, 4},
    [1] = {"f16", 1, 2},
    [2] = {"q4_0", 32, 18},
    [3] = {"q4_1", 32, 20},
    [6] = {"q5_0", 32, 22},
    [7] = {"q5_1", 32, 24},
    [8] = {"q8_0", 32, 34},
    [9] = {"q8_1", 32, 36},
    [10] = {"q2_K", 256, 84},
    [11] = {"q3_K", 256, 110},
    [12] = {"q4_K", 256, 144},
    [13] = {"q5_K", 256, 176},
    [14] = {"q6_K", 256, 210},
    [15] = {"q8_K", 256, 292},
    [16] = {"iq2_xxs", 256, 66},
    [17] = {"iq2_xs", 256, 74},
    [18] = {"iq3_xxs", 256, 98},
    [19] = {"iq1_s", 256, 50},
    [20] = {"iq4_nl", 32, 18},
    [21] = {"iq3_s", 256, 110},
    [22] = {"iq2_s", 256, 82},
    [23] = {"iq4_xs", 256, 136},
    [24] = {"i8", 1, 1},
    [25] = {"i16", 1, 2},
    [26] = {"i32", 1, 4},
    [27] = {"i64", 1, 8},
    [28] = {"f64", 1, 8},
    [29] = {"iq1_m", 256, 56},
    [30] = {"bf16", 1, 2},
    [34] = {"tq1_0", 256, 54},
    [35] = {"tq2_0", 256, 66},
    [39] = {"mxfp4", 32, 17},
    [40] = {"nvfp4", 64, 36},
    [41] = {"q1_0", 128, 18},
    [42] = {"q2_0", 64, 18},
};

static const char *const value_type_names[GGUF_VALUE_TYPE_COUNT] = {
    "uint8", "int8", "uint16", "int16", "uint32", "int32", "float32",
    "bool", "string", "array", "uint64", "int64", "float64",
};

static const uint8_t value_type_sizes[GGUF_VALUE_TYPE_COUNT] = {
    1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8,
};

const gguf_type_info *gguf_ggml_type(uint32_t type) {
    if (type >= sizeof(ggml_types) / sizeof(ggml_types[0])) return NULL;
    return ggml_types[type].name ? &ggml_types[type] : NULL;
}

const char *gguf_value_type_name(uint32_t type) {
    return type < GGUF_VALUE_TYPE_COUNT ? value_type_names[type] : NULL;
}

static bool fail(gguf_file *f, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(f->error, sizeof(f->error), fmt, ap);
    va_end(ap);
    return false;
}

static bool have(const gguf_file *f, uint64_t pos, uint64_t n) {
    return pos <= f->size && n <= f->size - pos;
}

static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
           (uint32_t)p[3] << 24;
}

static uint64_t le64(const uint8_t *p) {
    return (uint64_t)le32(p) | (uint64_t)le32(p + 4) << 32;
}

static bool read_u32(gguf_file *f, uint64_t *pos, uint32_t *out) {
    if (!have(f, *pos, 4)) return fail(f, "truncated at byte %" PRIu64, *pos);
    *out = le32(f->map + *pos);
    *pos += 4;
    return true;
}

static bool read_u64(gguf_file *f, uint64_t *pos, uint64_t *out) {
    if (!have(f, *pos, 8)) return fail(f, "truncated at byte %" PRIu64, *pos);
    *out = le64(f->map + *pos);
    *pos += 8;
    return true;
}

static bool read_str(gguf_file *f, uint64_t *pos, gguf_str *out) {
    uint64_t len = 0;
    if (!read_u64(f, pos, &len)) return false;
    if (!have(f, *pos, len)) {
        return fail(f, "string of %" PRIu64 " bytes overruns the file at byte %" PRIu64,
                    len, *pos);
    }
    out->ptr = (const char *)(f->map + *pos);
    out->len = len;
    *pos += len;
    return true;
}

static bool skip_value(gguf_file *f, uint32_t type, uint64_t *pos, int depth) {
    if (type >= GGUF_VALUE_TYPE_COUNT) return fail(f, "unknown metadata type %" PRIu32, type);
    if (type == GGUF_STRING) {
        gguf_str s;
        return read_str(f, pos, &s);
    }
    if (type == GGUF_ARRAY) {
        uint32_t item = 0;
        uint64_t n = 0;
        if (depth >= GGUF_MAX_ARRAY_DEPTH) return fail(f, "metadata arrays nested too deeply");
        if (!read_u32(f, pos, &item) || !read_u64(f, pos, &n)) return false;
        if (item >= GGUF_VALUE_TYPE_COUNT) return fail(f, "unknown array element type %" PRIu32, item);
        if (value_type_sizes[item]) {
            if (n > (f->size - *pos) / value_type_sizes[item]) {
                return fail(f, "array of %" PRIu64 " elements overruns the file", n);
            }
            *pos += n * value_type_sizes[item];
            return true;
        }
        /* Variable-size elements need at least 8 bytes each. */
        if (n > (f->size - *pos) / 8) return fail(f, "array of %" PRIu64 " elements overruns the file", n);
        for (uint64_t i = 0; i < n; i++) {
            if (!skip_value(f, item, pos, depth + 1)) return false;
        }
        return true;
    }
    if (!have(f, *pos, value_type_sizes[type])) return fail(f, "truncated at byte %" PRIu64, *pos);
    *pos += value_type_sizes[type];
    return true;
}

bool gguf_value_skip(const gguf_file *f, uint32_t type, uint64_t *pos) {
    /* Positions handed out by gguf_open() were already validated; the cast only
     * lets the shared walker record an error, which cannot occur here. */
    return skip_value((gguf_file *)f, type, pos, 1);
}

static bool str_eq(gguf_str a, const char *b) {
    size_t n = strlen(b);
    return a.len == n && memcmp(a.ptr, b, n) == 0;
}

static int str_cmp(gguf_str a, gguf_str b) {
    uint64_t n = a.len < b.len ? a.len : b.len;
    int c = n ? memcmp(a.ptr, b.ptr, (size_t)n) : 0;
    if (c) return c;
    return a.len < b.len ? -1 : a.len > b.len;
}

static int cmp_str_ptr(const void *x, const void *y) {
    return str_cmp(**(const gguf_str *const *)x, **(const gguf_str *const *)y);
}

/* Duplicate keys or tensor names make a lookup ambiguous; ggml refuses them. */
static bool unique_names(gguf_file *f, const gguf_str **names, uint64_t n, const char *what) {
    qsort(names, (size_t)n, sizeof(*names), cmp_str_ptr);
    for (uint64_t i = 1; i < n; i++) {
        if (!str_cmp(*names[i - 1], *names[i])) {
            int len = names[i]->len > 64 ? 64 : (int)names[i]->len;
            return fail(f, "duplicate %s '%.*s'", what, len, names[i]->ptr);
        }
    }
    return true;
}

const gguf_kv *gguf_find_kv(const gguf_file *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++) {
        if (str_eq(f->kv[i].key, key)) return &f->kv[i];
    }
    return NULL;
}

static bool map_file(gguf_file *f, const char *path) {
#ifdef _WIN32
    (void)path;
    return fail(f, "memory-mapped GGUF is not implemented on Windows yet");
#else
    struct stat st;
    void *map;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(f, "cannot open '%s': %s", path, strerror(errno));
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        close(fd);
        return fail(f, "'%s' is not a regular file", path);
    }
    if (st.st_size < 24) {
        close(fd);
        return fail(f, "'%s' is too small to be GGUF", path);
    }
    map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return fail(f, "cannot map '%s': %s", path, strerror(errno));
    f->map = map;
    f->size = (uint64_t)st.st_size;
    return true;
#endif
}

static bool parse_alignment(gguf_file *f) {
    const gguf_kv *kv = gguf_find_kv(f, "general.alignment");
    f->alignment = GGUF_DEFAULT_ALIGNMENT;
    if (!kv) return true;
    if (kv->type != GGUF_UINT32) return fail(f, "general.alignment must be uint32");
    f->alignment = le32(f->map + kv->value_pos);
    if (f->alignment == 0 || (f->alignment & (f->alignment - 1))) {
        return fail(f, "general.alignment %" PRIu64 " is not a power of two", f->alignment);
    }
    return true;
}

static bool parse_tensor(gguf_file *f, uint64_t *pos, gguf_tensor *t) {
    const gguf_type_info *info;
    uint64_t relative;
    if (!read_str(f, pos, &t->name) || !read_u32(f, pos, &t->n_dims)) return false;
    if (t->n_dims == 0 || t->n_dims > GGUF_MAX_DIMS) {
        return fail(f, "tensor '%.*s' has %" PRIu32 " dimensions",
                    (int)(t->name.len > 64 ? 64 : t->name.len), t->name.ptr, t->n_dims);
    }
    t->n_elements = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) {
        if (!read_u64(f, pos, &t->dims[d])) return false;
        if (t->dims[d] > INT64_MAX ||
            (t->dims[d] && t->n_elements > (uint64_t)INT64_MAX / t->dims[d])) {
            return fail(f, "tensor '%.*s' element count overflows",
                        (int)(t->name.len > 64 ? 64 : t->name.len), t->name.ptr);
        }
        t->n_elements *= t->dims[d];
    }
    if (!read_u32(f, pos, &t->type) || !read_u64(f, pos, &relative)) return false;
    t->offset = relative;  /* made absolute once data_offset is known */
    info = gguf_ggml_type(t->type);
    t->n_bytes = 0;
    if (info) {
        if (t->dims[0] % info->block_elements) {
            return fail(f, "tensor '%.*s' row of %" PRIu64 " is not a multiple of the %s block",
                        (int)(t->name.len > 64 ? 64 : t->name.len), t->name.ptr,
                        t->dims[0], info->name);
        }
        t->n_bytes = t->n_elements / info->block_elements * info->block_bytes;
    }
    return true;
}

bool gguf_open(gguf_file *f, const char *path) {
    uint64_t pos = 0;
    uint32_t magic = 0;
    const gguf_str **names = NULL;
    bool ok = false;

    memset(f, 0, sizeof(*f));
    if (!map_file(f, path)) return false;
    if (!read_u32(f, &pos, &magic) || magic != 0x46554747u) {  /* "GGUF" */
        return fail(f, "'%s' does not start with the GGUF magic", path);
    }
    if (!read_u32(f, &pos, &f->version)) return false;
    if (f->version != 2 && f->version != 3) {
        return fail(f, "GGUF version %" PRIu32 " is not supported", f->version);
    }
    if (!read_u64(f, &pos, &f->n_tensors) || !read_u64(f, &pos, &f->n_kv)) return false;
    /* A key/value needs >= 13 bytes and a tensor info >= 32; reject counts the
     * file cannot hold before allocating for them. */
    if (f->n_kv > f->size / 13 || f->n_tensors > f->size / 32) {
        return fail(f, "header counts exceed what the file can hold");
    }
    f->kv = calloc(f->n_kv ? (size_t)f->n_kv : 1, sizeof(*f->kv));
    f->tensors = calloc(f->n_tensors ? (size_t)f->n_tensors : 1, sizeof(*f->tensors));
    names = malloc(sizeof(*names) * ((size_t)(f->n_kv > f->n_tensors ? f->n_kv : f->n_tensors) + 1));
    if (!f->kv || !f->tensors || !names) {
        fail(f, "out of memory");
        goto done;
    }

    for (uint64_t i = 0; i < f->n_kv; i++) {
        gguf_kv *kv = &f->kv[i];
        if (!read_str(f, &pos, &kv->key) || !read_u32(f, &pos, &kv->type)) goto done;
        if (kv->type >= GGUF_VALUE_TYPE_COUNT) {
            fail(f, "metadata '%.*s' has unknown type %" PRIu32,
                 (int)(kv->key.len > 64 ? 64 : kv->key.len), kv->key.ptr, kv->type);
            goto done;
        }
        if (kv->type == GGUF_ARRAY) {
            uint64_t start = pos;
            if (!read_u32(f, &pos, &kv->array_type) || !read_u64(f, &pos, &kv->array_len)) goto done;
            kv->value_pos = pos;
            pos = start;
        } else {
            kv->value_pos = pos;
        }
        if (!skip_value(f, kv->type, &pos, 0)) goto done;
        names[i] = &kv->key;
    }
    if (!unique_names(f, names, f->n_kv, "metadata key") || !parse_alignment(f)) goto done;

    for (uint64_t i = 0; i < f->n_tensors; i++) {
        if (!parse_tensor(f, &pos, &f->tensors[i])) goto done;
        names[i] = &f->tensors[i].name;
    }
    if (!unique_names(f, names, f->n_tensors, "tensor")) goto done;

    f->data_offset = pos + (f->alignment - pos % f->alignment) % f->alignment;
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        gguf_tensor *t = &f->tensors[i];
        if (t->offset % f->alignment) {
            fail(f, "tensor '%.*s' offset is not aligned",
                 (int)(t->name.len > 64 ? 64 : t->name.len), t->name.ptr);
            goto done;
        }
        if (t->offset > UINT64_MAX - f->data_offset ||
            !have(f, f->data_offset + t->offset, t->n_bytes)) {
            fail(f, "tensor '%.*s' payload overruns the file",
                 (int)(t->name.len > 64 ? 64 : t->name.len), t->name.ptr);
            goto done;
        }
        t->offset += f->data_offset;
    }
    ok = true;
done:
    free(names);
    return ok;
}

void gguf_close(gguf_file *f) {
#ifndef _WIN32
    if (f->map) munmap((void *)f->map, (size_t)f->size);
#endif
    free(f->kv);
    free(f->tensors);
    f->map = NULL;
    f->kv = NULL;
    f->tensors = NULL;
}

bool gguf_scalar_text(const gguf_file *f, uint32_t type, uint64_t pos,
                      char *buf, size_t cap, gguf_str *out_str) {
    const uint8_t *p;
    uint32_t u32;
    uint64_t u64;
    float f32;
    double f64;

    out_str->ptr = NULL;
    out_str->len = 0;
    if (type >= GGUF_VALUE_TYPE_COUNT || type == GGUF_ARRAY) return false;
    if (type == GGUF_STRING) {
        if (!have(f, pos, 8)) return false;
        u64 = le64(f->map + pos);
        if (!have(f, pos + 8, u64)) return false;
        out_str->ptr = (const char *)(f->map + pos + 8);
        out_str->len = u64;
        return true;
    }
    if (!have(f, pos, value_type_sizes[type])) return false;
    p = f->map + pos;
    switch (type) {
    case GGUF_UINT8: snprintf(buf, cap, "%u", (unsigned)p[0]); break;
    case GGUF_INT8: snprintf(buf, cap, "%d", (int)(int8_t)p[0]); break;
    case GGUF_UINT16: snprintf(buf, cap, "%u", (unsigned)(p[0] | p[1] << 8)); break;
    case GGUF_INT16: snprintf(buf, cap, "%d", (int)(int16_t)(p[0] | p[1] << 8)); break;
    case GGUF_UINT32: snprintf(buf, cap, "%" PRIu32, le32(p)); break;
    case GGUF_INT32: snprintf(buf, cap, "%" PRId32, (int32_t)le32(p)); break;
    case GGUF_FLOAT32:
        u32 = le32(p);
        memcpy(&f32, &u32, sizeof(f32));
        snprintf(buf, cap, "%.9g", (double)f32);
        break;
    case GGUF_BOOL: snprintf(buf, cap, "%s", p[0] ? "true" : "false"); break;
    case GGUF_UINT64: snprintf(buf, cap, "%" PRIu64, le64(p)); break;
    case GGUF_INT64: snprintf(buf, cap, "%" PRId64, (int64_t)le64(p)); break;
    case GGUF_FLOAT64:
        u64 = le64(p);
        memcpy(&f64, &u64, sizeof(f64));
        snprintf(buf, cap, "%.17g", f64);
        break;
    default: return false;
    }
    out_str->ptr = buf;
    out_str->len = strlen(buf);
    return true;
}
