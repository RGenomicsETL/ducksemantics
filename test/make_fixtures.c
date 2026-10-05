/* make_fixtures DIR: write the small GGUF fixtures used by test/run.sh.
 * SPDX-License-Identifier: MIT
 *
 * valid.gguf exercises every metadata value type, a nested array, a
 * non-default alignment and three tensors (f32, q8_0 and an unknown ggml type
 * id). many.gguf has more keys and tensors than one DuckDB vector. Every other
 * file carries one deliberate defect that the reader must reject cleanly.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 };

typedef struct {
    unsigned char *p;
    size_t len, cap;
} buf;

static void put(buf *b, const void *data, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2 + 256;
        if (!(b->p = realloc(b->p, b->cap))) {
            perror("realloc");
            exit(1);
        }
    }
    if (n) memcpy(b->p + b->len, data, n);
    b->len += n;
}

static void le(buf *b, uint64_t v, int bytes) {
    for (int i = 0; i < bytes; i++) {
        unsigned char c = (unsigned char)(v >> (8 * i));
        put(b, &c, 1);
    }
}

static void u32(buf *b, uint32_t v) { le(b, v, 4); }
static void u64(buf *b, uint64_t v) { le(b, v, 8); }
static void str(buf *b, const char *s) { u64(b, strlen(s)); put(b, s, strlen(s)); }
static void zeros(buf *b, size_t n) { while (n--) put(b, "", 1); }
static void pad(buf *b, size_t alignment) { zeros(b, (alignment - b->len % alignment) % alignment); }

static void f32(buf *b, float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    u32(b, u);
}

static void header(buf *b, const char *magic, uint32_t version, uint64_t tensors, uint64_t kvs) {
    put(b, magic, 4);
    u32(b, version);
    u64(b, tensors);
    u64(b, kvs);
}

/* Key plus value type; the caller appends the value. */
static void key(buf *b, const char *k, uint32_t type) { str(b, k); u32(b, type); }

static void tensor(buf *b, const char *name, int n_dims, const uint64_t *dims,
                   uint32_t type, uint64_t offset) {
    str(b, name);
    u32(b, (uint32_t)n_dims);
    for (int i = 0; i < n_dims; i++) u64(b, dims[i]);
    u32(b, type);
    u64(b, offset);
}

static const char *dir;

static void emit(const char *name, buf *b) {
    char path[4096];
    FILE *f;
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    if (!(f = fopen(path, "wb")) || fwrite(b->p, 1, b->len, f) != b->len || fclose(f)) {
        perror(path);
        exit(1);
    }
    free(b->p);
    memset(b, 0, sizeof(*b));
}

/* One key "general.architecture" = "fixture"; the common malformed prefix. */
static void arch(buf *b) { key(b, "general.architecture", STR); str(b, "fixture"); }

static void valid(void) {
    buf b = {0};
    const uint64_t d42[] = {4, 2}, d32[] = {32}, d7[] = {7};
    header(&b, "GGUF", 3, 3, 18);
    arch(&b);
    key(&b, "general.alignment", U32); u32(&b, 64);
    key(&b, "general.name", STR); str(&b, "\xc3\x9c" "n" "\xc3\xaf" "code \xe2\x9c\x93 model");
    key(&b, "t.u8", U8); le(&b, 255, 1);
    key(&b, "t.i8", I8); le(&b, 0x80, 1);
    key(&b, "t.u16", U16); le(&b, 65535, 2);
    key(&b, "t.i16", I16); le(&b, 0x8000, 2);
    key(&b, "t.u32", U32); u32(&b, 4294967295u);
    key(&b, "t.i32", I32); u32(&b, 0x80000000u);
    key(&b, "t.f32", F32); f32(&b, 0.1f);
    key(&b, "t.bool", BOOL); le(&b, 1, 1);
    key(&b, "t.u64", U64); u64(&b, UINT64_MAX);
    key(&b, "t.i64", I64); u64(&b, UINT64_C(1) << 63);
    key(&b, "t.f64", F64); {
        double v = 2.5e-300;
        uint64_t u;
        memcpy(&u, &v, sizeof(u));
        u64(&b, u);
    }
    key(&b, "t.tokens", ARR); u32(&b, STR); u64(&b, 4);
    str(&b, "<s>"); str(&b, "\xe2\x96\x81hello"); str(&b, ""); str(&b, "\xe2\x9c\x93");
    key(&b, "t.scores", ARR); u32(&b, F32); u64(&b, 3); f32(&b, 0.0f); f32(&b, -1.5f); f32(&b, 3.25f);
    key(&b, "t.empty", ARR); u32(&b, I32); u64(&b, 0);
    key(&b, "t.nested", ARR); u32(&b, ARR); u64(&b, 2);
    u32(&b, U8); u64(&b, 2); le(&b, 1, 1); le(&b, 2, 1);
    u32(&b, U8); u64(&b, 1); le(&b, 3, 1);
    tensor(&b, "w.f32", 2, d42, 0, 0);
    tensor(&b, "w.q8_0", 1, d32, 8, 64);
    tensor(&b, "w.future", 1, d7, 99, 128);
    pad(&b, 64);
    for (int i = 0; i < 8; i++) f32(&b, (float)i);
    zeros(&b, 64 - 32);
    le(&b, 0x3c00, 2);  /* q8_0 scale 1.0 as fp16 */
    for (int i = 0; i < 32; i++) le(&b, (uint64_t)i, 1);
    zeros(&b, 64 - 34);
    zeros(&b, 64);
    emit("valid.gguf", &b);
}

static void many(void) {
    buf b = {0};
    char name[32];
    header(&b, "GGUF", 3, 3000, 3001);
    key(&b, "general.alignment", U32); u32(&b, 4);
    for (int i = 0; i < 3000; i++) {
        snprintf(name, sizeof(name), "k.%05d", i);
        key(&b, name, U32); u32(&b, (uint32_t)i);
    }
    for (int i = 0; i < 3000; i++) {
        const uint64_t ones[] = {1, 1, 1};
        snprintf(name, sizeof(name), "t.%05d", i);
        tensor(&b, name, 1 + i % 3, ones, 0, (uint64_t)4 * i);
    }
    pad(&b, 4);
    for (int i = 0; i < 3000; i++) f32(&b, (float)i);
    emit("many.gguf", &b);
}

/* The two floats 1, 2: the payload of a two-element f32 tensor. */
static void pair(buf *b) { f32(b, 1.0f); f32(b, 2.0f); }

static void malformed(void) {
    buf b = {0};
    const uint64_t d2[] = {2}, d16[] = {16}, d64[] = {64}, d11111[] = {1, 1, 1, 1, 1};

    header(&b, "GGUF", 3, 0, 0);
    pad(&b, 32);
    emit("empty.gguf", &b);

    header(&b, "GGML", 3, 0, 1); arch(&b); pad(&b, 32);
    emit("bad_magic.gguf", &b);

    header(&b, "GGUF", 1, 0, 1); arch(&b); pad(&b, 32);
    emit("version1.gguf", &b);

    header(&b, "GGUF", 3, 1, 1); arch(&b); tensor(&b, "w", 1, d2, 0, 0); pad(&b, 32);
    f32(&b, 1.0f);  /* second float missing */
    emit("truncated.gguf", &b);

    header(&b, "GGUF", 3, 0, UINT64_C(1) << 60); arch(&b); pad(&b, 32);
    emit("huge_counts.gguf", &b);

    header(&b, "GGUF", 3, 0, 1); key(&b, "k", STR); u64(&b, 100); put(&b, "abc", 3);
    emit("string_overrun.gguf", &b);

    header(&b, "GGUF", 3, 0, 2); arch(&b); arch(&b); pad(&b, 32);
    emit("duplicate_key.gguf", &b);

    header(&b, "GGUF", 3, 2, 1); arch(&b);
    tensor(&b, "w", 1, d2, 0, 0); tensor(&b, "w", 1, d2, 0, 32); pad(&b, 32);
    pair(&b); zeros(&b, 24); pair(&b);
    emit("duplicate_tensor.gguf", &b);

    header(&b, "GGUF", 3, 1, 1); arch(&b); tensor(&b, "w", 1, d2, 0, 4); pad(&b, 32);
    pair(&b); pair(&b);
    emit("misaligned.gguf", &b);

    header(&b, "GGUF", 3, 1, 1); arch(&b); tensor(&b, "w", 1, d64, 0, 0); pad(&b, 32);
    pair(&b);
    emit("payload_overrun.gguf", &b);

    header(&b, "GGUF", 3, 1, 1); arch(&b); tensor(&b, "w", 5, d11111, 0, 0); pad(&b, 32);
    pair(&b);
    emit("too_many_dims.gguf", &b);

    header(&b, "GGUF", 3, 0, 1); key(&b, "general.alignment", U32); u32(&b, 3); pad(&b, 32);
    emit("bad_alignment.gguf", &b);

    header(&b, "GGUF", 3, 0, 1); key(&b, "bad\xff" "key", U8); le(&b, 1, 1); pad(&b, 32);
    emit("bad_utf8.gguf", &b);

    header(&b, "GGUF", 3, 1, 1); arch(&b); tensor(&b, "w", 1, d16, 8, 0); pad(&b, 32);
    zeros(&b, 34);
    emit("partial_block.gguf", &b);

    header(&b, "GGUF", 3, 0, 1); key(&b, "k", 77); zeros(&b, 8);
    emit("unknown_value_type.gguf", &b);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s DIR\n", argv[0]);
        return 2;
    }
    dir = argv[1];
    valid();
    many();
    malformed();
    return 0;
}
