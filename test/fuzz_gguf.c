/* fuzz_gguf ITERATIONS SEED.gguf...: mutation smoke-fuzz of the GGUF reader.
 * SPDX-License-Identifier: MIT
 *
 * Each iteration flips bytes, overwrites 8-byte fields with extreme values,
 * or truncates one seed, then opens it and touches everything the SQL views
 * read: every scalar, every array element, every tensor field and the first
 * and last payload byte. Rejection is expected; build with
 * -fsanitize=address,undefined so any out-of-bounds access aborts.
 */
#include "../src/gguf.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint64_t rng = 0x9E3779B97F4A7C15u;

static uint64_t next(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}

static unsigned char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    unsigned char *p;
    long n;
    if (!f || fseek(f, 0, SEEK_END) || (n = ftell(f)) <= 0 || fseek(f, 0, SEEK_SET)) {
        perror(path);
        exit(1);
    }
    if (!(p = malloc((size_t)n)) || fread(p, 1, (size_t)n, f) != (size_t)n) {
        perror(path);
        exit(1);
    }
    fclose(f);
    *len = (size_t)n;
    return p;
}

static volatile unsigned sink;

static void touch(const gguf_file *f) {
    char scratch[64];
    gguf_str s;
    for (uint64_t i = 0; i < f->n_kv; i++) {
        const gguf_kv *kv = &f->kv[i];
        sink += kv->key.len ? (unsigned char)kv->key.ptr[kv->key.len - 1] : 0;
        if (kv->type != GGUF_ARRAY) {
            if (gguf_scalar_text(f, kv->type, kv->value_pos, scratch, sizeof(scratch), &s) && s.len) {
                sink += (unsigned char)s.ptr[s.len - 1];
            }
            continue;
        }
        uint64_t pos = kv->value_pos;
        for (uint64_t k = 0; k < kv->array_len; k++) {
            if (gguf_scalar_text(f, kv->array_type, pos, scratch, sizeof(scratch), &s) && s.len) {
                sink += (unsigned char)s.ptr[s.len - 1];
            }
            if (!gguf_value_skip(f, kv->array_type, &pos)) {
                fprintf(stderr, "validated array element did not skip\n");
                abort();
            }
        }
    }
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const gguf_tensor *t = &f->tensors[i];
        sink += t->name.len ? (unsigned char)t->name.ptr[0] : 0;
        if (t->n_bytes) sink += f->map[t->offset] + f->map[t->offset + t->n_bytes - 1];
    }
}

int main(int argc, char **argv) {
    static const uint64_t extremes[] = {
        0, 1, 0x7F, 0xFF, 0xFFFF, 0xFFFFFFFFu, UINT64_C(1) << 31, UINT64_C(1) << 63,
        UINT64_MAX, 2048, 4096,
    };
    char path[] = "/tmp/fuzz_gguf_XXXXXX";
    long iterations;
    int fd;
    unsigned long opened = 0;

    if (argc < 3 || (iterations = strtol(argv[1], NULL, 10)) <= 0) {
        fprintf(stderr, "usage: %s ITERATIONS SEED.gguf...\n", argv[0]);
        return 2;
    }
    if ((fd = mkstemp(path)) < 0) {
        perror("mkstemp");
        return 1;
    }
    close(fd);
    for (int seed = 2; seed < argc; seed++) {
        size_t len;
        unsigned char *orig = slurp(argv[seed], &len);
        unsigned char *data = malloc(len);
        for (long it = 0; it < iterations; it++) {
            size_t n = len;
            int edits = 1 + (int)(next() % 4);
            gguf_file f;
            FILE *out;
            memcpy(data, orig, len);
            for (int e = 0; e < edits && n > 0; e++) {
                uint64_t roll = next() % 100;
                if (roll < 45) {
                    data[next() % n] = (unsigned char)next();
                } else if (roll < 85) {
                    size_t at = n > 8 ? next() % (n - 8) : 0;
                    uint64_t v = extremes[next() % (sizeof(extremes) / sizeof(extremes[0]))];
                    for (int k = 0; k < 8 && at + k < n; k++) data[at + k] = (unsigned char)(v >> (8 * k));
                } else {
                    n = next() % n;
                }
            }
            if (!(out = fopen(path, "wb")) || (n && fwrite(data, 1, n, out) != n) || fclose(out)) {
                perror(path);
                return 1;
            }
            if (gguf_open(&f, path)) {
                touch(&f);
                opened++;
            }
            gguf_close(&f);
        }
        free(data);
        free(orig);
    }
    unlink(path);
    printf("ok %ld mutations x %d seeds: %lu opened, rest rejected, no fault\n",
           iterations, argc - 2, opened);
    return 0;
}
