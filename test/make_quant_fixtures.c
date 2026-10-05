/* Deterministic GGUF quant fixtures.
 * SPDX-License-Identifier: MIT
 */
#include "../src/quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tensor {
    const char *name;
    uint32_t type, rank;
    uint64_t dims[4], bytes, offset;
};

static void put(FILE *f, uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; i++) {
        if (fputc((int)((value >> (8 * i)) & 255), f) == EOF) {
            perror("write fixture");
            exit(1);
        }
    }
}

static void string(FILE *f, const char *s) {
    put(f, strlen(s), 8);
    if (fwrite(s, 1, strlen(s), f) != strlen(s)) { perror("write fixture"); exit(1); }
}

static void pad(FILE *f, unsigned alignment) {
    long offset = ftell(f);
    if (offset < 0) { perror("fixture offset"); exit(1); }
    while ((unsigned long)offset++ % alignment) put(f, 0, 1);
}

static void write_fixture(const char *dir, const char *name, struct tensor *t,
                           unsigned n, const uint8_t *data, size_t bytes, unsigned alignment) {
    char path[1024];
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) exit(1);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    put(f, UINT32_C(0x46554747), 4);
    put(f, 3, 4);
    put(f, n, 8);
    put(f, 1, 8);
    string(f, "general.alignment");
    put(f, 4, 4);
    put(f, alignment, 4);
    for (unsigned i = 0; i < n; i++) {
        string(f, t[i].name);
        put(f, t[i].rank, 4);
        for (unsigned j = 0; j < t[i].rank; j++) put(f, t[i].dims[j], 8);
        put(f, t[i].type, 4);
        put(f, t[i].offset, 8);
    }
    pad(f, alignment);
    if (fwrite(data, 1, bytes, f) != bytes || fclose(f)) { perror(path); exit(1); }
}

static uint32_t random32(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * UINT64_C(2685821657736338717)) >> 32);
}

static void store(uint8_t *p, uint32_t u, unsigned bytes) {
    for (unsigned i = 0; i < bytes; i++) p[i] = (uint8_t)(u >> (8 * i));
}

static void store_float(uint8_t *p, float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    store(p, bits, 4);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "build/fixtures";
    struct tensor t[] = {
        {"w.0", 0, 2, {512, 7}, 0, 0}, {"w.1", 1, 2, {512, 7}, 0, 0},
        {"w.30", 30, 2, {512, 7}, 0, 0}, {"w.8", 8, 2, {512, 7}, 0, 0},
        {"w.10", 10, 2, {512, 7}, 0, 0}, {"w.11", 11, 2, {512, 7}, 0, 0},
        {"w.12", 12, 2, {512, 7}, 0, 0}, {"w.13", 13, 2, {512, 7}, 0, 0},
        {"w.14", 14, 2, {512, 7}, 0, 0}, {"w.15", 15, 2, {512, 7}, 0, 0}
    };
    const unsigned n = sizeof(t) / sizeof(t[0]);
    uint64_t total = 0, state = 20260710;
    for (unsigned i = 0; i < n; i++) {
        const gguf_type_info *g = gguf_ggml_type(t[i].type);
        total = (total + 31) & ~UINT64_C(31);
        t[i].offset = total;
        t[i].bytes = 3584 / g->block_elements * g->block_bytes;
        total += t[i].bytes;
    }
    uint8_t *payload = calloc(1, (size_t)total);
    if (!payload) return 1;
    for (unsigned i = 0; i < n; i++) {
        const gguf_type_info *g = gguf_ggml_type(t[i].type);
        uint8_t *p = payload + t[i].offset;
        for (uint64_t block = 0; block < 3584 / g->block_elements; block++, p += g->block_bytes) {
            float value = (float)(random32(&state) >> 8) * 0x1p-22f - 2.f;
            if (t[i].type == QUANT_F32) { store_float(p, value); continue; }
            if (t[i].type == QUANT_F16 || t[i].type == QUANT_BF16) {
                store(p, t[i].type == QUANT_F16 ? quant_f32_to_f16(value) : quant_f32_to_bf16(value), 2);
                continue;
            }
            for (unsigned j = 0; j < g->block_bytes; j++) p[j] = (uint8_t)random32(&state);
            unsigned off = t[i].type == 10 ? 80 : t[i].type == 11 ? 108 : t[i].type == 14 ? 208 : 0;
            float d = value * 0.005f;
            if (t[i].type == 15) store_float(p, d);
            else store(p + off, quant_f32_to_f16(d), 2);
            if (t[i].type == 10 || t[i].type == 12 || t[i].type == 13) {
                store(p + (t[i].type == 10 ? 82 : 2), quant_f32_to_f16(0.007f), 2);
            }
            if (t[i].type == 15) {
                for (unsigned j = 0; j < 16; j++) {
                    int sum = 0;
                    for (unsigned k = 0; k < 16; k++) {
                        unsigned u = p[4 + j * 16 + k];
                        sum += u < 128 ? (int)u : (int)u - 256;
                    }
                    store(p + 260 + j * 2, (uint16_t)sum, 2);
                }
            }
        }
    }
    write_fixture(dir, "quant.gguf", t, n, payload, (size_t)total, 32);
    free(payload);
    struct tensor odd[] = {{"half", 1, 2, {3, 2}, 12, 0}, {"f32", 0, 2, {2, 2}, 16, 12}};
    uint8_t data[32] = {0};
    const float half[] = {1, -2, 3, 4, 5, -6};
    for (unsigned i = 0; i < 6; i++) store(data + i * 2, quant_f32_to_f16(half[i]), 2);
    for (unsigned i = 0; i < 4; i++) store_float(data + 12 + i * 4, (float)(i + 1));
    write_fixture(dir, "unaligned.gguf", odd, 2, data, 28, 4);
    struct tensor rank3 = {"w", 0, 3, {2, 2, 2}, 32, 0};
    for (unsigned i = 0; i < 8; i++) store_float(data + i * 4, (float)(i + 1));
    write_fixture(dir, "rank3.gguf", &rank3, 1, data, 32, 32);
    struct tensor bad = {"w", 8, 2, {16, 2}, 34, 0};
    uint8_t bad_data[34] = {0};
    write_fixture(dir, "bad_quant_row.gguf", &bad, 1, bad_data, 34, 32);
    puts("wrote quant fixtures");
    return 0;
}
