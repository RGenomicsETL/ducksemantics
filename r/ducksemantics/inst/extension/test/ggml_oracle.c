/* Read-only ggml reference adapter for the quant audit.
 * SPDX-License-Identifier: MIT
 */
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "../src/quant.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void oracle_dequant(int type, const void *data, float *out, int64_t n) {
    switch (type) {
    case 0: memcpy(out, data, (size_t)n * sizeof(float)); break;
    case 1:
        for (int64_t i = 0; i < n; i++) out[i] = ggml_compute_fp16_to_fp32(((const uint16_t *)data)[i]);
        break;
    case 30:
        for (int64_t i = 0; i < n; i++) out[i] = ggml_compute_bf16_to_fp32(((const ggml_bf16_t *)data)[i]);
        break;
    case 8: dequantize_row_q8_0(data, out, n); break;
    case 10: dequantize_row_q2_K(data, out, n); break;
    case 11: dequantize_row_q3_K(data, out, n); break;
    case 12: dequantize_row_q4_K(data, out, n); break;
    case 13: dequantize_row_q5_K(data, out, n); break;
    case 14: dequantize_row_q6_K(data, out, n); break;
    case 15: dequantize_row_q8_K(data, out, n); break;
    }
}

EXPORT void oracle_quant(int type, const float *x, void *out, int64_t n) {
    if (type == 8) quantize_row_q8_0_ref(x, out, n);
    else quantize_row_q8_K_ref(x, out, n);
}

static uint64_t check_widen(void) {
    uint64_t errors = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        float a = ggml_compute_fp16_to_fp32((uint16_t)h);
        float b = quant_f16_to_f32((uint16_t)h);
        errors += memcmp(&a, &b, sizeof(a)) != 0;
        a = ggml_compute_bf16_to_fp32((ggml_bf16_t){(uint16_t)h});
        b = quant_bf16_to_f32((uint16_t)h);
        errors += memcmp(&a, &b, sizeof(a)) != 0;
    }
    return errors;
}

static uint32_t random32(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * UINT64_C(2685821657736338717)) >> 32);
}

static uint64_t check_narrow(const float *x, int64_t n) {
    uint64_t errors = 0;
    for (int64_t i = 0; i < n; i++) {
        errors += ggml_compute_fp32_to_fp16(x[i]) != quant_f32_to_f16(x[i]);
        errors += ggml_compute_fp32_to_bf16(x[i]).bits != quant_f32_to_bf16(x[i]);
    }
    return errors;
}

/* .C entry points take caller-owned outputs and report status explicitly. */
EXPORT void oracle_activation(int *type, double *input, int *n, double *out, int *status) {
    *status = 1;
    uint64_t bytes;
    if ((*type != 8 && *type != 15) || *n <= 0 ||
        !quant_row_bytes((uint32_t)*type, (uint64_t)*n, &bytes)) return;
    float *x = malloc((size_t)*n * sizeof(float));
    float *values = malloc((size_t)*n * sizeof(float));
    void *packed = calloc(1, (size_t)bytes);
    if (x && values && packed) {
        for (int i = 0; i < *n; i++) x[i] = (float)input[i];
        oracle_quant(*type, x, packed, *n);
        oracle_dequant(*type, packed, values, *n);
        for (int i = 0; i < *n; i++) out[i] = values[i];
        *status = 0;
    }
    free(packed);
    free(values);
    free(x);
}

EXPORT void oracle_values(char **path, char **name, double *first, int *n,
                            double *out, int *status) {
    *status = 1;
    gguf_file f;
    if (!gguf_open(&f, *path)) { fprintf(stderr, "%s\n", f.error); gguf_close(&f); return; }
    const gguf_tensor *t = NULL;
    for (uint64_t i = 0; i < f.n_tensors; i++) {
        if (f.tensors[i].name.len == strlen(*name) &&
            !memcmp(f.tensors[i].name.ptr, *name, strlen(*name))) t = f.tensors + i;
    }
    quant_view v;
    char error[GGUF_ERROR_SIZE] = {0};
    if (!t || !quant_view_init(&v, &f, t, error) || *n < 0 ||
        !isfinite(*first) || *first < 0 || *first > (double)t->n_elements ||
        *first != floor(*first) || (uint64_t)*n > t->n_elements - (uint64_t)*first) {
        gguf_close(&f);
        return;
    }
    const gguf_type_info *g = gguf_ggml_type(t->type);
    float block[256];
    uint64_t index = (uint64_t)*first;
    int remaining = *n;
    while (remaining) {
        const uint8_t *row;
        if (!quant_view_row(&v, index / v.columns, &row, error)) { gguf_close(&f); return; }
        uint64_t col = index % v.columns;
        uint64_t offset = col / g->block_elements * g->block_bytes;
        if (offset > v.row_stride || g->block_bytes > v.row_stride - offset) { gguf_close(&f); return; }
        oracle_dequant((int)t->type, row + offset, block, g->block_elements);
        unsigned skip = (unsigned)(col % g->block_elements);
        int take = (int)g->block_elements - (int)skip;
        if (take > remaining) take = remaining;
        for (int i = 0; i < take; i++) *out++ = block[skip + (unsigned)i];
        index += (uint64_t)take;
        remaining -= take;
    }
    *status = 0;
    gguf_close(&f);
}

EXPORT void oracle_selftest(int *failures, int *widen, int *narrow, int *activations) {
    uint64_t errors = check_widen(), state = 811;
    *widen = 65536;
    *narrow = 0;
    *activations = 109;
    for (uint32_t h = 0; h < 65536; h++) {
        float half = ggml_compute_fp16_to_fp32((uint16_t)h);
        uint32_t bits;
        memcpy(&bits, &half, sizeof(bits));
        uint32_t patterns[] = {bits, bits - 1, bits + 1, h << 16};
        float x[4];
        memcpy(x, patterns, sizeof(x));
        errors += check_narrow(x, 4);
        *narrow += 4;
    }
    for (int i = 0; i < 100000; i++) {
        uint32_t bits = random32(&state);
        float x;
        memcpy(&x, &bits, sizeof(x));
        errors += check_narrow(&x, 1);
        (*narrow)++;
    }
    const float large[] = {1e-35f, -1e-35f, 1e30f, -1e30f, 0x1.fffffep127f, -0x1.fffffep127f};
    for (int trial = 0; trial < *activations; trial++) {
        float x[1024];
        for (int i = 0; i < 1024; i++) {
            if (!trial) x[i] = 0;
            else if (trial == 1) x[i] = (float)(i % 255 - 127);
            else if (trial == 2) { const float ties[] = {127, -127, .5f, -.5f, 1.5f, -1.5f}; x[i] = ties[i % 6]; }
            else if (trial < 9) x[i] = large[trial - 3];
            else x[i] = ((float)(random32(&state) >> 8) * 0x1p-22f - 2.f) * (float)(trial - 8);
        }
        for (int kind = 8; kind <= 15; kind += 7) {
            uint64_t bytes;
            quant_row_bytes((uint32_t)kind, 1024, &bytes);
            uint8_t a[1168] = {0}, b[1168] = {0};
            char error[GGUF_ERROR_SIZE];
            oracle_quant(kind, x, a, 1024);
            bool ok = kind == 8 ? quantize_row_q8_0(x, 1024, b, bytes, error) :
                                 quantize_row_q8_K(x, 1024, b, bytes, error);
            errors += !ok || memcmp(a, b, (size_t)bytes) != 0;
        }
    }
    *failures = (int)errors;
}

int main(void) {
    int failures, widen, narrow, activations;
    oracle_selftest(&failures, &widen, &narrow, &activations);
    printf("ggml oracle: %d widening patterns/type, %d narrowing patterns/type, "
           "%d activation vectors/type (%d values), %d failures\n",
           widen, narrow, activations, activations * 1024, failures);
    return failures != 0;
}
