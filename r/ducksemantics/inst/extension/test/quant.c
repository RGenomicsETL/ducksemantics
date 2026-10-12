/* Quant API bounds, unaligned bytes, and row-range contracts.
 * SPDX-License-Identifier: MIT
 */
#include "../src/quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #expr, error); \
    return 1; \
} } while (0)

int main(void) {
    char error[GGUF_ERROR_SIZE] = {0};
    gguf_file f;
    CHECK(gguf_open(&f, "build/fixtures/quant.gguf"));
    float x[512], full[7], part[7], deq[512], values[512];
    uint8_t scratch[584 + 1];
    for (int i = 0; i < 512; i++) x[i] = sinf((float)i * 0.037f);
    for (uint64_t i = 0; i < f.n_tensors; i++) {
        quant_view v;
        const uint8_t *row;
        uint64_t bytes;
        CHECK(quant_view_init(&v, &f, f.tensors + i, error));
        CHECK(quant_matvec_scratch(&v, &bytes));
        CHECK(bytes <= sizeof(scratch) - 1);
        CHECK(quant_matvec(&v, x, 512, 0, 7, full, 7, scratch + 1, bytes, error));
        for (uint64_t j = 0; j < 7; j++) {
            CHECK(quant_matvec(&v, x, 512, j, 1, part + j, 1, scratch + 1, bytes, error));
            CHECK(quant_view_row(&v, j, &row, error));
            CHECK(quant_dequantize_row(v.type, row, v.row_stride, deq, 512, error));
            CHECK(quant_values(&v, j * 512, 512, values, error));
            CHECK(!memcmp(deq, values, sizeof(deq)));
        }
        CHECK(!memcmp(full, part, sizeof(full)));
        CHECK(quant_values(&v, 0, 0, NULL, error));
        CHECK(quant_values(&v, 3584, 0, NULL, error));
        CHECK(!quant_values(&v, 3585, 0, NULL, error));
        CHECK(!quant_values(&v, 3583, 2, values, error));
        CHECK(!quant_values(&v, 1, UINT64_MAX, values, error));
        CHECK(!quant_view_row(&v, 7, &row, error));
        CHECK(!quant_view_row(&v, UINT64_MAX, &row, error));
        CHECK(!quant_matvec(&v, x, 511, 0, 7, full, 7, scratch, bytes, error));
        CHECK(!quant_matvec(&v, x, 512, 1, 7, full, 7, scratch, bytes, error));
        CHECK(!quant_matvec(&v, x, 512, 0, 7, full, 6, scratch, bytes, error));
        if (bytes) CHECK(!quant_matvec(&v, x, 512, 0, 7, full, 7, scratch, bytes - 1, error));
        CHECK(quant_matvec(&v, x, 512, 7, 0, NULL, 0, scratch, bytes, error));
        CHECK(!quant_dequantize_row(v.type, v.data, v.row_stride - 1, deq, 512, error));
        CHECK(!quant_vec_dot(v.type, v.data, v.row_stride - 1, x, sizeof(x), 512, full, error));
        CHECK(!quant_vec_dot(v.type, v.data, v.row_stride, x, 0, 512, full, error));
    }
    gguf_close(&f);
    uint64_t bytes;
    CHECK(!quant_row_bytes(99, 512, &bytes));
    CHECK(!quant_row_bytes(QUANT_F32, UINT64_MAX, &bytes));
    CHECK(!quant_row_bytes(QUANT_Q8_0, 31, &bytes));
    CHECK(!quant_row_bytes(QUANT_Q8_K, 255, &bytes));
    CHECK(!quantize_row_q8_0(x, 31, scratch, sizeof(scratch), error));
    CHECK(!quantize_row_q8_K(x, 255, scratch, sizeof(scratch), error));
    x[0] = NAN;
    CHECK(!quantize_row_q8_0(x, 32, scratch, sizeof(scratch), error));
    x[0] = INFINITY;
    CHECK(!quantize_row_q8_K(x, 256, scratch, sizeof(scratch), error));
    puts("ok   quant C API");
    return 0;
}
