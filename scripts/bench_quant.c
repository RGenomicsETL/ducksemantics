/* Single-core, page-warmed matvec payload-rate benchmark; no assertions.
 * SPDX-License-Identifier: MIT
 */
#include "../src/quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    bool seen[31] = {0};
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf ...\n", argv[0]); return 1; }
    for (int arg = 1; arg < argc; arg++) {
        gguf_file f;
        if (!gguf_open(&f, argv[arg])) { fprintf(stderr, "%s\n", f.error); return 1; }
        for (uint64_t i = 0; i < f.n_tensors; i++) {
            const gguf_tensor *t = f.tensors + i;
            if (t->type > 30 || seen[t->type] || t->n_dims < 2) continue;
            char error[GGUF_ERROR_SIZE] = {0};
            quant_view v;
            if (!quant_view_init(&v, &f, t, error)) continue;
            uint64_t bytes;
            if (!quant_matvec_scratch(&v, &bytes) || v.columns > SIZE_MAX / sizeof(float) ||
                v.rows > SIZE_MAX / sizeof(float)) continue;
            float *x = malloc((size_t)v.columns * sizeof(float));
            float *y = malloc((size_t)v.rows * sizeof(float));
            void *scratch = bytes ? malloc((size_t)bytes) : NULL;
            if (!x || !y || (bytes && !scratch)) { fprintf(stderr, "out of memory\n"); return 1; }
            for (uint64_t j = 0; j < v.columns; j++) x[j] = sinf((float)j * 0.037f);
            if (!quant_matvec(&v, x, v.columns, 0, v.rows, y, v.rows, scratch, bytes, error)) {
                fprintf(stderr, "%s\n", error);
                return 1;
            }
            double start = seconds(), elapsed;
            uint64_t iterations = 0;
            do {
                if (!quant_matvec(&v, x, v.columns, 0, v.rows, y, v.rows, scratch, bytes, error)) {
                    fprintf(stderr, "%s\n", error);
                    return 1;
                }
                iterations++;
                elapsed = seconds() - start;
            } while (elapsed < 0.5);
            printf("%s %llu x %llu %.3f GB/s (%llu B, %llu iterations, %.3fs) %.*s checksum=%.9g\n",
                   gguf_ggml_type(v.type)->name, (unsigned long long)v.rows,
                   (unsigned long long)v.columns, (double)v.n_bytes * iterations / elapsed / 1e9,
                   (unsigned long long)v.n_bytes, (unsigned long long)iterations, elapsed,
                   (int)t->name.len, t->name.ptr, (double)y[0] + y[v.rows - 1]);
            seen[v.type] = true;
            free(scratch);
            free(y);
            free(x);
        }
        gguf_close(&f);
    }
    return 0;
}
