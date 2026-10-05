/* CPU quantization kernels
 * SPDX-License-Identifier: MIT
 */
#ifndef DUCKSEMANTICS_QUANT_H
#define DUCKSEMANTICS_QUANT_H

#include "gguf.h"

/* GGML type IDs. Payloads and activation scratch are little-endian bytes. */
enum quant_type {
    QUANT_F32 = 0, QUANT_F16 = 1, QUANT_Q8_0 = 8,
    QUANT_Q2_K = 10, QUANT_Q3_K = 11, QUANT_Q4_K = 12,
    QUANT_Q5_K = 13, QUANT_Q6_K = 14, QUANT_Q8_K = 15, QUANT_BF16 = 30
};

typedef struct quant_view {
    const uint8_t *data;
    uint32_t type, n_dims;
    uint64_t dims[GGUF_MAX_DIMS];
    uint64_t columns, rows, row_stride, n_bytes;
} quant_view;

float quant_f16_to_f32(uint16_t h);
uint16_t quant_f32_to_f16(float f);
float quant_bf16_to_f32(uint16_t h);
uint16_t quant_f32_to_bf16(float f);

/* The file owns the view; close it only after all view users have finished.
 * Errors are written to error[GGUF_ERROR_SIZE]. No kernel allocates memory.
 */
bool quant_view_init(quant_view *v, const gguf_file *f, const gguf_tensor *t,
                     char *error);
bool quant_view_row(const quant_view *v, uint64_t row, const uint8_t **data,
                    char *error);
bool quant_row_bytes(uint32_t type, uint64_t n, uint64_t *bytes);
bool quant_dequantize_row(uint32_t type, const void *data, uint64_t bytes,
                          float *out, uint64_t n, char *error);
bool quant_values(const quant_view *v, uint64_t first, uint64_t count,
                   float *out, char *error);

/* Q8 reference quantizers; n must be a multiple of 32 or 256 respectively.
 * Input is finite native f32, with finite reciprocal scale. IEEE round-to-nearest
 * is required. Output is unaligned-safe GGML block storage. Q8_K zero blocks
 * include initialized bsums. Q8_0 may encode an infinite fp16 scale, as ggml does.
 */
bool quantize_row_q8_0(const float *x, uint64_t n, void *out, uint64_t bytes,
                       char *error);
bool quantize_row_q8_K(const float *x, uint64_t n, void *out, uint64_t bytes,
                       char *error);

/* Activation: Q8_0 for Q8_0 weights, Q8_K for all K weights, native f32
 * otherwise. K dots reduce exact integer sub-dots to one f32 per block,
 * then sum blocks left-to-right (bebelm order). Floating dots sum products
 * left-to-right in f32. Compile with -ffp-contract=off, without fast-math.
 */
bool quant_vec_dot(uint32_t type, const void *w, uint64_t w_bytes,
                    const void *x, uint64_t x_bytes, uint64_t n,
                    float *out, char *error);
bool quant_matvec_scratch(const quant_view *v, uint64_t *bytes);
/* y[0] corresponds to row first; scratch must not alias x, W, or y.
 * Matvec requires finite x and representable activation scales.
 */
bool quant_matvec(const quant_view *v, const float *x, uint64_t n,
                   uint64_t first, uint64_t count, float *y, uint64_t y_count,
                   void *scratch, uint64_t scratch_bytes, char *error);

/* Rbebelm K-quants use positive amax/127 scales and half-away rounding.
 * Other weight types multiply f32 activations directly. Scratch is caller-owned.
 */
typedef struct quant_bebel_block {
    int8_t q[256];
    float scale;
    int32_t sums[8];
} quant_bebel_block;

bool quant_bebel_quantize(const float *x, uint64_t n, quant_bebel_block *scratch,
                          uint64_t blocks, char *error);
bool quant_bebel_matvec(const quant_view *v, const float *x, bool integer_dot,
                        uint64_t first, uint64_t count, float *out,
                        quant_bebel_block *scratch, uint64_t blocks, char *error);
bool quant_bebel_matmul(const quant_view *v, const float *x, uint64_t tokens,
                        float *out, quant_bebel_block *scratch, uint64_t blocks, char *error);

#endif
