/* SPDX-License-Identifier: MIT */
#include "../src/lfm2.h"
#include "../src/quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(bool ok, const char *label) {
    if (!ok) { fprintf(stderr, "lfm2_test: %s\n", label); exit(1); }
}

static void close_to(float a, float b, const char *label) {
    check(isfinite(a) && fabsf(a - b) <= 1e-6f * fmaxf(1.f, fabsf(b)), label);
}

int main(void) {
    float x[4] = {1, 2, 3, 4}, gain[4] = {1, 2, 3, 4}, norm[4];
    lfm2_rmsnorm(x, gain, 4, 0, norm);
    for (size_t i = 0; i < 4; i++) close_to(norm[i], x[i] * gain[i] / sqrtf(7.5f), "RMSNorm gains");
    memset(x, 0, sizeof(x));
    lfm2_rmsnorm(x, gain, 4, 1e-5f, norm);
    for (size_t i = 0; i < 4; i++) close_to(norm[i], 0, "RMSNorm zero");
    float rotary[4] = {1, 2, 3, 4};
    lfm2_rope(rotary, 4, 1, 10000);
    close_to(rotary[0], cosf(1) - 3 * sinf(1), "RoPE NeoX first half");
    close_to(rotary[3], 2 * sinf(.01f) + 4 * cosf(.01f), "RoPE NeoX second half");
    float bx[6] = {1, 10, 2, 20, 3, 30}, c[6] = {2, 3, 2, 3, 2, 3};
    float w[6] = {1, 2, 4, 1, 2, 4}, out[6], state[6] = {0};
    lfm2_conv_centered(bx, c, w, 3, 2, 3, out);
    close_to(out[0], 20, "centered convolution left padding");
    close_to(out[2], 34, "centered convolution interior");
    close_to(out[4], 16, "centered convolution right padding");
    for (size_t t = 0; t < 3; t++) lfm2_conv_step(bx + t * 2, c + t * 2, w, 2, 3, state, out + t * 2);
    close_to(out[0], 8, "causal convolution empty state");
    close_to(out[2], 20, "causal convolution shift");
    close_to(out[4], 34, "causal convolution full state");
    check(!memcmp(state, bx, sizeof(bx)), "convolution state order");
    float q[8] = {0}, k[4] = {1, 0, -1, 0}, v[4] = {2, 4, 6, 8}, scores[2], attended[8];
    lfm2_attention(q, k, v, 2, 2, 1, 2, scores, attended);
    for (size_t i = 0; i < 8; i++) close_to(attended[i], i % 2 ? 6 : 4, "bidirectional GQA values");
    float large[1024], ones[1024];
    for (size_t i = 0; i < 1024; i++) { large[i] = i ? 1 : 10000; ones[i] = 1; }
    lfm2_rmsnorm(large, ones, 1024, 0, large);
    close_to(large[1], 1.f / sqrtf((float)((1e8 + 1023) / 1024)), "RMSNorm f64 sum of f32 squares");
    float activation[512] = {0};
    activation[0] = 127; activation[1] = .5f; activation[2] = -.5f;
    activation[3] = 1.5f; activation[4] = -1.5f;
    quant_bebel_block scratch[4]; char error[GGUF_ERROR_SIZE];
    check(quant_bebel_quantize(activation, 512, scratch, 2, error), error);
    check(scratch[0].scale == 1 && scratch[0].q[1] == 1 && scratch[0].q[2] == -1 &&
          scratch[0].q[3] == 2 && scratch[0].q[4] == -2 && scratch[0].sums[0] == 127,
          "half-away Q8 rounding and sums");
    check(scratch[1].scale == 0 && scratch[1].sums[0] == 0, "zero block");
    check(!quant_bebel_quantize(activation, 512, scratch, 1, error), "scratch bound");
    activation[3] = INFINITY;
    check(!quant_bebel_quantize(activation, 256, scratch, 2, error), "finite activation contract");
    activation[3] = 1.5f;
    unsigned char packed[288] = {0};
    for (size_t b = 0; b < 2; b++) {
        packed[b * 144 + 1] = 0x3c;
        packed[b * 144 + 3] = 0x38;
        memset(packed + b * 144 + 4, 1, 8);
        memset(packed + b * 144 + 16, 0x22, 128);
    }
    quant_view view = {.data = packed, .columns = 512, .rows = 1, .row_stride = 288,
                       .type = QUANT_Q4_K};
    float scalar[2], batched[2], inputs[1024];
    memcpy(inputs, activation, sizeof(activation));
    memcpy(inputs + 512, activation, sizeof(activation));
    check(quant_bebel_matvec(&view, activation, true, 0, 1, scalar, scratch, 2, error), error);
    close_to(scalar[0], 190.5f, "Q4_K weighted integer dots and min sums");
    check(quant_bebel_matmul(&view, inputs, 2, batched, scratch, 4, error), error);
    check(batched[0] == scalar[0] && batched[1] == scalar[0], "batch and vector reduction order");
    check(!quant_bebel_matvec(&view, activation, true, 1, 1, scalar, scratch, 2, error), "row bounds");
    activation[1] = .25f;
    check(quant_bebel_matvec(&view, activation, false, 0, 1, scalar, NULL, 0, error), error);
    close_to(scalar[0], 190.125f, "Q4_K f32 activation path");
    unsigned char halves[8] = {0, 0x3c, 0, 0x40, 0, 0x42, 0, 0x44};
    quant_view half_view = {.data = halves, .columns = 2, .rows = 2, .row_stride = 4, .type = QUANT_F16};
    float two[2] = {1, 1};
    check(quant_bebel_matvec(&half_view, two, true, 0, 2, scalar, NULL, 0, error), error);
    close_to(scalar[0], 3, "F16 unquantized activations first row");
    close_to(scalar[1], 7, "F16 unquantized activations second row");
    unsigned char q8[34] = {0, 0x38};
    for (unsigned j = 0; j < 32; j++) { q8[2 + j] = (unsigned char)((int)j - 16); activation[j] = 1; }
    quant_view q8_view = {.data = q8, .columns = 32, .rows = 1, .row_stride = 34, .type = QUANT_Q8_0};
    check(quant_bebel_matvec(&q8_view, activation, true, 0, 1, scalar, NULL, 0, error), error);
    close_to(scalar[0], -8, "Q8_0 unquantized activations");
    unsigned char q6[210] = {0}; float weights[256];
    q6[209] = 0x3c;
    for (unsigned j = 0; j < 128; j++) q6[j] = (unsigned char)(j * 53 + 17);
    for (unsigned j = 0; j < 64; j++) q6[128 + j] = (unsigned char)(j * 97 + 5);
    for (unsigned j = 0; j < 16; j++) q6[192 + j] = (unsigned char)(j * 29 + 3);
    for (unsigned j = 0; j < 256; j++) activation[j] = ((int)(j % 5) - 2) * .1f;
    check(quant_dequantize_row(QUANT_Q6_K, q6, sizeof(q6), weights, 256, error), error);
    quant_view q6_view = {.data = q6, .columns = 256, .rows = 1, .row_stride = 210, .type = QUANT_Q6_K};
    check(quant_bebel_matvec(&q6_view, activation, false, 0, 1, scalar, NULL, 0, error), error);
    double expected = 0, magnitude = 0;
    for (unsigned j = 0; j < 256; j++) {
        expected += (double)weights[j] * activation[j]; magnitude += fabs((double)weights[j] * activation[j]);
    }
    check(fabs(scalar[0] - expected) <= 1e-5 * magnitude, "Q6_K f32 grouped sub-blocks");
    check(quant_bebel_matvec(&q6_view, activation, true, 0, 1, scalar, scratch, 1, error), error);
    expected = 0; magnitude = 0;
    for (unsigned j = 0; j < 256; j++) {
        double product = (double)weights[j] * (scratch[0].scale * scratch[0].q[j]);
        expected += product; magnitude += fabs(product);
    }
    check(fabs(scalar[0] - expected) <= 1e-5 * magnitude, "Q6_K rounded activation dot");
    check(lfm2_model_get("build/fixtures/valid.gguf", error) == NULL, "closed architecture profile");
    puts("lfm2_test: RMSNorm, RoPE, convolution, GQA, bebelm quantization OK");
    return 0;
}
