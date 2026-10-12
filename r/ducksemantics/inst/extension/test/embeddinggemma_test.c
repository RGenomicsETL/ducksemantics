/* EmbeddingGemma admission, scratch and numeric contracts without model weights. */
#include "embeddinggemma.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

static float reference_dot(const float *a, const float *b, size_t n) {
    float acc[4][8] = {{0}}, lanes[8];
    size_t i = 0;
    while (i + 32 <= n) {
        for (unsigned j = 0; j < 4; j++)
            for (unsigned k = 0; k < 8; k++) acc[j][k] = fmaf(a[i + j * 8 + k], b[i + j * 8 + k], acc[j][k]);
        i += 32;
    }
    while (i + 8 <= n) {
        for (unsigned k = 0; k < 8; k++) acc[0][k] = fmaf(a[i + k], b[i + k], acc[0][k]);
        i += 8;
    }
    for (unsigned k = 0; k < 8; k++) lanes[k] = (acc[0][k] + acc[1][k]) + (acc[2][k] + acc[3][k]);
    float sum = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) +
                ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
    while (i < n) { sum += a[i] * b[i]; i++; }
    return sum;
}

int main(void) {
    static const char *tasks[] = {"retrieval_query", "retrieval_document", "question_answering",
        "fact_verification", "classification", "clustering", "semantic_similarity",
        "code_retrieval", "summarization", "raw"};
    for (int i = 0; i < GEMMA_TASK_COUNT; i++) CHECK(gemma_task(tasks[i], strlen(tasks[i])) == i);
    CHECK(gemma_task("query", 5) == -1);
    CHECK(gemma_task("", 0) == -1);
    CHECK(gemma_task("raw\0x", 5) == -1);
    CHECK(gemma_dimensions(768) && gemma_dimensions(512) && gemma_dimensions(256) && gemma_dimensions(128));
    CHECK(!gemma_dimensions(0) && !gemma_dimensions(129) && !gemma_dimensions(-1));
    float a[GEMMA_DIM], b[GEMMA_DIM], output[GEMMA_DIM];
    for (size_t i = 0; i < GEMMA_DIM; i++) {
        a[i] = (float)((int)(i % 17) - 8) * 0.1234567f;
        b[i] = (float)((int)(i % 23) - 11) * 0.9876543f;
    }
    for (size_t n = 0; n <= GEMMA_DIM; n++) CHECK(quant_bebelm_dot(a, b, n) == reference_dot(a, b, n));
    gemma_select(a, 128, false, output);
    CHECK(!memcmp(a, output, 128 * sizeof(float)));
    gemma_select(a, 128, true, output);
    double ss = 0;
    for (size_t i = 0; i < 128; i++) ss += (double)output[i] * output[i];
    CHECK(fabs(ss - 1.0) < 1e-7);
    memset(a, 0, sizeof(a));
    gemma_select(a, GEMMA_DIM, true, output);
    CHECK(!memcmp(a, output, sizeof(a)));

    /* Three Q8_0 matrix rows, two blocks per row, spanning a token tile. */
    uint8_t weights[3 * 68];
    for (size_t r = 0; r < 3; r++) for (size_t block = 0; block < 2; block++) {
        uint8_t *p = weights + r * 68 + block * 34;
        p[0] = 0; p[1] = 0x38; /* fp16 0.5 */
        for (size_t i = 0; i < 32; i++) p[i + 2] = (uint8_t)((int)((i + r + block) % 19) - 9);
    }
    quant_view v = {.type = QUANT_Q8_0, .data = weights, .columns = 64, .rows = 3, .row_stride = 68};
    float x[40 * 64], y[40 * 3], row[64], scratch[64];
    char error[GGUF_ERROR_SIZE];
    for (size_t i = 0; i < 40 * 64; i++) x[i] = b[i % GEMMA_DIM];
    quant_bebelm_matmul(&v, x, 40, y, scratch);
    for (size_t r = 0; r < 3; r++) {
        CHECK(quant_dequantize_row(QUANT_Q8_0, weights + r * 68, 68, row, 64, error));
        for (size_t t = 0; t < 40; t++) CHECK(y[t * 3 + r] == reference_dot(row, x + t * 64, 64));
    }
    uint8_t f32_weights[3 * 64 * 4];
    for (size_t i = 0; i < 3 * 64; i++) {
        float value = (float)((int)(i % 17) - 8) * 0.314159f;
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        for (unsigned j = 0; j < 4; j++) f32_weights[i * 4 + j] = (uint8_t)(bits >> (8 * j));
    }
    v.type = QUANT_F32; v.data = f32_weights; v.row_stride = 64 * 4;
    quant_bebelm_matmul(&v, x, 40, y, scratch);
    for (size_t r = 0; r < 3; r++) {
        CHECK(quant_dequantize_row(QUANT_F32, f32_weights + r * 256, 256, row, 64, error));
        for (size_t t = 0; t < 40; t++) CHECK(y[t * 3 + r] == reference_dot(row, x + t * 64, 64));
    }
    CHECK(!gemma_workspace_new(0, error));
    CHECK(!gemma_workspace_new(GEMMA_CONTEXT + 1, error));
    gemma_workspace *w = gemma_workspace_new(4, error);
    CHECK(w);
    int32_t ids[] = {-1, 2, 1, 2};
    gemma_sequence seq = {0, 4};
    CHECK(!gemma_forward(NULL, w, ids, 0, &seq, 1, output, error));
    CHECK(!gemma_forward(NULL, w, ids, 5, &seq, 1, output, error));
    seq.first = 1;
    CHECK(!gemma_forward(NULL, w, ids, 4, &seq, 1, output, error));
    seq.first = 0; seq.count = 5;
    CHECK(!gemma_forward(NULL, w, ids, 4, &seq, 1, output, error));
    seq.count = 3;
    CHECK(!gemma_forward(NULL, w, ids, 4, &seq, 1, output, error));
    seq.count = 4;
    CHECK(!gemma_forward(NULL, w, ids, 4, &seq, 1, output, error));
    CHECK(strstr(error, "token id out of range"));
    gemma_workspace_free(w);
    gemma_workspace_free(NULL);
    const gemma_model *m;
    CHECK(!gemma_model_cached("build/fixtures/absent.gguf", &m, error) && !m);
    CHECK(!gemma_model_cached("build/fixtures", &m, error) && !m);
    CHECK(!gemma_model_cached("build/fixtures/valid.gguf", &m, error) && !m);
    const char *fixtures[] = {"build/fixtures/gemma_shape.gguf", "build/fixtures/gemma_type.gguf", "build/fixtures/gemma_missing.gguf"};
    for (unsigned i = 0; i < 3; i++) {
        CHECK(!gemma_model_cached(fixtures[i], &m, error) && !m);
        CHECK(strstr(error, "tensor token_embd.weight"));
    }
    puts("EmbeddingGemma admission and numeric contracts passed");
    return 0;
}
