/* Closed-profile admission fixtures; tensor payloads are sparse. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void u32(FILE *f, uint32_t v) {
    for (unsigned i = 0; i < 4; i++) fputc((int)(v >> (8 * i) & 255), f);
}
static void u64(FILE *f, uint64_t v) {
    for (unsigned i = 0; i < 8; i++) fputc((int)(v >> (8 * i) & 255), f);
}
static void text(FILE *f, const char *s) {
    u64(f, strlen(s));
    fwrite(s, 1, strlen(s), f);
}
static void key_uint(FILE *f, const char *key, uint32_t value) {
    text(f, key); u32(f, 4); u32(f, value);
}
static void key_float(FILE *f, const char *key, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    text(f, key); u32(f, 6); u32(f, bits);
}
static void key_bool(FILE *f, const char *key) {
    text(f, key); u32(f, 7); fputc(1, f);
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    static const struct { const char *key; uint32_t value; } integers[] = {
        {"context_length", 2048}, {"embedding_length", 768}, {"block_count", 24},
        {"feed_forward_length", 1152}, {"attention.head_count", 3},
        {"attention.head_count_kv", 1}, {"attention.key_length", 256},
        {"attention.value_length", 256}, {"attention.sliding_window", 512},
        {"dense_2_feat_in", 768}, {"dense_2_feat_out", 3072},
        {"dense_3_feat_in", 3072}, {"dense_3_feat_out", 768}, {"pooling_type", 1}
    };
    size_t ni = sizeof(integers) / sizeof(integers[0]);
    for (unsigned mode = 0; mode < 3; mode++) {
        char path[1024];
        const char *names[] = {"gemma_shape.gguf", "gemma_type.gguf", "gemma_missing.gguf"};
        if (snprintf(path, sizeof(path), "%s/%s", argv[1], names[mode]) >= (int)sizeof(path)) return 1;
        FILE *f = fopen(path, "wb");
        if (!f) return 1;
        fwrite("GGUF", 1, 4, f); u32(f, 3); u64(f, 316); u64(f, ni + 10);
        text(f, "general.architecture"); u32(f, 8); text(f, "gemma-embedding");
        for (size_t i = 0; i < ni; i++) {
            char key[128];
            snprintf(key, sizeof(key), "gemma-embedding.%s", integers[i].key);
            key_uint(f, key, integers[i].value);
        }
        key_float(f, "gemma-embedding.rope.freq_base", 1000000.0f);
        key_float(f, "gemma-embedding.rope.freq_base_swa", 10000.0f);
        key_float(f, "gemma-embedding.attention.layer_norm_rms_epsilon", 1e-6f);
        key_uint(f, "tokenizer.ggml.bos_token_id", 2);
        key_uint(f, "tokenizer.ggml.eos_token_id", 1);
        key_uint(f, "tokenizer.ggml.padding_token_id", 0);
        key_bool(f, "tokenizer.ggml.add_bos_token");
        key_bool(f, "tokenizer.ggml.add_eos_token");
        text(f, "tokenizer.ggml.tokens"); u32(f, 9); u32(f, 8); u64(f, 262144);
        for (size_t i = 0; i < 262144; i++) u64(f, 0);
        for (unsigned i = 0; i < 316; i++) {
            char name[64];
            if (!i && mode < 2) snprintf(name, sizeof(name), "token_embd.weight");
            else snprintf(name, sizeof(name), "dummy_%u.weight", i);
            text(f, name); u32(f, 2);
            u64(f, !i && mode == 1 ? 768 : 32);
            u64(f, !i && mode == 1 ? 262144 : 1);
            u32(f, !i && mode == 1 ? 1 : 0); u64(f, 0);
        }
        off_t pos = ftello(f);
        if (pos < 0) return 1;
        off_t data = (pos + 31) & ~(off_t)31;
        off_t payload = mode == 1 ? (off_t)768 * 262144 * 2 : 128;
        if (fflush(f) || ftruncate(fileno(f), data + payload) || fclose(f)) return 1;
    }
    return 0;
}
