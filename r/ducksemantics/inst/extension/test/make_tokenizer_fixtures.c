/* Small GGUF tokenizer fixtures. SPDX-License-Identifier: MIT */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void bytes(FILE *out, const void *p, size_t n) {
    if (fwrite(p, 1, n, out) != n) { perror("write fixture"); exit(1); }
}

static void u32(FILE *out, uint32_t v) {
    uint8_t b[4];
    for (unsigned i = 0; i < 4; i++) b[i] = (uint8_t)(v >> (i * 8));
    bytes(out, b, sizeof(b));
}

static void u64(FILE *out, uint64_t v) {
    uint8_t b[8];
    for (unsigned i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (i * 8));
    bytes(out, b, sizeof(b));
}

static void string(FILE *out, const char *s) { size_t n = strlen(s); u64(out, n); bytes(out, s, n); }
static void key(FILE *out, const char *s, uint32_t type) { string(out, s); u32(out, type); }
static void str_value(FILE *out, const char *name, const char *s) { key(out, name, 8); string(out, s); }
static void int_value(FILE *out, const char *name, uint32_t n) { key(out, name, 4); u32(out, n); }
static void bool_value(FILE *out, const char *name, int value) { key(out, name, 7); uint8_t b = (uint8_t)value; bytes(out, &b, 1); }
static void array(FILE *out, const char *name, uint32_t type, uint64_t n) { key(out, name, 9); u32(out, type); u64(out, n); }

static FILE *begin(const char *dir, const char *name, uint64_t n_kv) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s.gguf", dir, name);
    if (n < 0 || (size_t)n >= sizeof(path)) { fprintf(stderr, "fixture path too long\n"); exit(1); }
    FILE *out = fopen(path, "wb");
    if (!out) { perror(path); exit(1); }
    bytes(out, "GGUF", 4); u32(out, 3); u64(out, 0); u64(out, n_kv);
    return out;
}

static void finish(FILE *out) {
    long pos = ftell(out);
    if (pos < 0) { perror("ftell fixture"); exit(1); }
    uint8_t zero = 0;
    while (pos++ % 32) bytes(out, &zero, 1);
    if (fclose(out)) { perror("close fixture"); exit(1); }
}

static void byte_char(uint32_t cp, char s[3]) {
    if (cp < 128) { s[0] = (char)cp; s[1] = '\0'; }
    else { s[0] = (char)(0xc0 | cp >> 6); s[1] = (char)(0x80 | (cp & 63)); s[2] = '\0'; }
}

static void lfm(const char *dir, const char *name, unsigned bad) {
    static const char *const extra[] = {"bc", "ab", "abc", "aa", "aaaa", "12", "123",
                                       "<|b|>", "<|e|>", "<|x|>", "<|x|>long"};
    static const char *const merges[] = {"b c", "a b", "a bc", "a a", "aa aa", "1 2", "12 3"};
    FILE *out = begin(dir, name, bad == 1 ? 7 : 8);
    str_value(out, "general.architecture", "lfm2");
    str_value(out, "tokenizer.ggml.model", "gpt2");
    str_value(out, "tokenizer.ggml.pre", "lfm2");
    array(out, "tokenizer.ggml.tokens", 8, 267);
    unsigned extra_cp = 0;
    for (unsigned b = 0; b < 256; b++) {
        unsigned cp = b;
        if (!(b >= 33 && b <= 126) && !(b >= 161 && b <= 172) && !(b >= 174)) cp = 256 + extra_cp++;
        char s[3];
        byte_char(cp, s);
        string(out, s);
    }
    for (unsigned i = 0; i < 11; i++) string(out, bad == 4 && !i ? "\xff" : extra[i]);
    if (bad != 1) {
        array(out, "tokenizer.ggml.token_type", 5, bad == 5 ? 266 : 267);
        for (unsigned i = 0; i < (bad == 5 ? 266U : 267U); i++) u32(out, i >= 263 ? 3 : 1);
    }
    array(out, "tokenizer.ggml.merges", 8, 7);
    for (unsigned i = 0; i < 7; i++) string(out, bad == 3 && !i ? "b absent" : merges[i]);
    int_value(out, "tokenizer.ggml.bos_token_id", bad == 2 ? 999 : 263);
    int_value(out, "tokenizer.ggml.eos_token_id", 264);
    finish(out);
}

static void gemma(const char *dir, const char *name, unsigned bad) {
    static const char *const tokens[] = {"<pad>", "<eos>", "<bos>", "<unk>", "a", "b", "c",
                                         "ab", "bc", "abc", "▁", "z", "é", "<0x61>", "zz", "za"};
    FILE *out = begin(dir, name, 11);
    str_value(out, "general.architecture", "gemma-embedding");
    str_value(out, "tokenizer.ggml.model", "llama");
    array(out, "tokenizer.ggml.tokens", 8, 272);
    for (unsigned i = 0; i < 16; i++) string(out, tokens[i]);
    for (unsigned b = 0; b < 256; b++) {
        char token[7];
        snprintf(token, sizeof(token), "<0x%02X>", b);
        string(out, token);
    }
    array(out, "tokenizer.ggml.token_type", 5, 272);
    for (unsigned i = 0; i < 272; i++) u32(out, i < 3 ? 3 : i == 3 ? 2 : i < 16 ? 1 : 6);
    array(out, "tokenizer.ggml.scores", 6, 272);
    for (unsigned i = 0; i < 272; i++) {
        float score = i == 7 ? 1.0f : i == 8 ? 2.0f : i == 9 ? 3.0f : 0.0f;
        uint32_t bits;
        memcpy(&bits, &score, 4);
        if (i == 14) bits = 0x80000000U;
        if (bad == 1 && i == 4) bits = 0x7fc00000U;
        u32(out, bits);
    }
    int_value(out, "tokenizer.ggml.bos_token_id", 2);
    int_value(out, "tokenizer.ggml.eos_token_id", 1);
    int_value(out, "tokenizer.ggml.unknown_token_id", 3);
    int_value(out, "tokenizer.ggml.padding_token_id", 0);
    bool_value(out, "tokenizer.ggml.add_bos_token", bad != 2);
    bool_value(out, "tokenizer.ggml.add_eos_token", bad != 2);
    finish(out);
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s fixture-directory\n", argv[0]); return 1; }
    lfm(argv[1], "tokenizer_lfm", 0);
    lfm(argv[1], "tokenizer_missing_types", 1);
    lfm(argv[1], "tokenizer_bad_bos", 2);
    lfm(argv[1], "tokenizer_bad_merge", 3);
    lfm(argv[1], "tokenizer_bad_utf8", 4);
    lfm(argv[1], "tokenizer_bad_lengths", 5);
    gemma(argv[1], "tokenizer_gemma", 0);
    gemma(argv[1], "tokenizer_bad_score", 1);
    gemma(argv[1], "tokenizer_no_special", 2);
    return 0;
}
