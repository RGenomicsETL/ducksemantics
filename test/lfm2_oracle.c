/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include "../src/lfm2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

static void die(const char *error) { fprintf(stderr, "%s\n", error); exit(1); }

static char *read_prompt(const char *path, size_t *bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open oracle prompt");
    char *text = malloc(LFM2_MAX_INPUT_BYTES + 1);
    if (!text) die("out of memory for oracle prompt");
    *bytes = fread(text, 1, LFM2_MAX_INPUT_BYTES + 1, f);
    if (ferror(f) || *bytes > LFM2_MAX_INPUT_BYTES) die("oracle prompt exceeds byte limit");
    fclose(f); text[*bytes] = 0; return text;
}

static void emit(size_t index, const lfm2_generation *g) {
    struct rusage usage; getrusage(RUSAGE_SELF, &usage);
    printf("%zu\t%zu\t%zu\t%.9g\t%.9g\t%ld\t%d\t%d\t%.9g\t", index, g->prompt_tokens,
           g->count, g->prefill_seconds, g->decode_seconds, usage.ru_maxrss, g->stopped, g->terminal_id, g->terminal_margin);
    for (size_t i = 0; i < g->count; i++) printf("%s%d", i ? "," : "", g->ids[i]);
    putchar('\t');
    for (size_t i = 0; i < g->count; i++) printf("%s%.9g", i ? "," : "", g->margins[i]);
    putchar('\n'); fflush(stdout);
}

static void fork_test(const lfm2_model *m) {
    const char *prefix = "<|im_start|>system\nExplain biomedical evidence carefully.<|im_end|>\n<|im_start|>";
    const char *continuations[2] = {
        "user\nExplain BRCA1 variants.<|im_end|>\n<|im_start|>assistant\n",
        "user\nExplain CFTR variants.<|im_end|>\n<|im_start|>assistant\n"
    };
    char error[GGUF_ERROR_SIZE]; int32_t *prefix_ids = NULL; size_t prefix_count;
    if (!lfm2_prompt_ids(m, prefix, strlen(prefix), true, &prefix_ids, &prefix_count, error)) die(error);
    lfm2_state *shared = lfm2_state_new(m, 256, error);
    if (!shared || !lfm2_prefill(shared, prefix_ids, prefix_count, error)) die(error);
    for (size_t branch = 0; branch < 2; branch++) {
        char full[1024]; snprintf(full, sizeof(full), "%s%s", prefix, continuations[branch]);
        int32_t *tail = NULL, *all = NULL; size_t ntail, nall;
        if (!lfm2_prompt_ids(m, continuations[branch], strlen(continuations[branch]), false, &tail, &ntail, error) ||
            !lfm2_prompt_ids(m, full, strlen(full), true, &all, &nall, error)) die(error);
        if (nall != prefix_count + ntail || memcmp(prefix_ids, all, prefix_count * sizeof(*all)) ||
            memcmp(tail, all + prefix_count, ntail * sizeof(*all))) die("fork tokenization seam differs from cold prompt");
        lfm2_state *hot = lfm2_state_fork(shared, error), *cold = lfm2_state_new(m, 256, error);
        if (!hot || !cold || !lfm2_prefill(hot, tail, ntail, error) || !lfm2_prefill(cold, all, nall, error)) die(error);
        int32_t options[5] = {0, 16, 124900, 124901, 127999}; float a[5], b[5];
        if (!lfm2_state_logits(hot, options, 5, a, error) || !lfm2_state_logits(cold, options, 5, b, error)) die(error);
        if (memcmp(a, b, sizeof(a))) die("fork logits differ from cold run");
        lfm2_generation gh, gc;
        if (!lfm2_decode(hot, 8, &gh, error) || !lfm2_decode(cold, 8, &gc, error)) die(error);
        if (gh.count != gc.count || memcmp(gh.ids, gc.ids, gh.count * sizeof(*gh.ids)) ||
            gh.bytes != gc.bytes || memcmp(gh.text, gc.text, gh.bytes)) die("fork generation differs from cold run");
        fprintf(stderr, "fork %zu: %zu shared + %zu appended; logits exact; %zu generated ids exact\n",
                branch + 1, prefix_count, ntail, gh.count);
        lfm2_generation_free(&gh); lfm2_generation_free(&gc); lfm2_state_free(hot); lfm2_state_free(cold);
        free(tail); free(all);
    }
    free(prefix_ids); lfm2_state_free(shared);
}

int main(int argc, char **argv) {
    if (argc != 5) die("usage: lfm2_oracle model prompt_directory prompt_count max_tokens");
    char error[GGUF_ERROR_SIZE]; const lfm2_model *model = lfm2_model_get(argv[1], error);
    if (!model) die(error);
    size_t count = (size_t)strtoul(argv[3], NULL, 10), max_tokens = (size_t)strtoul(argv[4], NULL, 10);
    if (!count || count > 1000 || !max_tokens || max_tokens > LFM2_MAX_GENERATED) die("invalid oracle sizes");
    puts("case\tprompt_tokens\ttokens\tprefill_seconds\tdecode_seconds\trss_kib\tstopped\tterminal_id\tterminal_margin\tids\tmargins");
    for (size_t i = 1; i <= count; i++) {
        char path[4096];
        if (snprintf(path, sizeof(path), "%s/%zu.txt", argv[2], i) >= (int)sizeof(path)) die("oracle prompt path too long");
        size_t bytes; char *prompt = read_prompt(path, &bytes); lfm2_generation result;
        if (!lfm2_generate(model, prompt, bytes, max_tokens, &result, error)) die(error);
        emit(i, &result); free(prompt); lfm2_generation_free(&result);
    }
    fork_test(model);
    return 0;
}
