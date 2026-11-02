/* Batched LFM2-MoE prefill parity against the per-token scalar oracle.
 * SPDX-License-Identifier: MIT
 * Usage: lfm2_prefill_test model.gguf [parity|bench|all]
 */
#define _POSIX_C_SOURCE 200809L
#include "lfm2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CAPACITY 640
#define DECODE 4

static const size_t sizes[] = {1, 2, 7, 16, 32, 64};
static const size_t nsizes = sizeof(sizes) / sizeof(sizes[0]);
static const int32_t probe[] = {0, 1, 16, 1000, 50000, 124893, 124894, 124900, 124901, 124902, 127999};
static const size_t nprobe = sizeof(probe) / sizeof(probe[0]);
static int failures;

static void die(const char *error) { fprintf(stderr, "error: %s\n", error); exit(1); }

static void check(bool ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); failures++; }
}

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static double max_diff(const float *a, const float *b, size_t n) {
    double d = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)a[i] - (double)b[i]);
        if (e > d || isnan(e)) d = isnan(e) ? INFINITY : e;
    }
    return d;
}

/* Maximum absolute difference over every valid K/V row, conv state and the
 * final norm; positions must agree. Exact parity reports 0. */
static double cache_diff(const lfm2_state *a, const lfm2_state *b) {
    lfm2_cache x = {0}, y = {0};
    char error[GGUF_ERROR_SIZE];
    if (!lfm2_state_cache_snapshot(&x, a, error)) die(error);
    if (!lfm2_state_cache_snapshot(&y, b, error)) {
        lfm2_cache_free(&x);
        die(error);
    }

    double d = INFINITY;
    if (x.position == y.position) {
        d = max_diff(x.norm, y.norm, LFM2_MOE_HIDDEN);
        for (size_t i = 0; i < LFM2_MOE_LAYERS; i++) {
            d = fmax(d, max_diff(x.conv[i], y.conv[i], 2 * LFM2_MOE_HIDDEN));
            if (!x.k[i] != !y.k[i]) { d = INFINITY; break; }
            if (x.k[i]) {
                d = fmax(d, max_diff(x.k[i], y.k[i], x.position * LFM2_MOE_KV_DIM));
                d = fmax(d, max_diff(x.v[i], y.v[i], x.position * LFM2_MOE_KV_DIM));
            }
        }
    }
    lfm2_cache_free(&y);
    lfm2_cache_free(&x);
    return d;
}

static double logit_diff(lfm2_state *a, lfm2_state *b) {
    float x[sizeof(probe) / sizeof(probe[0])], y[sizeof(probe) / sizeof(probe[0])];
    char error[GGUF_ERROR_SIZE];
    if (!lfm2_state_logits(a, probe, nprobe, x, error) || !lfm2_state_logits(b, probe, nprobe, y, error)) die(error);
    return max_diff(x, y, nprobe);
}

/* Greedy continuation must agree in ids, margins, terminal and text. */
static bool same_greedy(const lfm2_generation *a, const lfm2_generation *b) {
    return a->count == b->count && a->stopped == b->stopped && a->terminal_id == b->terminal_id &&
           a->terminal_margin == b->terminal_margin && a->decode_tokens == b->decode_tokens &&
           !memcmp(a->ids, b->ids, a->count * sizeof(*a->ids)) &&
           !memcmp(a->margins, b->margins, a->count * sizeof(*a->margins)) &&
           a->bytes == b->bytes && !memcmp(a->text, b->text, a->bytes);
}

static void decode_pair(lfm2_state *a, lfm2_state *b, const char *label) {
    char error[GGUF_ERROR_SIZE], what[160];
    lfm2_generation ga, gb;
    if (!lfm2_decode(a, DECODE, &ga, error) || !lfm2_decode(b, DECODE, &gb, error)) die(error);
    snprintf(what, sizeof(what), "%s greedy continuation", label);
    check(same_greedy(&ga, &gb), what);
    snprintf(what, sizeof(what), "%s decode forward count", label);
    check(ga.decode_tokens == (ga.stopped ? ga.count : ga.count - 1), what);
    printf("  %s greedy %zu ids%s, first id %d margin %.6g\n", label, ga.count,
           ga.stopped ? " (stopped)" : "", ga.count ? ga.ids[0] : ga.terminal_id,
           ga.count ? ga.margins[0] : ga.terminal_margin);
    lfm2_generation_free(&ga); lfm2_generation_free(&gb);
}

static int32_t *prompt_ids(const lfm2_model *m, const char *text, bool bos, size_t *count) {
    char error[GGUF_ERROR_SIZE]; int32_t *ids = NULL;
    if (!lfm2_prompt_ids(m, text, strlen(text), bos, &ids, count, error)) die(error);
    return ids;
}

static lfm2_state *prefilled(const lfm2_model *m, const int32_t *ids, size_t count, size_t batch) {
    char error[GGUF_ERROR_SIZE];
    lfm2_state *s = lfm2_state_new(m, CAPACITY, error);
    if (!s) die(error);
    bool ok = batch ? lfm2_prefill_batch(s, ids, count, batch, error) : lfm2_prefill_scalar(s, ids, count, error);
    if (!ok) die(error);
    return s;
}

static void compare(lfm2_state *scalar, lfm2_state *batched, const char *label) {
    char what[160];
    double cache = cache_diff(scalar, batched), logits = logit_diff(scalar, batched);
    printf("  %-34s cache max|d| %.3g  logits max|d| %.3g\n", label, cache, logits);
    snprintf(what, sizeof(what), "%s cache parity", label); check(cache == 0, what);
    snprintf(what, sizeof(what), "%s logit parity", label); check(logits == 0, what);
}

static const char *prompts[] = {
    "<|im_start|>user\nHi<|im_end|>\n",
    "<|im_start|>system\nYou answer biomedical questions carefully and briefly.<|im_end|>\n"
    "<|im_start|>user\nWhat does BRCA1 do in DNA repair?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a careful clinical genetics assistant. Cite the evidence type, "
    "state uncertainty explicitly, and avoid speculation beyond the provided context.<|im_end|>\n"
    "<|im_start|>user\nA patient carries a heterozygous CFTR F508del variant and a second variant of "
    "uncertain significance in trans. Summarize how segregation data, functional assays and population "
    "frequency would each change the classification, in three short sentences.<|im_end|>\n"
    "<|im_start|>assistant\n"
};

static void parity(const lfm2_model *m) {
    char error[GGUF_ERROR_SIZE], label[96];
    const char *tails[2] = {"Explain BRCA1 variants.<|im_end|>\n<|im_start|>assistant\n",
                            "Explain CFTR variants in one sentence, please.<|im_end|>\n<|im_start|>assistant\n"};
    for (size_t p = 0; p < sizeof(prompts) / sizeof(prompts[0]); p++) {
        size_t n; int32_t *ids = prompt_ids(m, prompts[p], true, &n);
        printf("prompt %zu: %zu tokens\n", p, n);
        lfm2_state *scalar = prefilled(m, ids, n, 0);
        for (size_t b = 0; b < nsizes; b++) {
            lfm2_state *batched = prefilled(m, ids, n, sizes[b]);
            snprintf(label, sizeof(label), "batch %zu", sizes[b]);
            compare(scalar, batched, label);
            lfm2_state_free(batched);
        }
        lfm2_state *batched = prefilled(m, ids, n, 7);
        decode_pair(scalar, batched, "batch 7 vs scalar");
        lfm2_state_free(batched); lfm2_state_free(scalar); free(ids);
    }

    /* Nonzero prefix extended and forked: scalar prefix + scalar tail against
     * every combination of batched prefix/tail on forked branches. */
    size_t np; int32_t *prefix = prompt_ids(m, prompts[1], true, &np);
    lfm2_state *base_scalar = prefilled(m, prefix, np, 0), *base_batch = prefilled(m, prefix, np, 16);
    for (size_t branch = 0; branch < 2; branch++) {
        size_t nt; int32_t *tail = prompt_ids(m, tails[branch], false, &nt);
        printf("branch %zu: %zu prefix + %zu tail tokens\n", branch, np, nt);
        lfm2_state *oracle = lfm2_state_fork(base_scalar, error);
        if (!oracle || !lfm2_prefill_scalar(oracle, tail, nt, error)) die(error);
        for (size_t b = 0; b < nsizes; b++) {
            lfm2_state *fork_s = lfm2_state_fork(base_scalar, error), *fork_b = lfm2_state_fork(base_batch, error);
            if (!fork_s || !fork_b || !lfm2_prefill_batch(fork_s, tail, nt, sizes[b], error) ||
                !lfm2_prefill_batch(fork_b, tail, nt, sizes[b], error)) die(error);
            snprintf(label, sizeof(label), "scalar prefix, batch %zu tail", sizes[b]); compare(oracle, fork_s, label);
            snprintf(label, sizeof(label), "batch 16 prefix, batch %zu tail", sizes[b]); compare(oracle, fork_b, label);
            lfm2_state_free(fork_s); lfm2_state_free(fork_b);
        }
        lfm2_state *fork_b = lfm2_state_fork(base_batch, error);
        if (!fork_b || !lfm2_prefill(fork_b, tail, nt, error)) die(error);
        decode_pair(oracle, fork_b, "fork default route vs scalar");
        lfm2_state_free(fork_b); lfm2_state_free(oracle); free(tail);
    }

    /* Rejected admission leaves the cache untouched. */
    lfm2_state *snapshot = lfm2_state_fork(base_batch, error);
    if (!snapshot) die(error);
    int32_t bad_last[3] = {16, 17, 128000}, bad_first[2] = {-1, 16};
    int32_t *many = calloc(CAPACITY, sizeof(int32_t));
    if (!many) die("out of memory for admission test");
    check(!lfm2_prefill_batch(base_batch, prefix, np, 0, error), "batch size 0 rejected");
    check(!lfm2_prefill_batch(base_batch, prefix, np, LFM2_PREFILL_MAX_BATCH + 1, error), "batch size above bound rejected");
    check(!lfm2_prefill_batch(base_batch, bad_last, 3, 2, error), "trailing invalid id rejected");
    check(!lfm2_prefill_batch(base_batch, bad_first, 2, 2, error), "negative id rejected");
    check(!lfm2_prefill_scalar(base_batch, bad_last, 3, error), "scalar invalid id rejected");
    check(!lfm2_prefill(base_batch, many, CAPACITY - np + 1, error), "capacity overflow rejected");
    check(lfm2_prefill_batch(base_batch, many, 0, 16, error), "empty prefill accepted");
    check(cache_diff(snapshot, base_batch) == 0, "rejected admission leaves state unchanged");
    printf("admission: invalid batch size, ids and capacity rejected without cache mutation\n");
    free(many); lfm2_state_free(snapshot);
    lfm2_state_free(base_scalar); lfm2_state_free(base_batch); free(prefix);

    /* Generation counters and timing boundaries. */
    lfm2_generation one, two;
    if (!lfm2_generate_batch(m, prompts[2], strlen(prompts[2]), DECODE, 1, &one, error) ||
        !lfm2_generate(m, prompts[2], strlen(prompts[2]), DECODE, &two, error)) die(error);
    check(same_greedy(&one, &two), "generate batch 1 and default agree");
    check(two.prefill_batch_size == LFM2_PREFILL_BATCH &&
          two.prefill_batches == (two.prompt_tokens + LFM2_PREFILL_BATCH - 1) / LFM2_PREFILL_BATCH, "prefill batch counters");
    check(one.prefill_batches == one.prompt_tokens, "batch 1 runs one chunk per token");
    check(two.total_seconds >= two.prefill_seconds + two.decode_seconds, "total spans prefill and decode");
    printf("generate: %zu prompt tokens in %zu batches of %zu, %zu ids, %zu decode forwards, "
           "prefill %.3fs decode %.3fs total %.3fs\n", two.prompt_tokens, two.prefill_batches,
           two.prefill_batch_size, two.count, two.decode_tokens, two.prefill_seconds,
           two.decode_seconds, two.total_seconds);
    lfm2_generation_free(&one); lfm2_generation_free(&two);
}

static void bench(const lfm2_model *m) {
    char error[GGUF_ERROR_SIZE], text[8192] = "";
    strcat(text, prompts[2]);
    size_t n; int32_t *ids = prompt_ids(m, text, true, &n);
    const size_t bench_sizes[] = {0, 16, 32, 64};
    lfm2_state *oracle = NULL;
    lfm2_state_free(prefilled(m, ids, n, 0));
    printf("bench: %zu prompt tokens after one warm-up scalar prefill, best of 2\n", n);
    for (size_t b = 0; b < sizeof(bench_sizes) / sizeof(bench_sizes[0]); b++) {
        double best = INFINITY;
        lfm2_state *s = NULL;
        for (int rep = 0; rep < 2; rep++) {
            lfm2_state_free(s);
            s = lfm2_state_new(m, CAPACITY, error);
            if (!s) die(error);
            double start = now();
            bool ok = bench_sizes[b] ? lfm2_prefill_batch(s, ids, n, bench_sizes[b], error) :
                                       lfm2_prefill_scalar(s, ids, n, error);
            double elapsed = now() - start;
            if (!ok) die(error);
            if (elapsed < best) best = elapsed;
        }
        if (!oracle) {
            oracle = s;
            printf("  scalar   %8.3fs %8.2f tok/s\n", best, (double)n / best);
            continue;
        }
        double cache = cache_diff(oracle, s), logits = logit_diff(oracle, s);
        printf("  batch %2zu %8.3fs %8.2f tok/s  cache max|d| %.3g logits max|d| %.3g\n",
               bench_sizes[b], best, (double)n / best, cache, logits);
        check(cache == 0 && logits == 0, "benchmark parity");
        lfm2_state_free(s);
    }
    lfm2_state_free(oracle); free(ids);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) die("usage: lfm2_prefill_test model.gguf [parity|bench|smoke|all]");
    const char *mode = argc == 3 ? argv[2] : "all";
    bool run_parity = !strcmp(mode, "parity") || !strcmp(mode, "all");
    bool run_bench = !strcmp(mode, "bench") || !strcmp(mode, "all");
    bool smoke = !strcmp(mode, "smoke");
    if (!run_parity && !run_bench && !smoke) die("mode must be parity, bench, smoke or all");
    setvbuf(stdout, NULL, _IOLBF, 0);
    char error[GGUF_ERROR_SIZE];
    double start = now();
    const lfm2_model *m = lfm2_model_get(argv[1], error);
    if (!m) die(error);
    printf("model %s loaded in %.2fs\n", argv[1], now() - start);
    if (run_parity) parity(m);
    if (run_bench) bench(m);
    if (smoke) {
        size_t n;
        int32_t *ids = prompt_ids(m, prompts[0], true, &n);
        lfm2_state *scalar = prefilled(m, ids, n, 0);
        lfm2_state *batched = prefilled(m, ids, n, 2);
        compare(scalar, batched, "smoke uneven batch 2");
        const int32_t tail[] = {4, 5, 6};
        if (!lfm2_prefill_scalar(scalar, tail, 3, error) ||
            !lfm2_prefill_batch(batched, tail, 3, 64, error)) die(error);
        compare(scalar, batched, "smoke prefixed batch 64");
        decode_pair(scalar, batched, "smoke");
        lfm2_state_free(scalar); lfm2_state_free(batched); free(ids);
    }
    if (failures) { fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    puts("all checks passed");
    return 0;
}
