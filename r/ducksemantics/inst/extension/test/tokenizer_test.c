/* Tokenizer boundary and cache tests. SPDX-License-Identifier: MIT */
#include "../src/tokenizer.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static void check(bool ok, const char *label) {
    if (!ok) { fprintf(stderr, "tokenizer test: %s\n", label); exit(1); }
}

static const tokenizer *load(const char *path, char *error) {
    const tokenizer *t = tokenizer_cached(path, error);
    check(t != NULL, error);
    return t;
}

typedef struct {
    const char *path;
    const tokenizer *value;
} cache_request;

static void *cache_worker(void *arg) {
    cache_request *request = arg;
    char error[GGUF_ERROR_SIZE];
    for (unsigned i = 0; i < 100; i++) {
        const tokenizer *t = load(request->path, error);
        if (i) check(t == request->value, "stable concurrent cache identity");
        request->value = t;
        int32_t *ids;
        size_t count;
        check(tokenizer_encode(t, "abc", 3, true, &ids, &count, error), error);
        check(count == 2 && ids[0] == 263 && ids[1] == 258, "concurrent encode");
        free(ids);
    }
    return NULL;
}

int main(int argc, char **argv) {
    check(argc == 2, "usage: tokenizer_test fixture-directory");
    char path[4096], error[GGUF_ERROR_SIZE];
    check(snprintf(path, sizeof(path), "%s/tokenizer_lfm.gguf", argv[1]) < (int)sizeof(path), "fixture path");
    pthread_t threads[8];
    cache_request requests[8];
    for (unsigned i = 0; i < 8; i++) {
        requests[i] = (cache_request){path, NULL};
        check(!pthread_create(&threads[i], NULL, cache_worker, &requests[i]), "pthread_create");
    }
    for (unsigned i = 0; i < 8; i++) {
        check(!pthread_join(threads[i], NULL), "pthread_join");
        check(requests[i].value == requests[0].value, "one loaded object per cache identity");
    }
    const tokenizer *t = requests[0].value;
    static const unsigned char invalid[][4] = {
        {0x80, 0, 0, 0}, {0xc0, 0xaf, 0, 0}, {0xe0, 0x80, 0xaf, 0},
        {0xed, 0xa0, 0x80, 0}, {0xf4, 0x90, 0x80, 0x80}, {0xf5, 0x80, 0x80, 0x80},
        {0xe2, 0x82, 0, 0}, {0xc2, 0, 0, 0}
    };
    static const size_t sizes[] = {1, 2, 3, 3, 4, 4, 2, 1};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        int32_t *ids;
        size_t count;
        check(!tokenizer_encode(t, (const char *)invalid[i], sizes[i], false, &ids, &count, error), "reject invalid UTF-8");
        check(!strcmp(error, "invalid UTF-8 input") && !ids && !count, "invalid input error state");
    }
    const char with_nul[] = {'a', 0, 'b'};
    int32_t *ids;
    size_t count, len;
    char *decoded;
    check(tokenizer_encode(t, with_nul, sizeof(with_nul), false, &ids, &count, error), error);
    check(tokenizer_decode(t, ids, count, &decoded, &len, error), error);
    check(len == sizeof(with_nul) && !memcmp(decoded, with_nul, len), "embedded NUL round-trip");
    free(decoded); free(ids);
    int32_t partial[] = {0xe2, 0x82, 'a', 0xed, 0xa0, 0x80};
    const char lossy[] = "\xef\xbf\xbd" "a" "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd";
    check(tokenizer_decode(t, partial, 6, &decoded, &len, error), error);
    check(len == sizeof(lossy) - 1 && !memcmp(decoded, lossy, len), "lossy decode subsequences");
    free(decoded);
    int32_t negative = -1;
    check(!tokenizer_decode(t, &negative, 1, &decoded, &len, error), "negative token id");
    struct stat st;
    check(!stat(path, &st), "stat fixture");
    struct timespec times[2] = {st.st_atim, st.st_mtim};
    times[1].tv_sec++;
    check(!utimensat(AT_FDCWD, path, times, 0), "change cache mtime");
    const tokenizer *new_t = load(path, error);
    check(new_t != t, "mtime invalidates cache key");
    FILE *append = fopen(path, "ab");
    check(append != NULL, "open fixture for append");
    check(fputc(0, append) != EOF && !fclose(append), "append fixture padding");
    check(!utimensat(AT_FDCWD, path, times, 0), "preserve mtime after size change");
    check(load(path, error) != new_t, "file size invalidates cache key");
    check(tokenizer_encode(t, "abc", 3, false, &ids, &count, error), "old mapping remains live");
    free(ids);
    puts("ok tokenizer: UTF-8, NUL, lossy decode, concurrent cache, size/mtime identity");
    return 0;
}
