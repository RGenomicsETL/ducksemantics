/* GGUF-driven LFM2 and EmbeddingGemma tokenizers. SPDX-License-Identifier: MIT */
#include "tokenizer.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "tokenizer_unicode.h"

#define HASH_BASE UINT64_C(257)
#define NO_NODE SIZE_MAX

typedef struct {
    gguf_str text;
    uint64_t hash;
    float score;
    uint32_t type;
} token;

typedef struct {
    int32_t left, right, result;
    uint32_t rank;
    bool used;
} merge;

struct tokenizer {
    bool gemma, add_bos, add_eos;
    int32_t bos, eos, unknown;
    token *tokens;
    size_t n_tokens, table_size, merge_size;
    int32_t *table;
    merge *merges;
    int32_t bytes[256];
    uint32_t byte_chars[256];
    int16_t byte_decoder[324];
    size_t special_offsets[257];
    int32_t *specials;
    size_t n_specials;
};

typedef struct {
    size_t start, len, prev, next, version;
    uint64_t hash, power;
    int32_t id;
    bool live;
} symbol;

typedef struct {
    size_t left, right, version, right_version;
    int32_t result;
    double priority;
} candidate;

typedef struct {
    symbol *nodes;
    candidate *heap;
    size_t heap_count, capacity;
    const char *text;
} bpe;

static bool fail(char *error, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, GGUF_ERROR_SIZE, fmt, ap);
    va_end(ap);
    return false;
}

static void *array_alloc(size_t n, size_t width, char *error) {
    if (n > SIZE_MAX / width) {
        fail(error, "tokenizer allocation overflows");
        return NULL;
    }
    void *p = calloc(n ? n : 1, width);
    if (!p) fail(error, "out of memory in tokenizer");
    return p;
}

static uint64_t text_hash(const char *s, size_t n) {
    uint64_t h = 0;
    for (size_t i = 0; i < n; i++) h = h * HASH_BASE + (uint8_t)s[i] + 1;
    return h;
}

static size_t hash_slot(uint64_t h, size_t size) {
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33;
    return (size_t)h & (size - 1);
}

static bool table_capacity(size_t n, size_t *out, char *error) {
    size_t size = 16;
    if (n > SIZE_MAX / 2) return fail(error, "tokenizer table overflows");
    while (size < n * 2) {
        if (size > SIZE_MAX / 2) return fail(error, "tokenizer table overflows");
        size *= 2;
    }
    *out = size;
    return true;
}

static int32_t token_lookup(const tokenizer *t, const char *s, size_t n, uint64_t hash) {
    size_t slot = hash_slot(hash, t->table_size);
    for (;;) {
        int32_t id = t->table[slot];
        if (id < 0) return -1;
        const token *v = &t->tokens[id];
        if (v->hash == hash && v->text.len == n && !memcmp(v->text.ptr, s, n)) return id;
        slot = (slot + 1) & (t->table_size - 1);
    }
}

static size_t merge_slot(const tokenizer *t, int32_t left, int32_t right) {
    return hash_slot((uint64_t)(uint32_t)left << 32 | (uint32_t)right, t->merge_size);
}

static const merge *merge_lookup(const tokenizer *t, int32_t left, int32_t right) {
    size_t slot = merge_slot(t, left, right);
    while (t->merges[slot].used) {
        const merge *m = &t->merges[slot];
        if (m->left == left && m->right == right) return m;
        slot = (slot + 1) & (t->merge_size - 1);
    }
    return NULL;
}

/* Reject overlong encodings, surrogates and values above U+10FFFF. */
static bool utf8_next(const char *s, size_t n, size_t *at, uint32_t *cp) {
    size_t i = *at;
    if (i >= n) return false;
    uint8_t c = (uint8_t)s[i++];
    unsigned tail;
    uint32_t value, min;
    if (c < 0x80) { *cp = c; *at = i; return true; }
    if (c >= 0xc2 && c <= 0xdf) { tail = 1; value = c & 31; min = 0x80; }
    else if (c >= 0xe0 && c <= 0xef) { tail = 2; value = c & 15; min = 0x800; }
    else if (c >= 0xf0 && c <= 0xf4) { tail = 3; value = c & 7; min = 0x10000; }
    else return false;
    if (tail > n - i) return false;
    for (unsigned k = 0; k < tail; k++) {
        c = (uint8_t)s[i++];
        if ((c & 0xc0) != 0x80) return false;
        value = value << 6 | (c & 63);
    }
    if (value < min || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return false;
    *cp = value;
    *at = i;
    return true;
}

static bool utf8_valid(const char *s, size_t n) {
    size_t at = 0;
    uint32_t cp;
    while (at < n) if (!utf8_next(s, n, &at, &cp)) return false;
    return true;
}

static size_t utf8_put(uint32_t cp, char *s) {
    if (cp < 0x80) { s[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        s[0] = (char)(0xc0 | cp >> 6); s[1] = (char)(0x80 | (cp & 63)); return 2;
    }
    if (cp < 0x10000) {
        s[0] = (char)(0xe0 | cp >> 12); s[1] = (char)(0x80 | ((cp >> 6) & 63));
        s[2] = (char)(0x80 | (cp & 63)); return 3;
    }
    s[0] = (char)(0xf0 | cp >> 18); s[1] = (char)(0x80 | ((cp >> 12) & 63));
    s[2] = (char)(0x80 | ((cp >> 6) & 63)); s[3] = (char)(0x80 | (cp & 63)); return 4;
}

static bool in_ranges(uint32_t cp, const uint32_t ranges[][2], size_t n) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid][0]) hi = mid;
        else if (cp > ranges[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

#define PROPERTY(cp, name) in_ranges(cp, unicode_##name, sizeof(unicode_##name) / sizeof(unicode_##name[0]))

typedef union {
    gguf_str s;
    uint32_t u;
    int32_t i;
    float f;
    bool b;
} gguf_scalar;

/* Only offsets issued by the validated GGUF reader reach this accessor. */
static bool gguf_scalar_read(const gguf_file *f, uint32_t type, uint64_t pos, gguf_scalar *v) {
    if (type == GGUF_STRING) return gguf_scalar_text(f, type, pos, NULL, 0, &v->s);
    size_t width = type == GGUF_BOOL ? 1 : 4;
    if (pos > f->size || width > f->size - pos) return false;
    const uint8_t *p = f->map + pos;
    if (type == GGUF_BOOL) { v->b = p[0] != 0; return true; }
    uint32_t bits = (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    if (type == GGUF_UINT32) v->u = bits;
    else if (type == GGUF_INT32) memcpy(&v->i, &bits, 4);
    else if (type == GGUF_FLOAT32) memcpy(&v->f, &bits, 4);
    else return false;
    return true;
}

static const gguf_kv *metadata_array(const gguf_file *f, const char *key, uint32_t type,
                                     size_t count, bool required, char *error) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    if (!kv && !required) return NULL;
    if (!kv || kv->type != GGUF_ARRAY || kv->array_type != type ||
        (count && kv->array_len != count) || kv->array_len > INT32_MAX) {
        fail(error, "invalid or missing %s", key);
        return NULL;
    }
    return kv;
}

static bool metadata_id(const gguf_file *f, const char *key, size_t count,
                        int32_t *id, bool required, char *error) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    if (!kv && !required) { *id = -1; return true; }
    gguf_scalar v;
    if (!kv || (kv->type != GGUF_UINT32 && kv->type != GGUF_INT32) ||
        !gguf_scalar_read(f, kv->type, kv->value_pos, &v) ||
        (kv->type == GGUF_INT32 && v.i < 0)) return fail(error, "invalid or missing %s", key);
    uint64_t value = kv->type == GGUF_UINT32 ? v.u : (uint64_t)v.i;
    if (value >= count) return fail(error, "%s outside vocabulary", key);
    *id = (int32_t)value;
    return true;
}

static bool metadata_bool(const gguf_file *f, const char *key, bool fallback,
                          bool *value, char *error) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    if (!kv) { *value = fallback; return true; }
    gguf_scalar v;
    if (kv->type != GGUF_BOOL || !gguf_scalar_read(f, kv->type, kv->value_pos, &v))
        return fail(error, "invalid %s", key);
    *value = v.b;
    return true;
}

static bool metadata_is(const gguf_file *f, const char *key, const char *value) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    gguf_scalar v;
    return kv && kv->type == GGUF_STRING && gguf_scalar_read(f, kv->type, kv->value_pos, &v) &&
        v.s.len == strlen(value) && !memcmp(v.s.ptr, value, v.s.len);
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int byte_token(gguf_str s) {
    if (s.len != 6 || memcmp(s.ptr, "<0x", 3) || s.ptr[5] != '>') return -1;
    int hi = hex_digit(s.ptr[3]), lo = hex_digit(s.ptr[4]);
    return hi < 0 || lo < 0 ? -1 : hi * 16 + lo;
}

static bool load_merges(tokenizer *t, const gguf_file *f, char *error) {
    const gguf_kv *kv = metadata_array(f, "tokenizer.ggml.merges", GGUF_STRING, 0, true, error);
    if (!kv || !table_capacity((size_t)kv->array_len, &t->merge_size, error)) return false;
    t->merges = array_alloc(t->merge_size, sizeof(*t->merges), error);
    if (!t->merges) return false;
    uint64_t pos = kv->value_pos;
    for (uint32_t rank = 0; rank < kv->array_len; rank++) {
        gguf_scalar v;
        if (!gguf_scalar_read(f, GGUF_STRING, pos, &v) || !gguf_value_skip(f, GGUF_STRING, &pos))
            return fail(error, "cannot read tokenizer merge");
        const char *sep = memchr(v.s.ptr, ' ', (size_t)v.s.len);
        if (!sep) return fail(error, "invalid tokenizer merge at rank %u", rank);
        size_t left_len = (size_t)(sep - v.s.ptr), right_len = (size_t)v.s.len - left_len - 1;
        int32_t left = token_lookup(t, v.s.ptr, left_len, text_hash(v.s.ptr, left_len));
        int32_t right = token_lookup(t, sep + 1, right_len, text_hash(sep + 1, right_len));
        if (left < 0 || right < 0) return fail(error, "merge operands outside vocabulary at rank %u", rank);
        /* The GGUF merge has a separator; the result is looked up without copying it. */
        uint64_t power = 1;
        for (size_t j = 0; j < right_len; j++) power *= HASH_BASE;
        uint64_t hash = t->tokens[left].hash * power + t->tokens[right].hash;
        size_t slot = hash_slot(hash, t->table_size);
        int32_t result = -1;
        while (t->table[slot] >= 0) {
            int32_t id = t->table[slot];
            const token *tok = &t->tokens[id];
            if (tok->hash == hash && tok->text.len == left_len + right_len &&
                !memcmp(tok->text.ptr, v.s.ptr, left_len) &&
                !memcmp(tok->text.ptr + left_len, sep + 1, right_len)) { result = id; break; }
            slot = (slot + 1) & (t->table_size - 1);
        }
        if (result < 0) return fail(error, "merge result outside vocabulary at rank %u", rank);
        slot = merge_slot(t, left, right);
        while (t->merges[slot].used &&
               (t->merges[slot].left != left || t->merges[slot].right != right))
            slot = (slot + 1) & (t->merge_size - 1);
        /* Rust HashMap::insert gives the last duplicate pair its rank. */
        t->merges[slot] = (merge){left, right, result, rank, true};
    }
    return true;
}

void tokenizer_destroy(tokenizer *t) {
    if (!t) return;
    free(t->tokens); free(t->table); free(t->merges); free(t->specials); free(t);
}

tokenizer *tokenizer_create(const gguf_file *f, char error[GGUF_ERROR_SIZE]) {
    error[0] = '\0';
    tokenizer *t = array_alloc(1, sizeof(*t), error);
    if (!t) return NULL;
    t->gemma = metadata_is(f, "tokenizer.ggml.model", "llama") &&
               metadata_is(f, "general.architecture", "gemma-embedding");
    bool lfm = metadata_is(f, "tokenizer.ggml.model", "gpt2") &&
               metadata_is(f, "tokenizer.ggml.pre", "lfm2") &&
               (metadata_is(f, "general.architecture", "lfm2") ||
                metadata_is(f, "general.architecture", "lfm2moe"));
    if (!t->gemma && !lfm) { fail(error, "unsupported GGUF tokenizer model/architecture"); goto bad; }
    const gguf_kv *vocab = metadata_array(f, "tokenizer.ggml.tokens", GGUF_STRING, 0, true, error);
    if (!vocab) goto bad;
    t->n_tokens = (size_t)vocab->array_len;
    if (!t->n_tokens) { fail(error, "empty tokenizer vocabulary"); goto bad; }
    const gguf_kv *types = metadata_array(f, "tokenizer.ggml.token_type", GGUF_INT32, t->n_tokens, true, error);
    const gguf_kv *scores = t->gemma ? metadata_array(f, "tokenizer.ggml.scores", GGUF_FLOAT32, t->n_tokens, true, error) : NULL;
    if (!types || (t->gemma && !scores)) goto bad;
    if (!table_capacity(t->n_tokens, &t->table_size, error)) goto bad;
    t->tokens = array_alloc(t->n_tokens, sizeof(*t->tokens), error);
    t->table = array_alloc(t->table_size, sizeof(*t->table), error);
    t->specials = array_alloc(t->n_tokens, sizeof(*t->specials), error);
    if (!t->tokens || !t->table || !t->specials) goto bad;
    for (size_t i = 0; i < t->table_size; i++) t->table[i] = -1;
    for (size_t i = 0; i < 256; i++) t->bytes[i] = -1;
    uint64_t pos = vocab->value_pos, type_pos = types->value_pos;
    uint64_t score_pos = scores ? scores->value_pos : 0;
    for (size_t i = 0; i < t->n_tokens; i++) {
        token *tok = &t->tokens[i];
        gguf_scalar v;
        if (!gguf_scalar_read(f, GGUF_STRING, pos, &v) || !gguf_value_skip(f, GGUF_STRING, &pos)) {
            fail(error, "cannot read tokenizer vocabulary"); goto bad;
        }
        tok->text = v.s;
        if (!v.s.len || !utf8_valid(v.s.ptr, (size_t)v.s.len)) {
            fail(error, "empty or invalid UTF-8 token at id %zu", i); goto bad;
        }
        if (!gguf_scalar_read(f, GGUF_INT32, type_pos, &v) || !gguf_value_skip(f, GGUF_INT32, &type_pos)) goto bad;
        tok->type = (uint32_t)v.i;
        if (scores) {
            if (!gguf_scalar_read(f, GGUF_FLOAT32, score_pos, &v) || !gguf_value_skip(f, GGUF_FLOAT32, &score_pos)) goto bad;
            tok->score = (float)v.f;
            if (!isfinite(tok->score)) { fail(error, "non-finite tokenizer score at id %zu", i); goto bad; }
        }
        tok->hash = text_hash(tok->text.ptr, (size_t)tok->text.len);
        size_t slot = hash_slot(tok->hash, t->table_size);
        while (t->table[slot] >= 0) {
            const token *old = &t->tokens[t->table[slot]];
            if (old->hash == tok->hash && old->text.len == tok->text.len &&
                !memcmp(old->text.ptr, tok->text.ptr, (size_t)tok->text.len)) break;
            slot = (slot + 1) & (t->table_size - 1);
        }
        t->table[slot] = (int32_t)i;
        if (!t->gemma && (tok->type == 3 || tok->type == 4)) t->specials[t->n_specials++] = (int32_t)i;
        if (t->gemma && tok->type == 6) {
            int b = byte_token(tok->text);
            if (b >= 0) t->bytes[b] = (int32_t)i;
        }
    }
    for (size_t i = 0; i < t->n_specials; i++)
        t->special_offsets[(uint8_t)t->tokens[t->specials[i]].text.ptr[0] + 1]++;
    for (size_t i = 1; i < 257; i++) t->special_offsets[i] += t->special_offsets[i - 1];
    size_t next_special[256];
    memcpy(next_special, t->special_offsets, sizeof(next_special));
    for (size_t i = 0; i < t->n_tokens; i++) {
        const token *tok = &t->tokens[i];
        if (!t->gemma && (tok->type == 3 || tok->type == 4))
            t->specials[next_special[(uint8_t)tok->text.ptr[0]]++] = (int32_t)i;
    }
    if (!metadata_id(f, "tokenizer.ggml.bos_token_id", t->n_tokens, &t->bos, true, error) ||
        !metadata_id(f, "tokenizer.ggml.eos_token_id", t->n_tokens, &t->eos, true, error) ||
        !metadata_id(f, "tokenizer.ggml.unknown_token_id", t->n_tokens, &t->unknown, t->gemma, error)) goto bad;
    t->add_bos = true;
    if (t->gemma) {
        if (!metadata_bool(f, "tokenizer.ggml.add_bos_token", true, &t->add_bos, error) ||
            !metadata_bool(f, "tokenizer.ggml.add_eos_token", true, &t->add_eos, error)) goto bad;
    } else {
        for (size_t i = 0; i < 324; i++) t->byte_decoder[i] = -1;
        unsigned extra = 0;
        for (unsigned b = 0; b < 256; b++) {
            uint32_t cp = b;
            if (!(b >= 33 && b <= 126) && !(b >= 161 && b <= 172) && !(b >= 174)) cp = 256 + extra++;
            t->byte_chars[b] = cp;
            t->byte_decoder[cp] = (int16_t)b;
            char encoded[4];
            size_t n = utf8_put(cp, encoded);
            t->bytes[b] = token_lookup(t, encoded, n, text_hash(encoded, n));
            if (t->bytes[b] < 0) { fail(error, "missing byte %u in tokenizer vocabulary", b); goto bad; }
        }
        if (!load_merges(t, f, error)) goto bad;
    }
    return t;
bad:
    if (!error[0]) fail(error, "invalid tokenizer metadata");
    tokenizer_destroy(t);
    return NULL;
}

static bool candidate_before(candidate a, candidate b) {
    if (a.priority != b.priority) return a.priority < b.priority;
    if (!!signbit(a.priority) != !!signbit(b.priority)) return !!signbit(a.priority);
    return a.left < b.left;
}

static void heap_push(bpe *work, candidate value) {
    size_t i = work->heap_count++;
    while (i) {
        size_t parent = (i - 1) / 2;
        if (!candidate_before(value, work->heap[parent])) break;
        work->heap[i] = work->heap[parent];
        i = parent;
    }
    work->heap[i] = value;
}

static candidate heap_pop(bpe *work) {
    candidate result = work->heap[0], last = work->heap[--work->heap_count];
    size_t i = 0;
    while (i * 2 + 1 < work->heap_count) {
        size_t child = i * 2 + 1;
        if (child + 1 < work->heap_count && candidate_before(work->heap[child + 1], work->heap[child])) child++;
        if (!candidate_before(work->heap[child], last)) break;
        work->heap[i] = work->heap[child];
        i = child;
    }
    if (work->heap_count) work->heap[i] = last;
    return result;
}

static void offer_pair(const tokenizer *t, bpe *work, size_t left) {
    if (left == NO_NODE) return;
    symbol *a = &work->nodes[left];
    if (a->next == NO_NODE) return;
    symbol *b = &work->nodes[a->next];
    int32_t result;
    double priority;
    if (t->gemma) {
        uint64_t hash = a->hash * b->power + b->hash;
        result = token_lookup(t, work->text + a->start, a->len + b->len, hash);
        if (result < 0) return;
        priority = -(double)t->tokens[result].score;
    } else {
        const merge *m = merge_lookup(t, a->id, b->id);
        if (!m) return;
        result = m->result;
        priority = m->rank;
    }
    heap_push(work, (candidate){left, a->next, a->version, b->version, result, priority});
}

static bool bpe_reserve(bpe *work, size_t count, char *error) {
    if (count <= work->capacity) return true;
    if (count > SIZE_MAX / 3) return fail(error, "tokenizer input too long");
    symbol *nodes = array_alloc(count, sizeof(*nodes), error);
    candidate *heap = array_alloc(count * 3, sizeof(*heap), error);
    if (!nodes || !heap) { free(nodes); free(heap); return false; }
    free(work->nodes); free(work->heap);
    work->nodes = nodes; work->heap = heap; work->capacity = count;
    return true;
}

/* Each merge creates at most two edges; stale heap entries are checked lazily. */
static void merge_symbols(const tokenizer *t, bpe *work, size_t count) {
    work->heap_count = 0;
    for (size_t i = 0; i < count; i++) offer_pair(t, work, i);
    while (work->heap_count) {
        candidate c = heap_pop(work);
        symbol *a = &work->nodes[c.left], *b = &work->nodes[c.right];
        if (!a->live || !b->live || a->next != c.right || a->version != c.version || b->version != c.right_version) continue;
        a->len += b->len;
        a->hash = a->hash * b->power + b->hash;
        a->power *= b->power;
        a->id = c.result;
        a->next = b->next;
        a->version++;
        b->live = false;
        if (a->next != NO_NODE) work->nodes[a->next].prev = c.left;
        offer_pair(t, work, a->prev);
        offer_pair(t, work, c.left);
    }
}

typedef struct {
    uint32_t cp;
    size_t offset;
    bool letter, number, space;
} character;

static bool punct(character c) { return !c.letter && !c.number && !c.space; }
static uint32_t lower_ascii(uint32_t cp) { return cp >= 'A' && cp <= 'Z' ? cp + 32 : cp; }

static size_t pretoken_end(const character *c, size_t n, size_t i) {
    if (c[i].cp == '\'' && i + 1 < n) {
        uint32_t a = lower_ascii(c[i + 1].cp);
        if (a == 's' || a == 'd' || a == 'm' || a == 't') return i + 2;
        if (i + 2 < n) {
            uint32_t b = lower_ascii(c[i + 2].cp);
            if ((a == 'l' && b == 'l') || (a == 'v' && b == 'e') || (a == 'r' && b == 'e')) return i + 3;
        }
    }
    size_t j = i;
    if (c[i].cp != '\r' && c[i].cp != '\n' && !c[i].letter && !c[i].number &&
        i + 1 < n && c[i + 1].letter) j++;
    size_t start = j;
    while (j < n && c[j].letter) j++;
    if (j > start) return j;
    j = i;
    while (j < n && j - i < 3 && c[j].number) j++;
    if (j > i) return j;
    j = i;
    if (c[i].cp == ' ' && i + 1 < n && punct(c[i + 1])) j++;
    start = j;
    while (j < n && punct(c[j])) j++;
    if (j > start) {
        while (j < n && (c[j].cp == '\r' || c[j].cp == '\n')) j++;
        return j;
    }
    j = i;
    size_t last_newline = i;
    while (j < n && c[j].space) {
        if (c[j].cp == '\r' || c[j].cp == '\n') last_newline = j + 1;
        j++;
    }
    if (last_newline > i) return last_newline;
    if (j > i) return j < n && j - i > 1 ? j - 1 : j;
    return i + 1;
}

static bool encode_lfm_span(const tokenizer *t, const char *text, size_t len,
                            bpe *work, int32_t *ids, size_t *count, char *error) {
    if (!len) return true;
    character *chars = array_alloc(len + 1, sizeof(*chars), error);
    if (len > SIZE_MAX / 2) { free(chars); return fail(error, "tokenizer input too long"); }
    char *encoded = array_alloc(len * 2, 1, error);
    if (!chars || !encoded) { free(chars); free(encoded); return false; }
    size_t at = 0, n_chars = 0;
    while (at < len) {
        size_t offset = at;
        uint32_t cp;
        if (!utf8_next(text, len, &at, &cp)) { free(chars); free(encoded); return fail(error, "invalid UTF-8 input"); }
        chars[n_chars++] = (character){cp, offset, PROPERTY(cp, alphabetic), PROPERTY(cp, numeric), PROPERTY(cp, whitespace)};
    }
    chars[n_chars].offset = len;
    bool ok = true;
    for (size_t i = 0; i < n_chars;) {
        size_t end = pretoken_end(chars, n_chars, i);
        size_t start = chars[i].offset, finish = chars[end].offset, bytes = finish - start;
        if (!bpe_reserve(work, bytes, error)) { ok = false; break; }
        size_t used = 0;
        for (size_t k = 0; k < bytes; k++) {
            unsigned b = (uint8_t)text[start + k];
            size_t width = utf8_put(t->byte_chars[b], encoded + used);
            uint64_t power = width == 1 ? HASH_BASE : HASH_BASE * HASH_BASE;
            work->nodes[k] = (symbol){used, width, k ? k - 1 : NO_NODE,
                k + 1 < bytes ? k + 1 : NO_NODE, 0, text_hash(encoded + used, width), power, t->bytes[b], true};
            used += width;
        }
        work->text = encoded;
        merge_symbols(t, work, bytes);
        for (size_t k = 0; k != NO_NODE; k = work->nodes[k].next) ids[(*count)++] = work->nodes[k].id;
        i = end;
    }
    free(chars); free(encoded);
    return ok;
}

static bool encode_gemma(const tokenizer *t, const char *text, size_t len,
                         bpe *work, int32_t *ids, size_t *count, char *error) {
    if (!len) return true;
    if (len > SIZE_MAX / 3) return fail(error, "tokenizer input too long");
    char *normalized = array_alloc(len * 3, 1, error);
    if (!normalized || !bpe_reserve(work, len, error)) { free(normalized); return false; }
    size_t at = 0, used = 0, n = 0;
    while (at < len) {
        uint32_t cp;
        if (!utf8_next(text, len, &at, &cp)) { free(normalized); return fail(error, "invalid UTF-8 input"); }
        size_t width = utf8_put(cp == ' ' ? 0x2581 : cp, normalized + used);
        uint64_t hash = text_hash(normalized + used, width), power = 1;
        for (size_t k = 0; k < width; k++) power *= HASH_BASE;
        int32_t id = token_lookup(t, normalized + used, width, hash);
        work->nodes[n] = (symbol){used, width, n ? n - 1 : NO_NODE, n + 1,
            0, hash, power, id, true};
        used += width;
        n++;
    }
    work->nodes[n - 1].next = NO_NODE;
    work->text = normalized;
    merge_symbols(t, work, n);
    for (size_t k = 0; k != NO_NODE; k = work->nodes[k].next) {
        symbol *s = &work->nodes[k];
        if (s->id >= 0) ids[(*count)++] = s->id;
        else for (size_t b = 0; b < s->len; b++) {
            int32_t id = t->bytes[(uint8_t)normalized[s->start + b]];
            ids[(*count)++] = id >= 0 ? id : t->unknown;
        }
    }
    free(normalized);
    return true;
}

bool tokenizer_encode(const tokenizer *t, const char *text, size_t len, bool add_special,
                      int32_t **ids, size_t *count, char error[GGUF_ERROR_SIZE]) {
    *ids = NULL; *count = 0; error[0] = '\0';
    if (!utf8_valid(text, len)) return fail(error, "invalid UTF-8 input");
    if (len > (SIZE_MAX - 2) / 3) return fail(error, "tokenizer input too long");
    int32_t *result = array_alloc(len * 3 + 2, sizeof(*result), error);
    if (!result) return false;
    bpe work = {0};
    bool ok = true;
    if (add_special && t->add_bos) result[(*count)++] = t->bos;
    if (t->gemma) ok = encode_gemma(t, text, len, &work, result, count, error);
    else {
        size_t plain = 0, at = 0;
        while (at < len) {
            int32_t found = -1;
            size_t longest = 0;
            unsigned first = (uint8_t)text[at];
            for (size_t k = t->special_offsets[first]; k < t->special_offsets[first + 1]; k++) {
                int32_t id = t->specials[k];
                const token *tok = &t->tokens[id];
                if (tok->text.len > longest && tok->text.len <= len - at &&
                    tok->text.ptr[0] == text[at] && !memcmp(tok->text.ptr, text + at, (size_t)tok->text.len)) {
                    found = id; longest = (size_t)tok->text.len;
                }
            }
            if (found < 0) { at++; continue; }
            if (!encode_lfm_span(t, text + plain, at - plain, &work, result, count, error)) { ok = false; break; }
            result[(*count)++] = found;
            at += longest;
            plain = at;
        }
        if (ok) ok = encode_lfm_span(t, text + plain, len - plain, &work, result, count, error);
    }
    if (ok && add_special && t->add_eos) result[(*count)++] = t->eos;
    free(work.nodes); free(work.heap);
    if (!ok) { free(result); *count = 0; return false; }
    *ids = result;
    return true;
}

/* Rust String::from_utf8_lossy replaces one invalid subsequence at a time. */
static size_t invalid_utf8_len(const char *s, size_t n) {
    uint8_t first = (uint8_t)s[0];
    unsigned width = first >= 0xc2 && first <= 0xdf ? 2 :
                     first >= 0xe0 && first <= 0xef ? 3 :
                     first >= 0xf0 && first <= 0xf4 ? 4 : 1;
    if (width == 1 || n == 1) return 1;
    uint8_t second = (uint8_t)s[1];
    if ((second & 0xc0) != 0x80 || (first == 0xe0 && second < 0xa0) ||
        (first == 0xed && second >= 0xa0) || (first == 0xf0 && second < 0x90) ||
        (first == 0xf4 && second >= 0x90)) return 1;
    for (unsigned i = 2; i < width; i++) {
        if (i >= n || ((uint8_t)s[i] & 0xc0) != 0x80) return i;
    }
    return width;
}

bool tokenizer_decode(const tokenizer *t, const int32_t *ids, size_t count,
                      char **text, size_t *len, char error[GGUF_ERROR_SIZE]) {
    *text = NULL; *len = 0; error[0] = '\0';
    size_t capacity = 0;
    for (size_t i = 0; i < count; i++) {
        if (ids[i] < 0 || (size_t)ids[i] >= t->n_tokens) return fail(error, "token id outside vocabulary");
        size_t width = (size_t)t->tokens[ids[i]].text.len;
        if (width > SIZE_MAX - capacity) return fail(error, "decoded text too long");
        capacity += width;
    }
    if (capacity > (SIZE_MAX - 1) / 3) return fail(error, "decoded text too long");
    char *raw = array_alloc(capacity + 1, 1, error);
    char *result = array_alloc(capacity * 3 + 1, 1, error);
    if (!raw || !result) { free(raw); free(result); return false; }
    size_t used = 0;
    for (size_t i = 0; i < count; i++) {
        const token *tok = &t->tokens[ids[i]];
        gguf_str s = tok->text;
        if (t->gemma && tok->type == 3) continue;
        int byte = t->gemma && tok->type == 6 ? byte_token(s) : -1;
        if (byte >= 0) { raw[used++] = (char)byte; continue; }
        size_t at = 0;
        while (at < s.len) {
            uint32_t cp;
            size_t start = at;
            if (!utf8_next(s.ptr, (size_t)s.len, &at, &cp)) {
                free(raw); free(result); return fail(error, "invalid UTF-8 vocabulary token");
            }
            if (t->gemma) {
                if (cp == 0x2581) raw[used++] = ' ';
                else { memcpy(raw + used, s.ptr + start, at - start); used += at - start; }
            } else {
                if (cp < 324 && t->byte_decoder[cp] >= 0) raw[used++] = (char)t->byte_decoder[cp];
            }
        }
    }
    for (size_t at = 0; at < used;) {
        size_t start = at;
        uint32_t cp;
        if (utf8_next(raw, used, &at, &cp)) {
            memcpy(result + *len, raw + start, at - start);
            *len += at - start;
        } else {
            memcpy(result + *len, "\xef\xbf\xbd", 3);
            *len += 3;
            at = start + invalid_utf8_len(raw + start, used - start);
        }
    }
    result[*len] = '\0';
    free(raw);
    *text = result;
    return true;
}

typedef struct tokenizer_cache_entry {
    char *path;
    off_t size;
    struct timespec mtime;
    gguf_file file;
    tokenizer *value;
    struct tokenizer_cache_entry *next;
} tokenizer_cache_entry;

static pthread_mutex_t cache_lock = PTHREAD_MUTEX_INITIALIZER;
static tokenizer_cache_entry *cache;
static bool cache_cleanup_registered;

static void cache_cleanup(void) {
    pthread_mutex_lock(&cache_lock);
    while (cache) {
        tokenizer_cache_entry *entry = cache;
        cache = entry->next;
        tokenizer_destroy(entry->value);
        gguf_close(&entry->file);
        free(entry->path); free(entry);
    }
    pthread_mutex_unlock(&cache_lock);
}

const tokenizer *tokenizer_cached(const char *path, char error[GGUF_ERROR_SIZE]) {
    struct stat before, after;
    error[0] = '\0';
    if (stat(path, &before)) {
        fail(error, "cannot stat tokenizer '%s': %s", path, strerror(errno)); return NULL;
    }
    if (!S_ISREG(before.st_mode)) { fail(error, "tokenizer path is not a regular file"); return NULL; }
    pthread_mutex_lock(&cache_lock);
    for (tokenizer_cache_entry *entry = cache; entry; entry = entry->next) {
        if (entry->size == before.st_size && entry->mtime.tv_sec == before.st_mtim.tv_sec &&
            entry->mtime.tv_nsec == before.st_mtim.tv_nsec && !strcmp(entry->path, path)) {
            pthread_mutex_unlock(&cache_lock);
            return entry->value;
        }
    }
    tokenizer_cache_entry *entry = array_alloc(1, sizeof(*entry), error);
    if (!entry) goto bad;
    entry->path = strdup(path);
    if (!entry->path) { fail(error, "out of memory in tokenizer cache"); goto bad; }
    if (!gguf_open(&entry->file, path)) { fail(error, "%s", entry->file.error); goto bad; }
    if (stat(path, &after) || before.st_size != after.st_size || before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
        before.st_mtim.tv_nsec != after.st_mtim.tv_nsec || before.st_dev != after.st_dev || before.st_ino != after.st_ino) {
        fail(error, "tokenizer file changed while loading"); goto bad;
    }
    entry->value = tokenizer_create(&entry->file, error);
    if (!entry->value) goto bad;
    if (!cache_cleanup_registered) {
        if (atexit(cache_cleanup)) { fail(error, "cannot register tokenizer cache cleanup"); goto bad; }
        cache_cleanup_registered = true;
    }
    entry->size = before.st_size;
    entry->mtime = before.st_mtim;
    entry->next = cache;
    cache = entry;
    pthread_mutex_unlock(&cache_lock);
    return entry->value;
bad:
    if (entry) {
        tokenizer_destroy(entry->value); gguf_close(&entry->file);
        free(entry->path); free(entry);
    }
    pthread_mutex_unlock(&cache_lock);
    return NULL;
}
