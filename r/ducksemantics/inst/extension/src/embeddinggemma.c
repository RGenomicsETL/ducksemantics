/* EmbeddingGemma-300M CPU encoder. SPDX-License-Identifier: MIT
 * Forward/prompt contract: Rbebelm embeddinggemma (MIT).
 */
#define _POSIX_C_SOURCE 200809L
#include "embeddinggemma.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define LAYERS 24
#define HEADS 3
#define HEAD 256
#define FF 1152
#define DENSE 3072
#define VOCAB 262144
#define EPS 1e-6f
#define TEXT_LIMIT (16U * 1024U * 1024U)

typedef struct gemma_layer {
    quant_view q, k, v, output, gate, up, down;
    float pre_attn[GEMMA_DIM], post_attn[GEMMA_DIM];
    float pre_ff[GEMMA_DIM], post_ff[GEMMA_DIM];
    float q_norm[HEAD], k_norm[HEAD];
} gemma_layer;

struct gemma_model {
    gguf_file file;
    tokenizer *tokens;
    quant_view embedding, dense2, dense3;
    float output_norm[GEMMA_DIM];
    gemma_layer layers[LAYERS];
};

struct gemma_workspace {
    size_t capacity;
    float *storage;
    float *x, *norm, *q, *k, *v, *attention, *projected, *gate, *up;
    float scratch[DENSE], pooled[GEMMA_DIM], dense[DENSE];
    float scores[GEMMA_CONTEXT];
    float sin_local[GEMMA_CONTEXT][HEAD / 2], cos_local[GEMMA_CONTEXT][HEAD / 2];
    float sin_global[GEMMA_CONTEXT][HEAD / 2], cos_global[GEMMA_CONTEXT][HEAD / 2];
};

static bool fail(char *error, const char *message) {
    snprintf(error, GGUF_ERROR_SIZE, "%s", message);
    return false;
}

static bool metadata(const gguf_file *f, const char *key, const char *value) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    gguf_str actual;
    char buf[64];
    return kv && gguf_scalar_text(f, kv->type, kv->value_pos, buf, sizeof(buf), &actual) &&
           actual.len == strlen(value) && !memcmp(actual.ptr, value, (size_t)actual.len);
}

static bool metadata_float(const gguf_file *f, const char *key, float expected, float tolerance) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    gguf_str actual;
    char buf[64];
    if (!kv || kv->type != GGUF_FLOAT32 ||
        !gguf_scalar_text(f, kv->type, kv->value_pos, buf, sizeof(buf), &actual)) return false;
    float value = strtof(buf, NULL);
    return isfinite(value) && fabsf(value - expected) <= tolerance;
}

static bool tensor(const gguf_file *f, const char *name, uint64_t columns,
                   uint64_t rows, bool vector, quant_view *out, char *error) {
    const gguf_tensor *t = NULL;
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        if (f->tensors[i].name.len == strlen(name) &&
            !memcmp(f->tensors[i].name.ptr, name, strlen(name))) {
            t = &f->tensors[i];
            break;
        }
    }
    if (!t || t->n_dims != (vector ? 1U : 2U) || t->dims[0] != columns ||
        (!vector && t->dims[1] != rows) ||
        (t->type != QUANT_F32 && t->type != QUANT_Q8_0)) {
        snprintf(error, GGUF_ERROR_SIZE, "EmbeddingGemma tensor %s: missing or unsupported shape/type", name);
        return false;
    }
    return quant_view_init(out, f, t, error);
}

static bool gain(const gguf_file *f, const char *name, size_t n, float *out, char *error) {
    quant_view v;
    if (!tensor(f, name, n, 1, true, &v, error) ||
        !quant_dequantize_row(v.type, v.data, v.row_stride, out, n, error)) return false;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(out[i])) return fail(error, "EmbeddingGemma norm contains nonfinite values");
    }
    return true;
}

static bool model_init(gemma_model *m, const char *path, char *error) {
    static const struct { const char *key, *value; } profile[] = {
        {"general.architecture", "gemma-embedding"},
        {"gemma-embedding.context_length", "2048"},
        {"gemma-embedding.embedding_length", "768"},
        {"gemma-embedding.block_count", "24"},
        {"gemma-embedding.feed_forward_length", "1152"},
        {"gemma-embedding.attention.head_count", "3"},
        {"gemma-embedding.attention.head_count_kv", "1"},
        {"gemma-embedding.attention.key_length", "256"},
        {"gemma-embedding.attention.value_length", "256"},
        {"gemma-embedding.attention.sliding_window", "512"},
        {"gemma-embedding.dense_2_feat_in", "768"},
        {"gemma-embedding.dense_2_feat_out", "3072"},
        {"gemma-embedding.dense_3_feat_in", "3072"},
        {"gemma-embedding.dense_3_feat_out", "768"},
        {"gemma-embedding.pooling_type", "1"},
        {"tokenizer.ggml.bos_token_id", "2"},
        {"tokenizer.ggml.eos_token_id", "1"},
        {"tokenizer.ggml.padding_token_id", "0"},
        {"tokenizer.ggml.add_bos_token", "true"},
        {"tokenizer.ggml.add_eos_token", "true"}
    };
    if (!gguf_open(&m->file, path)) return fail(error, m->file.error);
    const gguf_file *f = &m->file;
    for (size_t i = 0; i < sizeof(profile) / sizeof(profile[0]); i++) {
        if (!metadata(f, profile[i].key, profile[i].value)) {
            snprintf(error, GGUF_ERROR_SIZE, "EmbeddingGemma profile requires %s=%s", profile[i].key, profile[i].value);
            return false;
        }
    }
    if (!metadata_float(f, "gemma-embedding.rope.freq_base", 1000000.0f, 1.0f) ||
        !metadata_float(f, "gemma-embedding.rope.freq_base_swa", 10000.0f, 1e-3f) ||
        !metadata_float(f, "gemma-embedding.attention.layer_norm_rms_epsilon", EPS, 1e-9f))
        return fail(error, "unsupported EmbeddingGemma RoPE/RMS epsilon profile");
    const gguf_kv *vocab = gguf_find_kv(f, "tokenizer.ggml.tokens");
    if (!vocab || vocab->type != GGUF_ARRAY || vocab->array_len != VOCAB)
        return fail(error, "EmbeddingGemma vocabulary must have 262144 entries");
    if (f->n_tensors != 316) return fail(error, "EmbeddingGemma profile requires exactly 316 tensors");
    if (!tensor(f, "token_embd.weight", GEMMA_DIM, VOCAB, false, &m->embedding, error) ||
        !tensor(f, "dense_2.weight", GEMMA_DIM, DENSE, false, &m->dense2, error) ||
        !tensor(f, "dense_3.weight", DENSE, GEMMA_DIM, false, &m->dense3, error) ||
        !gain(f, "output_norm.weight", GEMMA_DIM, m->output_norm, error)) return false;
    for (unsigned l = 0; l < LAYERS; l++) {
        gemma_layer *b = &m->layers[l];
        struct { const char *suffix; size_t columns, rows; quant_view *view; } matrices[] = {
            {"attn_q", GEMMA_DIM, GEMMA_DIM, &b->q},
            {"attn_k", GEMMA_DIM, HEAD, &b->k},
            {"attn_v", GEMMA_DIM, HEAD, &b->v},
            {"attn_output", GEMMA_DIM, GEMMA_DIM, &b->output},
            {"ffn_gate", GEMMA_DIM, FF, &b->gate},
            {"ffn_up", GEMMA_DIM, FF, &b->up},
            {"ffn_down", FF, GEMMA_DIM, &b->down}
        };
        struct { const char *suffix; size_t n; float *out; } gains[] = {
            {"attn_norm", GEMMA_DIM, b->pre_attn},
            {"post_attention_norm", GEMMA_DIM, b->post_attn},
            {"ffn_norm", GEMMA_DIM, b->pre_ff},
            {"post_ffw_norm", GEMMA_DIM, b->post_ff},
            {"attn_q_norm", HEAD, b->q_norm},
            {"attn_k_norm", HEAD, b->k_norm}
        };
        char name[80];
        for (size_t i = 0; i < sizeof(matrices) / sizeof(matrices[0]); i++) {
            snprintf(name, sizeof(name), "blk.%u.%s.weight", l, matrices[i].suffix);
            if (!tensor(f, name, matrices[i].columns, matrices[i].rows, false,
                        matrices[i].view, error)) return false;
        }
        for (size_t i = 0; i < sizeof(gains) / sizeof(gains[0]); i++) {
            snprintf(name, sizeof(name), "blk.%u.%s.weight", l, gains[i].suffix);
            if (!gain(f, name, gains[i].n, gains[i].out, error)) return false;
        }
    }
    m->tokens = tokenizer_create(f, error);
    return m->tokens != NULL;
}

static const struct { const char *name, *prefix; } tasks[GEMMA_TASK_COUNT] = {
    {"retrieval_query", "task: search result | query: "},
    {"retrieval_document", "title: none | text: "},
    {"question_answering", "task: question answering | query: "},
    {"fact_verification", "task: fact checking | query: "},
    {"classification", "task: classification | query: "},
    {"clustering", "task: clustering | query: "},
    {"semantic_similarity", "task: sentence similarity | query: "},
    {"code_retrieval", "task: code retrieval | query: "},
    {"summarization", "task: summarization | query: "},
    {"raw", ""}
};

int gemma_task(const char *text, size_t len) {
    for (int i = 0; i < GEMMA_TASK_COUNT; i++)
        if (len == strlen(tasks[i].name) && !memcmp(text, tasks[i].name, len)) return i;
    return -1;
}

bool gemma_dimensions(int dimensions) {
    return dimensions == 768 || dimensions == 512 || dimensions == 256 || dimensions == 128;
}

bool gemma_tokenize(const gemma_model *m, const char *text, size_t len,
                    int task, int32_t **ids, size_t *count, char *error) {
    *ids = NULL;
    *count = 0;
    if (task < 0 || task >= GEMMA_TASK_COUNT) return fail(error, "unknown EmbeddingGemma task");
    size_t prefix = strlen(tasks[task].prefix);
    if (len > TEXT_LIMIT - prefix) return fail(error, "EmbeddingGemma input exceeds 16 MiB");
    char *formatted = malloc(prefix + len + 1);
    if (!formatted) return fail(error, "out of memory formatting EmbeddingGemma input");
    memcpy(formatted, tasks[task].prefix, prefix);
    memcpy(formatted + prefix, text, len);
    formatted[prefix + len] = '\0';
    bool ok = tokenizer_encode(m->tokens, formatted, prefix + len, true, ids, count, error);
    free(formatted);
    if (!ok) return false;
    if (*count > GEMMA_CONTEXT) {
        *count = GEMMA_CONTEXT;
        (*ids)[GEMMA_CONTEXT - 1] = 1;
    }
    return true;
}

gemma_workspace *gemma_workspace_new(size_t capacity, char *error) {
    if (!capacity || capacity > GEMMA_CONTEXT) {
        fail(error, "EmbeddingGemma workspace capacity must be 1..2048");
        return NULL;
    }
    gemma_workspace *w = calloc(1, sizeof(*w));
    if (!w) { fail(error, "out of memory allocating EmbeddingGemma workspace"); return NULL; }
    w->capacity = capacity;
    w->storage = malloc(capacity * (5 * GEMMA_DIM + 2 * HEAD + 2 * FF) * sizeof(float));
    if (!w->storage) {
        free(w);
        fail(error, "out of memory allocating EmbeddingGemma token scratch");
        return NULL;
    }
    float *p = w->storage;
    w->x = p; p += capacity * GEMMA_DIM;
    w->norm = p; p += capacity * GEMMA_DIM;
    w->q = p; p += capacity * GEMMA_DIM;
    w->k = p; p += capacity * HEAD;
    w->v = p; p += capacity * HEAD;
    w->attention = p; p += capacity * GEMMA_DIM;
    w->projected = p; p += capacity * GEMMA_DIM;
    w->gate = p; p += capacity * FF;
    w->up = p;
    float scales[2] = {powf(10000.0f, -2.0f / HEAD), powf(1000000.0f, -2.0f / HEAD)};
    for (size_t t = 0; t < capacity; t++) {
        float local = (float)t, global = (float)t;
        for (size_t i = 0; i < HEAD / 2; i++) {
            w->sin_local[t][i] = sinf(local); w->cos_local[t][i] = cosf(local);
            w->sin_global[t][i] = sinf(global); w->cos_global[t][i] = cosf(global);
            local *= scales[0]; global *= scales[1];
        }
    }
    return w;
}

void gemma_workspace_free(gemma_workspace *w) {
    if (!w) return;
    free(w->storage);
    free(w);
}

static void rmsnorm(const float *x, const float *gain_values, size_t n, float *out) {
    double ss = 0;
    for (size_t i = 0; i < n; i++) ss += (double)(x[i] * x[i]);
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + EPS);
    for (size_t i = 0; i < n; i++) out[i] = (x[i] * scale) * gain_values[i];
}

static void norm_rows(const float *x, const float *g, size_t rows, float *out) {
    for (size_t t = 0; t < rows; t++) rmsnorm(x + t * GEMMA_DIM, g, GEMMA_DIM, out + t * GEMMA_DIM);
}

static void rope(float *x, const float *sin_values, const float *cos_values) {
    for (size_t i = 0; i < HEAD / 2; i++) {
        float a = x[i], b = x[i + HEAD / 2];
        x[i] = a * cos_values[i] - b * sin_values[i];
        x[i + HEAD / 2] = a * sin_values[i] + b * cos_values[i];
    }
}

static void softmax(float *x, size_t n) {
    float max = x[0];
    for (size_t i = 1; i < n; i++) if (x[i] > max) max = x[i];
    double sum = 0;
    for (size_t i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += (double)x[i]; }
    float inv = (float)(1.0 / sum);
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

static void attention(gemma_workspace *w, const gemma_layer *b,
                       const gemma_sequence *seq, size_t ns, bool local) {
    for (size_t s = 0; s < ns; s++) {
        size_t first = seq[s].first, n = seq[s].count;
        for (size_t t = 0; t < n; t++) {
            const float *sn = local ? w->sin_local[t] : w->sin_global[t];
            const float *cs = local ? w->cos_local[t] : w->cos_global[t];
            float *k = w->k + (first + t) * HEAD;
            rmsnorm(k, b->k_norm, HEAD, k);
            rope(k, sn, cs);
            for (size_t h = 0; h < HEADS; h++) {
                float *q = w->q + (first + t) * GEMMA_DIM + h * HEAD;
                rmsnorm(q, b->q_norm, HEAD, q);
                rope(q, sn, cs);
            }
        }
        for (size_t t = 0; t < n; t++) {
            size_t start = local && t > 256 ? t - 256 : 0;
            size_t end = local && t + 257 < n ? t + 257 : n;
            for (size_t h = 0; h < HEADS; h++) {
                const float *q = w->q + (first + t) * GEMMA_DIM + h * HEAD;
                float *out = w->attention + (first + t) * GEMMA_DIM + h * HEAD;
                for (size_t j = start; j < end; j++)
                    w->scores[j - start] = quant_bebelm_dot(q, w->k + (first + j) * HEAD, HEAD) * 0.0625f;
                softmax(w->scores, end - start);
                memset(out, 0, HEAD * sizeof(float));
                for (size_t j = start; j < end; j++) {
                    const float *v = w->v + (first + j) * HEAD;
                    float p = w->scores[j - start];
                    for (size_t d = 0; d < HEAD; d++) out[d] += p * v[d];
                }
            }
        }
    }
}

bool gemma_forward(const gemma_model *m, gemma_workspace *w,
                   const int32_t *ids, size_t count, const gemma_sequence *seq,
                   size_t ns, float *out, char *error) {
    if (!count || count > w->capacity || !ns || ns > count)
        return fail(error, "invalid EmbeddingGemma packed token count");
    size_t end = 0;
    for (size_t s = 0; s < ns; s++) {
        if (seq[s].first != end || !seq[s].count || seq[s].count > count - end)
            return fail(error, "invalid EmbeddingGemma sequence partition");
        end += seq[s].count;
    }
    if (end != count) return fail(error, "incomplete EmbeddingGemma sequence partition");
    float scale = sqrtf((float)GEMMA_DIM);
    for (size_t t = 0; t < count; t++) {
        if (ids[t] < 0 || ids[t] >= VOCAB) return fail(error, "EmbeddingGemma token id out of range");
        float *x = w->x + t * GEMMA_DIM;
        if (!quant_dequantize_row(m->embedding.type,
                                 m->embedding.data + (size_t)ids[t] * m->embedding.row_stride,
                                 m->embedding.row_stride, x, GEMMA_DIM, error)) return false;
        for (size_t i = 0; i < GEMMA_DIM; i++) x[i] *= scale;
    }
    for (size_t l = 0; l < LAYERS; l++) {
        const gemma_layer *b = &m->layers[l];
        norm_rows(w->x, b->pre_attn, count, w->norm);
        quant_bebelm_matmul(&b->q, w->norm, count, w->q, w->scratch);
        quant_bebelm_matmul(&b->k, w->norm, count, w->k, w->scratch);
        quant_bebelm_matmul(&b->v, w->norm, count, w->v, w->scratch);
        attention(w, b, seq, ns, l % 6 < 5);
        quant_bebelm_matmul(&b->output, w->attention, count, w->projected, w->scratch);
        norm_rows(w->projected, b->post_attn, count, w->norm);
        for (size_t i = 0; i < count * GEMMA_DIM; i++) w->x[i] += w->norm[i];
        norm_rows(w->x, b->pre_ff, count, w->norm);
        quant_bebelm_matmul(&b->gate, w->norm, count, w->gate, w->scratch);
        quant_bebelm_matmul(&b->up, w->norm, count, w->up, w->scratch);
        for (size_t i = 0; i < count * FF; i++) {
            float g = w->gate[i];
            float gelu = (0.5f * g) * (1.0f + tanhf((0.7978846f * g) * (1.0f + (0.044715f * g) * g)));
            w->gate[i] = gelu * w->up[i];
        }
        quant_bebelm_matmul(&b->down, w->gate, count, w->projected, w->scratch);
        norm_rows(w->projected, b->post_ff, count, w->norm);
        for (size_t i = 0; i < count * GEMMA_DIM; i++) w->x[i] += w->norm[i];
    }
    norm_rows(w->x, m->output_norm, count, w->norm);
    for (size_t s = 0; s < ns; s++) {
        memset(w->pooled, 0, sizeof(w->pooled));
        for (size_t t = seq[s].first; t < seq[s].first + seq[s].count; t++)
            for (size_t i = 0; i < GEMMA_DIM; i++) w->pooled[i] += w->norm[t * GEMMA_DIM + i];
        for (size_t i = 0; i < GEMMA_DIM; i++) w->pooled[i] /= (float)seq[s].count;
        quant_bebelm_matmul(&m->dense2, w->pooled, 1, w->dense, w->scratch);
        quant_bebelm_matmul(&m->dense3, w->dense, 1, out + s * GEMMA_DIM, w->scratch);
        for (size_t i = 0; i < GEMMA_DIM; i++)
            if (!isfinite(out[s * GEMMA_DIM + i])) return fail(error, "EmbeddingGemma produced nonfinite embedding");
    }
    return true;
}

void gemma_select(const float *full, int dimensions, bool normalize, float *out) {
    double ss = 0;
    for (int i = 0; i < dimensions; i++) ss += (double)full[i] * (double)full[i];
    double norm = sqrt(ss);
    for (int i = 0; i < dimensions; i++)
        out[i] = normalize && norm > 0 && isfinite(norm) ? (float)((double)full[i] / norm) : full[i];
}

typedef struct model_entry {
    char *path;
    off_t size;
    struct timespec mtime;
    gemma_model model;
    struct model_entry *next;
} model_entry;

static pthread_mutex_t cache_lock = PTHREAD_MUTEX_INITIALIZER;
static model_entry *cache;
static bool cleanup_registered;

static void cache_cleanup(void) {
    pthread_mutex_lock(&cache_lock);
    while (cache) {
        model_entry *entry = cache;
        cache = entry->next;
        tokenizer_destroy(entry->model.tokens);
        gguf_close(&entry->model.file);
        free(entry->path);
        free(entry);
    }
    pthread_mutex_unlock(&cache_lock);
}

bool gemma_model_cached(const char *path, const gemma_model **out, char *error) {
    struct stat st;
    *out = NULL;
    if (stat(path, &st)) return fail(error, "cannot stat EmbeddingGemma model path");
    if (!S_ISREG(st.st_mode)) return fail(error, "EmbeddingGemma model path is not a regular file");
    pthread_mutex_lock(&cache_lock);
    for (model_entry *e = cache; e; e = e->next) {
        if (e->size == st.st_size && e->mtime.tv_sec == st.st_mtim.tv_sec &&
            e->mtime.tv_nsec == st.st_mtim.tv_nsec && !strcmp(e->path, path)) {
            *out = &e->model;
            pthread_mutex_unlock(&cache_lock);
            return true;
        }
    }
    model_entry *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        pthread_mutex_unlock(&cache_lock);
        return fail(error, "out of memory allocating EmbeddingGemma model");
    }
    entry->path = strdup(path);
    bool ok = entry->path && model_init(&entry->model, path, error);
    if (!entry->path) fail(error, "out of memory copying EmbeddingGemma model path");
    struct stat after;
    if (ok && (stat(path, &after) || after.st_size != st.st_size ||
               after.st_mtim.tv_sec != st.st_mtim.tv_sec || after.st_mtim.tv_nsec != st.st_mtim.tv_nsec))
        ok = fail(error, "EmbeddingGemma model changed during load");
    if (ok && !cleanup_registered) {
        if (atexit(cache_cleanup)) ok = fail(error, "cannot register EmbeddingGemma cache cleanup");
        else cleanup_registered = true;
    }
    if (ok) {
        entry->size = st.st_size;
        entry->mtime = st.st_mtim;
        entry->next = cache;
        cache = entry;
        *out = &entry->model;
    } else {
        tokenizer_destroy(entry->model.tokens);
        gguf_close(&entry->model.file);
        free(entry->path);
        free(entry);
    }
    pthread_mutex_unlock(&cache_lock);
    return ok;
}
