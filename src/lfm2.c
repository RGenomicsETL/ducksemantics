/* LFM2 CPU profiles, following bebelm (MIT), Rbebelm 507a887.
 * SPDX-License-Identifier: MIT
 */
#define _POSIX_C_SOURCE 200809L
#include "lfm2.h"
#include "quant.h"
#include "tokenizer.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define CB_HIDDEN 1024
#define CB_LAYERS 16
#define CB_FF 4608
#define CB_VOCAB 64402
#define CB_HEADS 16
#define CB_KV_HEADS 8
#define CB_HEAD_DIM 64
#define CB_KV_DIM 512
#define CB_EPS 1e-5f
#define CB_THETA 1000000.f

struct lfm2_layer {
    bool attention;
    quant_view norm, ffn_norm, gate, up, down;
    quant_view q, k, v, qnorm, knorm, output;
    quant_view conv, in_proj, router, bias;
    float *norm_gain, *ffn_gain, *q_gain, *k_gain, *conv_weights, *bias_values;
};

struct lfm2_model {
    gguf_file file;
    const tokenizer *tokenizer;
    quant_view embedding, final_norm, projection;
    float *final_gain;
    struct lfm2_layer layers[LFM2_MOE_LAYERS];
    bool generator;
    bool skip[CB_VOCAB];
};

struct model_entry {
    char *path;
    off_t size;
    struct timespec mtime;
    lfm2_model model;
    struct model_entry *next;
};

static pthread_mutex_t model_mutex = PTHREAD_MUTEX_INITIALIZER;
static struct model_entry *model_cache;

static bool fail(char *error, const char *text) {
    snprintf(error, GGUF_ERROR_SIZE, "%s", text);
    return false;
}

float lfm2_dot(const float *a, const float *b, size_t n) {
    float s[4][8] = {{0}}, lanes[8], sum;
    size_t i = 0;
    for (; i + 32 <= n; i += 32)
        for (unsigned k = 0; k < 4; k++)
            for (unsigned j = 0; j < 8; j++) s[k][j] = fmaf(a[i + k * 8 + j], b[i + k * 8 + j], s[k][j]);
    for (; i + 8 <= n; i += 8)
        for (unsigned j = 0; j < 8; j++) s[0][j] = fmaf(a[i + j], b[i + j], s[0][j]);
    for (unsigned j = 0; j < 8; j++) lanes[j] = (s[0][j] + s[1][j]) + (s[2][j] + s[3][j]);
    sum = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void lfm2_rmsnorm(const float *x, const float *gain, size_t n, float epsilon, float *out) {
    double ss = 0;
    for (size_t i = 0; i < n; i++) ss += (double)(x[i] * x[i]);
    float mean = (float)(ss / (double)n);
    float scale = 1.f / sqrtf(mean + epsilon);
    for (size_t i = 0; i < n; i++) out[i] = (x[i] * scale) * gain[i];
}

void lfm2_rope(float *x, size_t n, size_t position, float theta) {
    size_t half = n / 2;
    float theta_scale = powf(theta, -2.f / (float)n), angle = (float)position;
    for (size_t i = 0; i < half; i++) {
        float c = cosf(angle), s = sinf(angle), a = x[i], b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
        angle *= theta_scale;
    }
}

void lfm2_conv_centered(const float *bx, const float *c, const float *weights,
                        size_t tokens, size_t hidden, size_t width, float *out) {
    size_t pad = width / 2;
    for (size_t t = 0; t < tokens; t++) {
        for (size_t h = 0; h < hidden; h++) {
            float sum = 0;
            for (size_t j = 0; j < width; j++) {
                size_t source = t + j;
                if (source >= pad && source - pad < tokens)
                    sum += weights[h * width + j] * bx[(source - pad) * hidden + h];
            }
            out[t * hidden + h] = sum * c[t * hidden + h];
        }
    }
}

void lfm2_conv_step(const float *bx, const float *c, const float *weights,
                    size_t hidden, size_t width, float *state, float *out) {
    for (size_t h = 0; h < hidden; h++) {
        float sum = weights[h * width + width - 1] * bx[h];
        for (size_t j = 0; j + 1 < width; j++) sum += weights[h * width + j] * state[j * hidden + h];
        out[h] = sum * c[h];
    }
    if (width > 1) {
        if (width > 2) memmove(state, state + hidden, (width - 2) * hidden * sizeof(float));
        memcpy(state + (width - 2) * hidden, bx, hidden * sizeof(float));
    }
}

static void softmax(float *x, size_t n) {
    float max = -INFINITY;
    double sum = 0;
    for (size_t i = 0; i < n; i++) max = fmaxf(max, x[i]);
    for (size_t i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    if (sum > 0) {
        float inv = (float)(1.0 / sum);
        for (size_t i = 0; i < n; i++) x[i] *= inv;
    }
}

void lfm2_attention(const float *q, const float *k, const float *v,
                    size_t tokens, size_t heads, size_t kv_heads, size_t dim,
                    float *scores, float *out) {
    size_t qdim = heads * dim, kdim = kv_heads * dim, group = heads / kv_heads;
    float scale = 1.f / sqrtf((float)dim);
    memset(out, 0, tokens * qdim * sizeof(float));
    for (size_t t = 0; t < tokens; t++) {
        for (size_t h = 0; h < heads; h++) {
            size_t kh = h / group;
            for (size_t j = 0; j < tokens; j++)
                scores[j] = lfm2_dot(q + t * qdim + h * dim, k + j * kdim + kh * dim, dim) * scale;
            softmax(scores, tokens);
            float *dest = out + t * qdim + h * dim;
            for (size_t j = 0; j < tokens; j++) {
                const float *value = v + j * kdim + kh * dim;
                for (size_t i = 0; i < dim; i++) dest[i] += scores[j] * value[i];
            }
        }
    }
}

static bool is_cb_attention(size_t layer) {
    return layer == 2 || layer == 5 || layer == 8 || layer == 10 || layer == 12 || layer == 14;
}

static bool metadata_number(const gguf_file *f, const char *key, double expected,
                             double tolerance, char *error) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    char buf[128];
    gguf_str text;
    if (!kv || kv->type == GGUF_STRING || kv->type == GGUF_BOOL ||
        !gguf_scalar_text(f, kv->type, kv->value_pos, buf, sizeof(buf), &text) ||
        !isfinite(strtod(buf, NULL)) || fabs(strtod(buf, NULL) - expected) > tolerance) {
        snprintf(error, GGUF_ERROR_SIZE, "unsupported LFM2 profile: %s", key);
        return false;
    }
    return true;
}

static bool metadata_is(const gguf_file *f, const char *key, const char *value, char *error) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    char buf[128];
    gguf_str text;
    if (!kv || !gguf_scalar_text(f, kv->type, kv->value_pos, buf, sizeof(buf), &text) ||
        text.len != strlen(value) || memcmp(text.ptr, value, text.len)) {
        snprintf(error, GGUF_ERROR_SIZE, "unsupported LFM2 profile: %s", key);
        return false;
    }
    return true;
}

static bool moe_attention(size_t layer) {
    return layer == 2 || layer == 6 || layer == 10 || layer == 14 || layer == 18 || layer == 21;
}

static bool attention_schedule(const gguf_file *f, const char *key, size_t layers, bool generator, char *error) {
    const gguf_kv *kv = gguf_find_kv(f, key);
    if (!kv || kv->type != GGUF_ARRAY || kv->array_len != layers ||
        kv->array_type == GGUF_FLOAT32 || kv->array_type >= GGUF_FLOAT64 ||
        kv->array_type == GGUF_BOOL || kv->array_type == GGUF_STRING || kv->array_type == GGUF_ARRAY)
        return fail(error, "unsupported LFM2 profile: head_count_kv");
    uint64_t pos = kv->value_pos;
    for (size_t i = 0; i < layers; i++) {
        char buf[128]; gguf_str text;
        bool attention = generator ? moe_attention(i) : is_cb_attention(i);
        if (!gguf_scalar_text(f, kv->array_type, pos, buf, sizeof(buf), &text) ||
            strtod(buf, NULL) != (attention ? 8 : 0) || !gguf_value_skip(f, kv->array_type, &pos))
            return fail(error, "unsupported LFM2 profile: attention layer schedule");
    }
    return true;
}

static bool cb_validate(const gguf_file *f, char *error) {
    static const struct { const char *key; double value, tolerance; } values[] = {
        {"lfm2.block_count", 16, 0}, {"lfm2.context_length", 128000, 0},
        {"lfm2.embedding_length", 1024, 0}, {"lfm2.embedding_length_out", 128, 0},
        {"lfm2.feed_forward_length", 4608, 0}, {"lfm2.attention.head_count", 16, 0},
        {"lfm2.vocab_size", 64402, 0}, {"lfm2.shortconv.l_cache", 3, 0},
        {"lfm2.rope.freq_base", 1000000, 1},
        {"lfm2.attention.layer_norm_rms_epsilon", 1e-5, 1e-9},
        {"tokenizer.ggml.bos_token_id", 1, 0}, {"tokenizer.ggml.padding_token_id", 7, 0}
    };
    if (!metadata_is(f, "general.architecture", "lfm2", error) ||
        !metadata_is(f, "lfm2.attention.causal", "false", error)) return false;
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++)
        if (!metadata_number(f, values[i].key, values[i].value, values[i].tolerance, error)) return false;
    return attention_schedule(f, "lfm2.attention.head_count_kv", CB_LAYERS, false, error);
}

static bool tensor_load(const gguf_file *f, const char *name, uint64_t columns,
                         uint64_t rows, uint64_t depth, bool vector, quant_view *view, char *error) {
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const gguf_tensor *t = f->tensors + i;
        if (t->name.len != strlen(name) || memcmp(t->name.ptr, name, t->name.len)) continue;
        if (t->n_dims != (vector ? 1u : depth > 1 ? 3u : 2u) || t->dims[0] != columns ||
            (!vector && t->dims[1] != rows) || (depth > 1 && t->dims[2] != depth)) {
            snprintf(error, GGUF_ERROR_SIZE, "LFM2 tensor shape mismatch: %s", name);
            return false;
        }
        if (t->type != QUANT_F32 && t->type != QUANT_F16 && t->type != QUANT_Q8_0 &&
            t->type != QUANT_Q4_K && t->type != QUANT_Q6_K) {
            snprintf(error, GGUF_ERROR_SIZE, "unsupported LFM2 tensor dtype: %s", name);
            return false;
        }
        return quant_view_init(view, f, t, error);
    }
    snprintf(error, GGUF_ERROR_SIZE, "missing LFM2 tensor: %s", name);
    return false;
}

static bool layer_tensor(const gguf_file *f, size_t layer, const char *suffix,
                          uint64_t columns, uint64_t rows, bool vector,
                          quant_view *view, char *error) {
    char name[96];
    snprintf(name, sizeof(name), "blk.%zu.%s.weight", layer, suffix);
    return tensor_load(f, name, columns, rows, 1, vector, view, error);
}

static bool small_f32(const quant_view *view, float **out, char *error) {
    if (view->type != QUANT_F32) return fail(error, "LFM2 norm/conv weights must be F32");
    size_t n = (size_t)(view->columns * view->rows);
    *out = malloc(n * sizeof(float));
    if (!*out) return fail(error, "out of memory for LFM2 small weights");
    return quant_values(view, 0, n, *out, error);
}

static void model_free(lfm2_model *m) {
    for (size_t i = 0; i < LFM2_MOE_LAYERS; i++) {
        struct lfm2_layer *l = m->layers + i;
        free(l->norm_gain); free(l->ffn_gain); free(l->q_gain); free(l->k_gain); free(l->conv_weights); free(l->bias_values);
    }
    free(m->final_gain);
    gguf_close(&m->file);
}

static bool cb_load(lfm2_model *m, const char *path, char *error) {
    gguf_file *f = &m->file;
    if (!cb_validate(f, error)) return false;
    if (!tensor_load(f, "token_embd.weight", CB_HIDDEN, CB_VOCAB, 1, false, &m->embedding, error) ||
        !tensor_load(f, "token_embd_norm.weight", CB_HIDDEN, 1, 1, true, &m->final_norm, error) ||
        !tensor_load(f, "dense_2.weight", CB_HIDDEN, LFM2_COLBERT_DIM, 1, false, &m->projection, error) ||
        !small_f32(&m->final_norm, &m->final_gain, error)) return false;
    for (size_t i = 0; i < CB_LAYERS; i++) {
        struct lfm2_layer *l = m->layers + i;
        l->attention = is_cb_attention(i);
        if (!layer_tensor(f, i, "attn_norm", CB_HIDDEN, 1, true, &l->norm, error) ||
            !layer_tensor(f, i, "ffn_norm", CB_HIDDEN, 1, true, &l->ffn_norm, error) ||
            !layer_tensor(f, i, "ffn_gate", CB_HIDDEN, CB_FF, false, &l->gate, error) ||
            !layer_tensor(f, i, "ffn_up", CB_HIDDEN, CB_FF, false, &l->up, error) ||
            !layer_tensor(f, i, "ffn_down", CB_FF, CB_HIDDEN, false, &l->down, error) ||
            !small_f32(&l->norm, &l->norm_gain, error) || !small_f32(&l->ffn_norm, &l->ffn_gain, error))
            return false;
        if (l->attention) {
            if (!layer_tensor(f, i, "attn_q", CB_HIDDEN, CB_HIDDEN, false, &l->q, error) ||
                !layer_tensor(f, i, "attn_k", CB_HIDDEN, CB_KV_DIM, false, &l->k, error) ||
                !layer_tensor(f, i, "attn_v", CB_HIDDEN, CB_KV_DIM, false, &l->v, error) ||
                !layer_tensor(f, i, "attn_output", CB_HIDDEN, CB_HIDDEN, false, &l->output, error) ||
                !layer_tensor(f, i, "attn_q_norm", CB_HEAD_DIM, 1, true, &l->qnorm, error) ||
                !layer_tensor(f, i, "attn_k_norm", CB_HEAD_DIM, 1, true, &l->knorm, error) ||
                !small_f32(&l->qnorm, &l->q_gain, error) || !small_f32(&l->knorm, &l->k_gain, error)) return false;
        } else {
            if (!layer_tensor(f, i, "shortconv.conv", 3, CB_HIDDEN, false, &l->conv, error) ||
                !layer_tensor(f, i, "shortconv.in_proj", CB_HIDDEN, 3 * CB_HIDDEN, false, &l->in_proj, error) ||
                !layer_tensor(f, i, "shortconv.out_proj", CB_HIDDEN, CB_HIDDEN, false, &l->output, error) ||
                !small_f32(&l->conv, &l->conv_weights, error)) return false;
        }
    }
    const gguf_kv *vocab = gguf_find_kv(f, "tokenizer.ggml.tokens");
    if (!vocab || vocab->type != GGUF_ARRAY || vocab->array_type != GGUF_STRING || vocab->array_len != CB_VOCAB)
        return fail(error, "ColBERT tokenizer vocabulary size mismatch");
    m->tokenizer = tokenizer_cached(path, error);
    if (!m->tokenizer) return false;
    const char *punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    for (size_t i = 0; punctuation[i]; i++) {
        int32_t *ids = NULL;
        size_t n = 0;
        if (!tokenizer_encode(m->tokenizer, punctuation + i, 1, false, &ids, &n, error)) return false;
        for (size_t j = 0; j < n; j++) if ((uint32_t)ids[j] < CB_VOCAB) m->skip[ids[j]] = true;
        free(ids);
    }
    return true;
}

static bool moe_load(lfm2_model *m, const char *path, char *error) {
    gguf_file *f = &m->file;
    static const struct { const char *key; double value, tolerance; } fields[] = {
        {"lfm2moe.block_count", 24, 0}, {"lfm2moe.context_length", 128000, 0},
        {"lfm2moe.embedding_length", 2048, 0}, {"lfm2moe.feed_forward_length", 7168, 0},
        {"lfm2moe.attention.head_count", 32, 0}, {"lfm2moe.rope.freq_base", 5000000, 1},
        {"lfm2moe.attention.layer_norm_rms_epsilon", 1e-5, 1e-9},
        {"lfm2moe.expert_count", 32, 0}, {"lfm2moe.expert_used_count", 4, 0},
        {"lfm2moe.expert_feed_forward_length", 1792, 0}, {"lfm2moe.leading_dense_block_count", 2, 0},
        {"lfm2moe.expert_gating_func", 2, 0}, {"lfm2moe.vocab_size", 128000, 0},
        {"lfm2moe.shortconv.l_cache", 3, 0}, {"tokenizer.ggml.bos_token_id", 124894, 0},
        {"tokenizer.ggml.eos_token_id", 124900, 0}, {"tokenizer.ggml.padding_token_id", 124893, 0}
    };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++)
        if (!metadata_number(f, fields[i].key, fields[i].value, fields[i].tolerance, error)) return false;
    if (!attention_schedule(f, "lfm2moe.attention.head_count_kv", 24, true, error)) return false;
    const gguf_kv *causal = gguf_find_kv(f, "lfm2moe.attention.causal");
    if (causal && !metadata_is(f, "lfm2moe.attention.causal", "true", error)) return false;
    if (!tensor_load(f, "token_embd.weight", 2048, 128000, 1, false, &m->embedding, error) ||
        !tensor_load(f, "token_embd_norm.weight", 2048, 1, 1, true, &m->final_norm, error) ||
        !small_f32(&m->final_norm, &m->final_gain, error)) return false;
    m->projection = m->embedding;
    for (size_t i = 0; i < 24; i++) {
        struct lfm2_layer *l = m->layers + i;
        l->attention = moe_attention(i);
        if (!layer_tensor(f, i, "attn_norm", 2048, 1, true, &l->norm, error) ||
            !layer_tensor(f, i, "ffn_norm", 2048, 1, true, &l->ffn_norm, error) ||
            !small_f32(&l->norm, &l->norm_gain, error) || !small_f32(&l->ffn_norm, &l->ffn_gain, error)) return false;
        if (i < 2) {
            if (!layer_tensor(f, i, "ffn_gate", 2048, 7168, false, &l->gate, error) ||
                !layer_tensor(f, i, "ffn_up", 2048, 7168, false, &l->up, error) ||
                !layer_tensor(f, i, "ffn_down", 7168, 2048, false, &l->down, error)) return false;
        } else {
            char name[96];
            if (!layer_tensor(f, i, "ffn_gate_inp", 2048, 32, false, &l->router, error)) return false;
            snprintf(name, sizeof(name), "blk.%zu.exp_probs_b.bias", i);
            if (!tensor_load(f, name, 32, 1, 1, true, &l->bias, error) ||
                !small_f32(&l->bias, &l->bias_values, error)) return false;
            snprintf(name, sizeof(name), "blk.%zu.ffn_gate_exps.weight", i);
            if (!tensor_load(f, name, 2048, 1792, 32, false, &l->gate, error)) return false;
            snprintf(name, sizeof(name), "blk.%zu.ffn_up_exps.weight", i);
            if (!tensor_load(f, name, 2048, 1792, 32, false, &l->up, error)) return false;
            snprintf(name, sizeof(name), "blk.%zu.ffn_down_exps.weight", i);
            if (!tensor_load(f, name, 1792, 2048, 32, false, &l->down, error)) return false;
        }
        if (l->attention) {
            if (!layer_tensor(f, i, "attn_q", 2048, 2048, false, &l->q, error) ||
                !layer_tensor(f, i, "attn_k", 2048, 512, false, &l->k, error) ||
                !layer_tensor(f, i, "attn_v", 2048, 512, false, &l->v, error) ||
                !layer_tensor(f, i, "attn_output", 2048, 2048, false, &l->output, error) ||
                !layer_tensor(f, i, "attn_q_norm", 64, 1, true, &l->qnorm, error) ||
                !layer_tensor(f, i, "attn_k_norm", 64, 1, true, &l->knorm, error) ||
                !small_f32(&l->qnorm, &l->q_gain, error) || !small_f32(&l->knorm, &l->k_gain, error)) return false;
        } else {
            if (!layer_tensor(f, i, "shortconv.conv", 3, 2048, false, &l->conv, error) ||
                !layer_tensor(f, i, "shortconv.in_proj", 2048, 6144, false, &l->in_proj, error) ||
                !layer_tensor(f, i, "shortconv.out_proj", 2048, 2048, false, &l->output, error) ||
                !small_f32(&l->conv, &l->conv_weights, error)) return false;
        }
    }
    const gguf_kv *vocab = gguf_find_kv(f, "tokenizer.ggml.tokens");
    if (!vocab || vocab->type != GGUF_ARRAY || vocab->array_type != GGUF_STRING || vocab->array_len != 128000)
        return fail(error, "LFM2-MoE tokenizer vocabulary size mismatch");
    m->tokenizer = tokenizer_cached(path, error);
    if (!m->tokenizer) return false;
    m->generator = true;
    return true;
}

static bool model_load(lfm2_model *m, const char *path, char *error) {
    if (!gguf_open(&m->file, path)) return fail(error, m->file.error);
    if (metadata_is(&m->file, "general.architecture", "lfm2moe", error)) return moe_load(m, path, error);
    return cb_load(m, path, error);
}

const lfm2_model *lfm2_model_get(const char *path, char *error) {
    struct stat st;
    if (stat(path, &st)) { snprintf(error, GGUF_ERROR_SIZE, "cannot stat LFM2 model: %s", path); return NULL; }
    pthread_mutex_lock(&model_mutex);
    for (struct model_entry *e = model_cache; e; e = e->next) {
        if (e->size == st.st_size && e->mtime.tv_sec == st.st_mtim.tv_sec &&
            e->mtime.tv_nsec == st.st_mtim.tv_nsec && !strcmp(e->path, path)) {
            pthread_mutex_unlock(&model_mutex);
            return &e->model;
        }
    }
    struct model_entry *e = calloc(1, sizeof(*e));
    if (!e) { pthread_mutex_unlock(&model_mutex); fail(error, "out of memory for LFM2 model"); return NULL; }
    e->path = strdup(path);
    if (!e->path || !model_load(&e->model, path, error)) {
        if (!e->path) fail(error, "out of memory for LFM2 path");
        model_free(&e->model); free(e->path); free(e);
        pthread_mutex_unlock(&model_mutex);
        return NULL;
    }
    e->size = st.st_size; e->mtime = st.st_mtim;
    e->next = model_cache; model_cache = e;
    pthread_mutex_unlock(&model_mutex);
    return &e->model;
}

struct lfm2_state {
    const lfm2_model *model;
    lfm2_cache cache;
    float *work, *scores, *logits;
    quant_bebel_block scratch[7168 / 256];
};

void lfm2_cache_free(lfm2_cache *cache) {
    for (size_t i = 0; i < 24; i++) { free(cache->k[i]); free(cache->v[i]); free(cache->conv[i]); }
    memset(cache, 0, sizeof(*cache));
}

bool lfm2_cache_init(lfm2_cache *cache, size_t capacity, char *error) {
    memset(cache, 0, sizeof(*cache));
    if (!capacity || capacity > LFM2_MAX_CONTEXT) return fail(error, "LFM2 cache capacity out of bounds");
    cache->capacity = capacity;
    for (size_t i = 0; i < 24; i++) {
        cache->conv[i] = calloc(2 * 2048, sizeof(float));
        if (moe_attention(i)) {
            cache->k[i] = calloc(capacity * 512, sizeof(float));
            cache->v[i] = calloc(capacity * 512, sizeof(float));
        }
        if (!cache->conv[i] || (moe_attention(i) && (!cache->k[i] || !cache->v[i]))) {
            lfm2_cache_free(cache); return fail(error, "out of memory for LFM2 cache");
        }
    }
    return true;
}

static void cache_copy(lfm2_cache *out, const lfm2_cache *source) {
    out->position = source->position;
    memcpy(out->norm, source->norm, sizeof(out->norm));
    for (size_t i = 0; i < 24; i++) {
        memcpy(out->conv[i], source->conv[i], 2 * 2048 * sizeof(float));
        if (moe_attention(i)) {
            memcpy(out->k[i], source->k[i], source->position * 512 * sizeof(float));
            memcpy(out->v[i], source->v[i], source->position * 512 * sizeof(float));
        }
    }
}

bool lfm2_cache_fork(lfm2_cache *out, const lfm2_cache *source, char *error) {
    if (out == source || source->position > source->capacity)
        return fail(error, "invalid LFM2 cache fork");
    if (!lfm2_cache_init(out, source->capacity, error)) return false;
    cache_copy(out, source);
    return true;
}

void lfm2_state_free(lfm2_state *state) {
    if (!state) return;
    lfm2_cache_free(&state->cache);
    free(state->work); free(state->scores); free(state->logits); free(state);
}

lfm2_state *lfm2_state_new(const lfm2_model *m, size_t capacity, char *error) {
    if (!m->generator) { fail(error, "generation requires the LFM2-MoE profile"); return NULL; }
    lfm2_state *s = calloc(1, sizeof(*s));
    if (!s) { fail(error, "out of memory for LFM2 state"); return NULL; }
    s->model = m;
    if (!lfm2_cache_init(&s->cache, capacity, error)) { lfm2_state_free(s); return NULL; }
    s->work = calloc(8 * 2048 + 2 * 7168, sizeof(float));
    s->scores = malloc(capacity * sizeof(float));
    s->logits = malloc(128000 * sizeof(float));
    if (!s->work || !s->scores || !s->logits) {
        lfm2_state_free(s); fail(error, "out of memory for LFM2 workspace"); return NULL;
    }
    return s;
}

lfm2_state *lfm2_state_fork(const lfm2_state *source, char *error) {
    lfm2_state *s = lfm2_state_new(source->model, source->cache.capacity, error);
    if (!s) return NULL;
    cache_copy(&s->cache, &source->cache);
    return s;
}

static bool project(lfm2_state *s, const quant_view *w, const float *x,
                     size_t first, size_t rows, float *out, char *error) {
    return quant_bebel_matvec(w, x, true, first, rows, out, s->scratch, 7168 / 256, error);
}

static bool moe_ffn(lfm2_state *s, const struct lfm2_layer *l, const float *x,
                     float *gate, float *up, float *down, float *out, char *error) {
    float scores[32], ranked[32]; unsigned order[32];
    if (!project(s, &l->router, x, 0, 32, scores, error)) return false;
    for (unsigned e = 0; e < 32; e++) {
        scores[e] = 1.f / (1.f + expf(-scores[e]));
        ranked[e] = scores[e] + l->bias_values[e]; order[e] = e;
        if (!isfinite(ranked[e])) return fail(error, "non-finite MoE router score");
    }
    for (unsigned i = 1; i < 32; i++) {
        unsigned e = order[i], j = i;
        while (j && ranked[e] > ranked[order[j - 1]]) { order[j] = order[j - 1]; j--; }
        order[j] = e;
    }
    float denom = 0;
    for (unsigned k = 0; k < 4; k++) denom += scores[order[k]];
    denom += 1e-6f;
    memset(out, 0, 2048 * sizeof(float));
    for (unsigned k = 0; k < 4; k++) {
        unsigned e = order[k]; float weight = scores[e] / denom;
        if (!project(s, &l->gate, x, e * 1792, 1792, gate, error) ||
            !project(s, &l->up, x, e * 1792, 1792, up, error)) return false;
        for (size_t j = 0; j < 1792; j++) gate[j] = (gate[j] / (1.f + expf(-gate[j]))) * up[j];
        if (!project(s, &l->down, gate, e * 2048, 2048, down, error)) return false;
        for (size_t j = 0; j < 2048; j++) out[j] += weight * down[j];
    }
    return true;
}

static bool forward_one(lfm2_state *s, int32_t token, char *error) {
    if (token < 0 || token >= 128000) return fail(error, "LFM2 token id outside vocabulary");
    size_t pos = s->cache.position;
    if (pos >= s->cache.capacity) return fail(error, "LFM2 context exceeds workspace limit");
    const lfm2_model *m = s->model;
    float *h = s->work, *norm = h + 2048, *op = norm + 2048, *proj = op + 2048;
    float *bx = proj + 6144, *att = bx + 2048, *gate = att + 2048, *up = gate + 7168;
    if (!quant_values(&m->embedding, (uint64_t)token * 2048, 2048, h, error)) return false;
    for (size_t i = 0; i < 24; i++) {
        const struct lfm2_layer *l = m->layers + i;
        lfm2_rmsnorm(h, l->norm_gain, 2048, 1e-5f, norm);
        if (l->attention) {
            float *q = proj, *k = q + 2048, *v = k + 512;
            if (!project(s, &l->q, norm, 0, 2048, q, error) ||
                !project(s, &l->k, norm, 0, 512, k, error) || !project(s, &l->v, norm, 0, 512, v, error)) return false;
            for (size_t j = 0; j < 32; j++) {
                lfm2_rmsnorm(q + j * 64, l->q_gain, 64, 1e-5f, q + j * 64);
                lfm2_rope(q + j * 64, 64, pos, 5000000.f);
            }
            for (size_t j = 0; j < 8; j++) {
                lfm2_rmsnorm(k + j * 64, l->k_gain, 64, 1e-5f, k + j * 64);
                lfm2_rope(k + j * 64, 64, pos, 5000000.f);
            }
            memcpy(s->cache.k[i] + pos * 512, k, 512 * sizeof(float));
            memcpy(s->cache.v[i] + pos * 512, v, 512 * sizeof(float));
            memset(att, 0, 2048 * sizeof(float));
            for (size_t head = 0; head < 32; head++) {
                size_t kh = head / 4;
                for (size_t t = 0; t <= pos; t++)
                    s->scores[t] = lfm2_dot(q + head * 64, s->cache.k[i] + t * 512 + kh * 64, 64) * .125f;
                softmax(s->scores, pos + 1);
                for (size_t t = 0; t <= pos; t++)
                    for (size_t j = 0; j < 64; j++) att[head * 64 + j] += s->scores[t] * s->cache.v[i][t * 512 + kh * 64 + j];
            }
        } else {
            if (!project(s, &l->in_proj, norm, 0, 6144, proj, error)) return false;
            for (size_t j = 0; j < 2048; j++) bx[j] = proj[j] * proj[4096 + j];
            lfm2_conv_step(bx, proj + 2048, l->conv_weights, 2048, 3, s->cache.conv[i], att);
        }
        if (!project(s, &l->output, att, 0, 2048, op, error)) return false;
        for (size_t j = 0; j < 2048; j++) h[j] += op[j];
        lfm2_rmsnorm(h, l->ffn_gain, 2048, 1e-5f, norm);
        if (i < 2) {
            if (!project(s, &l->gate, norm, 0, 7168, gate, error) ||
                !project(s, &l->up, norm, 0, 7168, up, error)) return false;
            for (size_t j = 0; j < 7168; j++) gate[j] = (gate[j] / (1.f + expf(-gate[j]))) * up[j];
            if (!project(s, &l->down, gate, 0, 2048, op, error)) return false;
        } else if (!moe_ffn(s, l, norm, gate, up, att, op, error)) return false;
        for (size_t j = 0; j < 2048; j++) h[j] += op[j];
    }
    lfm2_rmsnorm(h, m->final_gain, 2048, 1e-5f, s->cache.norm);
    s->cache.position++;
    return true;
}

bool lfm2_prefill(lfm2_state *s, const int32_t *ids, size_t count, char *error) {
    if (count > s->cache.capacity - s->cache.position) return fail(error, "LFM2 context exceeds workspace limit");
    for (size_t i = 0; i < count; i++) if (ids[i] < 0 || ids[i] >= 128000) return fail(error, "LFM2 token id outside vocabulary");
    for (size_t i = 0; i < count; i++) if (!forward_one(s, ids[i], error)) return false;
    return true;
}

bool lfm2_prompt_ids(const lfm2_model *m, const char *text, size_t bytes, bool bos,
                     int32_t **ids, size_t *count, char *error) {
    if (!m->generator) return fail(error, "generation requires the LFM2-MoE profile");
    if (bos && !bytes) return fail(error, "LFM2 prompt must be non-empty");
    if (bytes > LFM2_MAX_INPUT_BYTES) return fail(error, "LFM2 prompt exceeds input byte limit");
    return tokenizer_encode(m->tokenizer, text, bytes, bos, ids, count, error);
}

bool lfm2_state_logits(lfm2_state *s, const int32_t *ids, size_t count, float *out, char *error) {
    if (!s->cache.position) return fail(error, "LFM2 logits require a prefilled state");
    if (count > 128000) return fail(error, "LFM2 requested logits exceed vocabulary size");
    for (size_t i = 0; i < count; i++) if (ids[i] < 0 || ids[i] >= 128000) return fail(error, "LFM2 requested token id outside vocabulary");
    for (size_t i = 0; i < count; i++)
        if (!project(s, &s->model->projection, s->cache.norm, ids[i], 1, out + i, error)) return false;
    return true;
}

static double monotonic_seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

void lfm2_generation_free(lfm2_generation *out) {
    free(out->ids); free(out->margins); free(out->text); memset(out, 0, sizeof(*out));
}

bool lfm2_decode(lfm2_state *s, size_t max_tokens, lfm2_generation *out, char *error) {
    memset(out, 0, sizeof(*out));
    if (!s->cache.position || !max_tokens || max_tokens > LFM2_MAX_GENERATED ||
        max_tokens > s->cache.capacity - s->cache.position + 1)
        return fail(error, "LFM2 generation length out of bounds");
    out->ids = malloc(max_tokens * sizeof(int32_t)); out->margins = malloc(max_tokens * sizeof(float));
    if (!out->ids || !out->margins) { lfm2_generation_free(out); return fail(error, "out of memory for LFM2 generation"); }
    double projection_start = monotonic_seconds();
    if (!project(s, &s->model->projection, s->cache.norm, 0, 128000, s->logits, error)) {
        lfm2_generation_free(out); return false;
    }
    out->prefill_seconds = monotonic_seconds() - projection_start;
    double start = monotonic_seconds(); bool ok = true, require_answer = false;
    for (size_t step = 0; step < max_tokens; step++) {
        if (require_answer) {
            s->logits[124900] = s->logits[124893] = s->logits[124894] = s->logits[124895] = s->logits[124899] = -INFINITY;
            require_answer = false;
        }
        int32_t best = 0; float first = -INFINITY, second = -INFINITY;
        for (int32_t id = 0; id < 128000; id++) {
            float value = s->logits[id];
            if (value > first) { second = first; first = value; best = id; }
            else if (value > second) second = value;
        }
        out->terminal_id = best; out->terminal_margin = first - second;
        if (best == 124900 || best == 124893 || best == 124894 || best == 124895 || best == 124899) {
            out->stopped = true; break;
        }
        out->ids[out->count] = best; out->margins[out->count] = first - second; out->count++;
        if (best == 124902) require_answer = true;
        if (step + 1 < max_tokens && (!forward_one(s, best, error) ||
            !project(s, &s->model->projection, s->cache.norm, 0, 128000, s->logits, error))) { ok = false; break; }
    }
    out->decode_seconds = monotonic_seconds() - start;
    if (ok) ok = tokenizer_decode(s->model->tokenizer, out->ids, out->count, &out->text, &out->bytes, error);
    if (!ok) lfm2_generation_free(out);
    return ok;
}

bool lfm2_generate(const lfm2_model *m, const char *prompt, size_t bytes,
                   size_t max_tokens, lfm2_generation *out, char *error) {
    memset(out, 0, sizeof(*out));
    if (!max_tokens || max_tokens > LFM2_MAX_GENERATED) return fail(error, "LFM2 generation length out of bounds");
    int32_t *ids = NULL; size_t count = 0;
    if (!lfm2_prompt_ids(m, prompt, bytes, true, &ids, &count, error)) return false;
    if (!count || count > LFM2_MAX_CONTEXT - max_tokens) {
        free(ids); return fail(error, "LFM2 context exceeds workspace limit");
    }
    lfm2_state *s = lfm2_state_new(m, count + max_tokens, error);
    if (!s) { free(ids); return false; }
    double start = monotonic_seconds();
    bool ok = lfm2_prefill(s, ids, count, error);
    double prefill = monotonic_seconds() - start;
    if (ok) ok = lfm2_decode(s, max_tokens, out, error);
    if (ok) { out->prefill_seconds += prefill; out->prompt_tokens = count; }
    free(ids); lfm2_state_free(s); return ok;
}

static bool batch(const quant_view *w, const float *x, size_t tokens, float *out,
                   quant_bebel_block *scratch, char *error) {
    return quant_bebel_matmul(w, x, tokens, out, scratch, tokens * (CB_FF / 256), error);
}

static void norm_batch(const float *x, const float *gain, size_t tokens, float *out) {
    for (size_t t = 0; t < tokens; t++)
        lfm2_rmsnorm(x + t * CB_HIDDEN, gain, CB_HIDDEN, CB_EPS, out + t * CB_HIDDEN);
}

static void norm_rope(float *x, const float *gain, size_t tokens, size_t heads) {
    for (size_t t = 0; t < tokens; t++) {
        for (size_t h = 0; h < heads; h++) {
            float *row = x + (t * heads + h) * CB_HEAD_DIM;
            lfm2_rmsnorm(row, gain, CB_HEAD_DIM, CB_EPS, row);
            lfm2_rope(row, CB_HEAD_DIM, t, CB_THETA);
        }
    }
}

void lfm2_embeddings_free(lfm2_embeddings *out) {
    free(out->values); free(out->ids);
    memset(out, 0, sizeof(*out));
}

bool lfm2_colbert_encode(const lfm2_model *m, const char *text, size_t bytes,
                         bool query, lfm2_embeddings *out, char *error) {
    int32_t *encoded = NULL;
    size_t count = 0, tokens;
    memset(out, 0, sizeof(*out));
    if (m->generator) return fail(error, "ColBERT requires the LFM2 encoder profile");
    if (!bytes) return fail(error, "ColBERT text must be non-empty");
    if (bytes > LFM2_MAX_INPUT_BYTES - 4) return fail(error, "ColBERT text exceeds input byte limit");
    char *prefixed = malloc(bytes + 5);
    if (!prefixed) return fail(error, "out of memory for ColBERT prefix");
    memcpy(prefixed, query ? "[Q] " : "[D] ", 4);
    memcpy(prefixed + 4, text, bytes); prefixed[bytes + 4] = 0;
    bool ok = tokenizer_encode(m->tokenizer, prefixed, bytes + 4, true, &encoded, &count, error);
    free(prefixed);
    if (!ok) return false;
    tokens = query ? 32 : (count < LFM2_COLBERT_MAX_TOKENS ? count : LFM2_COLBERT_MAX_TOKENS);
    out->ids = malloc(tokens * sizeof(uint32_t));
    out->values = malloc(tokens * LFM2_COLBERT_DIM * sizeof(float));
    /* Hidden, normalized, operator, 3-way conv/QKV, BX, attention, two FF columns. */
    size_t floats = tokens * (8 * CB_HIDDEN + 2 * CB_FF) + tokens;
    float *work = calloc(floats, sizeof(float));
    quant_bebel_block *scratch = malloc(tokens * (CB_FF / 256) * sizeof(*scratch));
    if (!out->ids || !out->values || !work || !scratch) {
        free(encoded); free(work); free(scratch); lfm2_embeddings_free(out);
        return fail(error, "out of memory for ColBERT workspace");
    }
    float *hidden = work, *normed = hidden + tokens * CB_HIDDEN;
    float *op = normed + tokens * CB_HIDDEN, *proj = op + tokens * CB_HIDDEN;
    float *bx = proj + tokens * 3 * CB_HIDDEN, *att = bx + tokens * CB_HIDDEN;
    float *gate = att + tokens * CB_HIDDEN, *up = gate + tokens * CB_FF;
    float *scores = up + tokens * CB_FF;
    for (size_t t = 0; t < tokens; t++) {
        out->ids[t] = t < count ? (uint32_t)encoded[t] : 7;
        if (out->ids[t] >= CB_VOCAB) {
            ok = fail(error, "ColBERT token id outside model vocabulary"); goto done;
        }
        if (!quant_values(&m->embedding, out->ids[t] * CB_HIDDEN, CB_HIDDEN, hidden + t * CB_HIDDEN, error)) {
            ok = false; goto done;
        }
    }
    for (size_t i = 0; i < CB_LAYERS; i++) {
        const struct lfm2_layer *l = m->layers + i;
        norm_batch(hidden, l->norm_gain, tokens, normed);
        if (l->attention) {
            float *q = proj, *k = q + tokens * CB_HIDDEN, *v = k + tokens * CB_KV_DIM;
            if (!batch(&l->q, normed, tokens, q, scratch, error) ||
                !batch(&l->k, normed, tokens, k, scratch, error) ||
                !batch(&l->v, normed, tokens, v, scratch, error)) { ok = false; goto done; }
            norm_rope(q, l->q_gain, tokens, CB_HEADS);
            norm_rope(k, l->k_gain, tokens, CB_KV_HEADS);
            lfm2_attention(q, k, v, tokens, CB_HEADS, CB_KV_HEADS, CB_HEAD_DIM, scores, att);
        } else {
            if (!batch(&l->in_proj, normed, tokens, proj, scratch, error)) { ok = false; goto done; }
            for (size_t t = 0; t < tokens; t++) {
                const float *p = proj + t * 3 * CB_HIDDEN;
                for (size_t h = 0; h < CB_HIDDEN; h++) bx[t * CB_HIDDEN + h] = p[h] * p[2 * CB_HIDDEN + h];
                memcpy(op + t * CB_HIDDEN, p + CB_HIDDEN, CB_HIDDEN * sizeof(float));
            }
            lfm2_conv_centered(bx, op, l->conv_weights, tokens, CB_HIDDEN, 3, att);
        }
        if (!batch(&l->output, att, tokens, op, scratch, error)) { ok = false; goto done; }
        for (size_t j = 0; j < tokens * CB_HIDDEN; j++) hidden[j] += op[j];
        norm_batch(hidden, l->ffn_gain, tokens, normed);
        if (!batch(&l->gate, normed, tokens, gate, scratch, error) ||
            !batch(&l->up, normed, tokens, up, scratch, error)) { ok = false; goto done; }
        for (size_t j = 0; j < tokens * CB_FF; j++) gate[j] = (gate[j] / (1.f + expf(-gate[j]))) * up[j];
        if (!batch(&l->down, gate, tokens, op, scratch, error)) { ok = false; goto done; }
        for (size_t j = 0; j < tokens * CB_HIDDEN; j++) hidden[j] += op[j];
    }
    norm_batch(hidden, m->final_gain, tokens, normed);
    if (!batch(&m->projection, normed, tokens, out->values, scratch, error)) { ok = false; goto done; }
    out->count = 0;
    for (size_t t = 0; t < tokens; t++) {
        float *row = out->values + t * LFM2_COLBERT_DIM;
        float ss = 0;
        for (size_t j = 0; j < LFM2_COLBERT_DIM; j++) ss += row[j] * row[j];
        float norm = sqrtf(ss);
        if (!isfinite(norm)) { fail(error, "non-finite ColBERT output"); ok = false; goto done; }
        if (norm > 0) for (size_t j = 0; j < LFM2_COLBERT_DIM; j++) row[j] /= norm;
        if (query || !m->skip[out->ids[t]]) {
            out->ids[out->count] = out->ids[t];
            memmove(out->values + out->count * LFM2_COLBERT_DIM, row, LFM2_COLBERT_DIM * sizeof(float));
            out->count++;
        }
    }
done:
    free(encoded); free(work); free(scratch);
    if (!ok) lfm2_embeddings_free(out);
    return ok;
}
