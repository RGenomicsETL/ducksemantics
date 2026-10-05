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
    quant_view conv, in_proj;
    float *norm_gain, *ffn_gain, *q_gain, *k_gain, *conv_weights;
};

struct lfm2_model {
    gguf_file file;
    const tokenizer *tokenizer;
    quant_view embedding, final_norm, projection;
    float *final_gain;
    struct lfm2_layer layers[CB_LAYERS];
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
    if (width > 1) memmove(state, state + hidden, (width - 1) * hidden * sizeof(float));
    memcpy(state + (width - 1) * hidden, bx, hidden * sizeof(float));
    for (size_t h = 0; h < hidden; h++) {
        float sum = 0;
        for (size_t j = 0; j < width; j++) sum += weights[h * width + j] * state[j * hidden + h];
        out[h] = sum * c[h];
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
    const gguf_kv *kv = gguf_find_kv(f, "lfm2.attention.head_count_kv");
    if (!kv || kv->type != GGUF_ARRAY || kv->array_len != CB_LAYERS ||
        kv->array_type == GGUF_FLOAT32 || kv->array_type >= GGUF_FLOAT64 ||
        kv->array_type == GGUF_BOOL || kv->array_type == GGUF_STRING || kv->array_type == GGUF_ARRAY)
        return fail(error, "unsupported LFM2 profile: head_count_kv");
    uint64_t pos = kv->value_pos;
    for (size_t i = 0; i < CB_LAYERS; i++) {
        char buf[128];
        gguf_str text;
        if (!gguf_scalar_text(f, kv->array_type, pos, buf, sizeof(buf), &text) ||
            strtod(buf, NULL) != (is_cb_attention(i) ? CB_KV_HEADS : 0) ||
            !gguf_value_skip(f, kv->array_type, &pos))
            return fail(error, "unsupported LFM2 profile: attention layer schedule");
    }
    return true;
}

static bool tensor_load(const gguf_file *f, const char *name, uint64_t columns,
                         uint64_t rows, bool vector, quant_view *view, char *error) {
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const gguf_tensor *t = f->tensors + i;
        if (t->name.len != strlen(name) || memcmp(t->name.ptr, name, t->name.len)) continue;
        if (t->n_dims != (vector ? 1u : 2u) || t->dims[0] != columns ||
            (!vector && t->dims[1] != rows)) {
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
    return tensor_load(f, name, columns, rows, vector, view, error);
}

static bool small_f32(const quant_view *view, float **out, char *error) {
    if (view->type != QUANT_F32) return fail(error, "LFM2 norm/conv weights must be F32");
    size_t n = (size_t)(view->columns * view->rows);
    *out = malloc(n * sizeof(float));
    if (!*out) return fail(error, "out of memory for LFM2 small weights");
    return quant_values(view, 0, n, *out, error);
}

static void model_free(lfm2_model *m) {
    for (size_t i = 0; i < CB_LAYERS; i++) {
        struct lfm2_layer *l = m->layers + i;
        free(l->norm_gain); free(l->ffn_gain); free(l->q_gain); free(l->k_gain); free(l->conv_weights);
    }
    free(m->final_gain);
    gguf_close(&m->file);
}

static bool cb_load(lfm2_model *m, const char *path, char *error) {
    if (!gguf_open(&m->file, path)) return fail(error, m->file.error);
    gguf_file *f = &m->file;
    if (!cb_validate(f, error)) return false;
    if (!tensor_load(f, "token_embd.weight", CB_HIDDEN, CB_VOCAB, false, &m->embedding, error) ||
        !tensor_load(f, "token_embd_norm.weight", CB_HIDDEN, 1, true, &m->final_norm, error) ||
        !tensor_load(f, "dense_2.weight", CB_HIDDEN, LFM2_COLBERT_DIM, false, &m->projection, error) ||
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
    if (!e->path || !cb_load(&e->model, path, error)) {
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
