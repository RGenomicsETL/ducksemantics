/* CPU quantization kernels
 * SPDX-License-Identifier: MIT
 *
 * Block layouts and conversion formulas: ggml (MIT), ggml-common.h,
 * ggml-quants.c and ggml-impl.h. K dot reduction: bebelm matmul.rs (MIT).
 * Byte access keeps mapped GGUF payloads independent of host alignment.
 */
#include "quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

static bool fail(char *error, const char *message) {
    snprintf(error, GGUF_ERROR_SIZE, "%s", message);
    return false;
}

static uint16_t le16(const uint8_t *p) {
    return (uint16_t)((unsigned)p[0] | (unsigned)p[1] << 8);
}

static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
           (uint32_t)p[3] << 24;
}

static float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static uint32_t to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static float load_float(const uint8_t *p) {
    return from_bits(le32(p));
}

static void store16(uint8_t *p, uint16_t u) {
    p[0] = (uint8_t)u;
    p[1] = (uint8_t)(u >> 8);
}

static void store_float(uint8_t *p, float f) {
    uint32_t u = to_bits(f);
    store16(p, (uint16_t)u);
    store16(p + 2, (uint16_t)(u >> 16));
}

static int signed8(uint8_t u) {
    return u < 128 ? u : (int)u - 256;
}

static int signed16(const uint8_t *p) {
    unsigned u = le16(p);
    return u < 32768 ? (int)u : (int)u - 65536;
}

float quant_f16_to_f32(uint16_t h) {
    uint32_t w = (uint32_t)h << 16;
    uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t twice = w + w;
    float normalized = from_bits((twice >> 4) + (UINT32_C(0xe0) << 23)) * 0x1p-112f;
    float denormalized = from_bits((twice >> 17) | (UINT32_C(126) << 23)) - 0.5f;
    return from_bits(sign | to_bits(twice < (UINT32_C(1) << 27) ? denormalized : normalized));
}

uint16_t quant_f32_to_f16(float f) {
    float base = (fabsf(f) * 0x1p112f) * 0x1p-110f;
    uint32_t w = to_bits(f), twice = w + w;
    uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = twice & UINT32_C(0xff000000);
    if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);
    base = from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    uint32_t bits = to_bits(base);
    uint32_t nonsign = ((bits >> 13) & 0x7c00) + (bits & 0x0fff);
    return (uint16_t)((sign >> 16) | (twice > UINT32_C(0xff000000) ? 0x7e00 : nonsign));
}

float quant_bf16_to_f32(uint16_t h) {
    return from_bits((uint32_t)h << 16);
}

uint16_t quant_f32_to_bf16(float f) {
    uint32_t u = to_bits(f);
    if ((u & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) return (uint16_t)((u >> 16) | 64);
    return (uint16_t)((u + (0x7fff + ((u >> 16) & 1))) >> 16);
}

bool quant_row_bytes(uint32_t type, uint64_t n, uint64_t *bytes) {
    switch (type) {
    case QUANT_F32: case QUANT_F16: case QUANT_BF16: case QUANT_Q8_0:
    case QUANT_Q2_K: case QUANT_Q3_K: case QUANT_Q4_K: case QUANT_Q5_K:
    case QUANT_Q6_K: case QUANT_Q8_K:
        break;
    default:
        return false;
    }
    const gguf_type_info *g = gguf_ggml_type(type);
    if (n % g->block_elements || n / g->block_elements > UINT64_MAX / g->block_bytes) return false;
    *bytes = n / g->block_elements * g->block_bytes;
    return *bytes <= SIZE_MAX;
}

bool quant_view_init(quant_view *v, const gguf_file *f, const gguf_tensor *t, char *error) {
    uint64_t stride;
    if (!t->n_dims || t->n_dims > GGUF_MAX_DIMS || !t->dims[0] ||
        !quant_row_bytes(t->type, t->dims[0], &stride)) {
        return fail(error, "unsupported quant type or invalid row width");
    }
    if (t->n_elements % t->dims[0]) return fail(error, "invalid tensor dimensions");
    uint64_t rows = t->n_elements / t->dims[0];
    if (!rows || rows > UINT64_MAX / stride || rows * stride != t->n_bytes ||
        t->offset > f->size || t->n_bytes > f->size - t->offset ||
        t->offset > SIZE_MAX || t->n_bytes > SIZE_MAX - t->offset) {
        return fail(error, "tensor payload out of bounds");
    }
    *v = (quant_view){.data = f->map + t->offset, .type = t->type,
                      .n_dims = t->n_dims, .columns = t->dims[0], .rows = rows,
                      .row_stride = stride, .n_bytes = t->n_bytes};
    memcpy(v->dims, t->dims, sizeof(v->dims));
    return true;
}

bool quant_view_row(const quant_view *v, uint64_t row, const uint8_t **data, char *error) {
    if (row >= v->rows || !v->row_stride || row > UINT64_MAX / v->row_stride) {
        return fail(error, "tensor row out of bounds");
    }
    uint64_t offset = row * v->row_stride;
    if (offset > v->n_bytes || v->row_stride > v->n_bytes - offset ||
        offset > SIZE_MAX) return fail(error, "tensor row out of bounds");
    *data = v->data + offset;
    return true;
}

/* q holds logical-order integers. Scales/minima apply to groups of 16 or 32. */
static void unpack_k(uint32_t type, const uint8_t *p, int8_t *q,
                      int *sc, int *mn, float *d, float *dm, int *group) {
    *dm = 0;
    *group = 16;
    memset(mn, 0, 16 * sizeof(*mn));
    if (type == QUANT_Q2_K || type == QUANT_Q3_K) {
        bool q3 = type == QUANT_Q3_K;
        const uint8_t *qs = p + (q3 ? 32 : 16);
        const uint8_t *s = q3 ? p + 96 : p;
        *d = quant_f16_to_f32(le16(p + (q3 ? 108 : 80)));
        if (!q3) *dm = quant_f16_to_f32(le16(p + 82));
        for (int j = 0; j < 16; j++) {
            sc[j] = q3 ? (((s[j % 8] >> (4 * (j / 8))) & 15) |
                         (((s[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32 : p[j] & 15;
            mn[j] = q3 ? 0 : p[j] >> 4;
        }
        for (int i = 0; i < 256; i++) {
            int shift = 2 * ((i % 128) / 32);
            int value = (qs[(i / 128) * 32 + i % 32] >> shift) & 3;
            if (q3 && !(p[i % 32] & (1u << (i / 32)))) value -= 4;
            q[i] = (int8_t)value;
        }
    } else if (type == QUANT_Q4_K || type == QUANT_Q5_K) {
        bool q5 = type == QUANT_Q5_K;
        const uint8_t *s = p + 4, *qs = p + (q5 ? 48 : 16);
        *group = 32;
        *d = quant_f16_to_f32(le16(p));
        *dm = quant_f16_to_f32(le16(p + 2));
        for (int j = 0; j < 8; j++) {
            sc[j] = j < 4 ? s[j] & 63 : (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4);
            mn[j] = j < 4 ? s[j + 4] & 63 : (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
        }
        for (int i = 0; i < 256; i++) {
            int value = (qs[(i / 64) * 32 + i % 32] >> (4 * ((i / 32) % 2))) & 15;
            if (q5 && (p[16 + i % 32] & (1u << (i / 32)))) value += 16;
            q[i] = (int8_t)value;
        }
    } else {
        *d = quant_f16_to_f32(le16(p + 208));
        for (int j = 0; j < 16; j++) sc[j] = signed8(p[192 + j]);
        for (int i = 0; i < 256; i++) {
            int quarter = (i % 128) / 32;
            int low = p[(i / 128) * 64 + (quarter % 2) * 32 + i % 32];
            low = (low >> (quarter >= 2 ? 4 : 0)) & 15;
            int high = (p[128 + (i / 128) * 32 + i % 32] >> (2 * quarter)) & 3;
            q[i] = (int8_t)((low | (high << 4)) - 32);
        }
    }
}

bool quant_dequantize_row(uint32_t type, const void *data, uint64_t bytes,
                          float *out, uint64_t n, char *error) {
    uint64_t need;
    if (!quant_row_bytes(type, n, &need) || need > bytes || n > SIZE_MAX / sizeof(float)) {
        return fail(error, "dequant row size out of bounds");
    }
    const uint8_t *p = data;
    if (type == QUANT_F32 || type == QUANT_F16 || type == QUANT_BF16) {
        unsigned step = type == QUANT_F32 ? 4 : 2;
        for (uint64_t i = 0; i < n; i++, p += step) {
            out[i] = type == QUANT_F32 ? load_float(p) :
                     type == QUANT_F16 ? quant_f16_to_f32(le16(p)) : quant_bf16_to_f32(le16(p));
        }
        return true;
    }
    const gguf_type_info *g = gguf_ggml_type(type);
    for (uint64_t i = 0; i < n; i += g->block_elements, p += g->block_bytes) {
        if (type == QUANT_Q8_0 || type == QUANT_Q8_K) {
            float d = type == QUANT_Q8_0 ? quant_f16_to_f32(le16(p)) : load_float(p);
            const uint8_t *q = p + (type == QUANT_Q8_0 ? 2 : 4);
            for (unsigned j = 0; j < g->block_elements; j++) out[i + j] = d * signed8(q[j]);
        } else {
            int8_t q[256];
            int sc[16], mn[16], group;
            float d, dm;
            unpack_k(type, p, q, sc, mn, &d, &dm, &group);
            bool affine = type == QUANT_Q2_K || type == QUANT_Q4_K || type == QUANT_Q5_K;
            for (int j = 0; j < 256; j++) {
                float value = (d * sc[j / group]) * q[j];
                out[i + j] = affine ? value - dm * mn[j / group] : value;
            }
        }
    }
    return true;
}

bool quant_values(const quant_view *v, uint64_t first, uint64_t count, float *out, char *error) {
    if (v->columns && v->rows > UINT64_MAX / v->columns) return fail(error, "invalid tensor size");
    uint64_t total = v->rows * v->columns;
    if (first > total || count > total - first || count > SIZE_MAX / sizeof(float)) {
        return fail(error, "tensor element range out of bounds");
    }
    const gguf_type_info *g = gguf_ggml_type(v->type);
    if (!g) return fail(error, "unsupported quant type");
    float block[256];
    while (count) {
        const uint8_t *row;
        if (!quant_view_row(v, first / v->columns, &row, error)) return false;
        uint64_t col = first % v->columns;
        uint64_t offset = col / g->block_elements * g->block_bytes;
        if (offset > v->row_stride || g->block_bytes > v->row_stride - offset ||
            !quant_dequantize_row(v->type, row + offset, g->block_bytes, block, g->block_elements, error)) {
            return fail(error, "tensor block out of bounds");
        }
        unsigned skip = (unsigned)(col % g->block_elements);
        uint64_t take = g->block_elements - skip;
        if (take > count) take = count;
        memcpy(out, block + skip, (size_t)take * sizeof(float));
        first += take;
        out += take;
        count -= take;
    }
    return true;
}

static bool activation_size(uint32_t type, const float *x, uint64_t n,
                             uint64_t bytes, char *error) {
    uint64_t need;
    if (!quant_row_bytes(type, n, &need) || need > bytes || n > SIZE_MAX / sizeof(float)) {
        return fail(error, "activation size out of bounds");
    }
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(x[i])) return fail(error, "activation must contain finite values");
    }
    return true;
}

bool quantize_row_q8_0(const float *x, uint64_t n, void *out, uint64_t bytes, char *error) {
    if (!activation_size(QUANT_Q8_0, x, n, bytes, error)) return false;
    uint8_t *p = out;
    for (uint64_t i = 0; i < n; i += 32, p += 34) {
        float amax = 0;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[i + j]));
        float d = amax / 127.f;
        float inv = d ? 1.f / d : 0;
        uint16_t h = quant_f32_to_f16(d);
        if (!isfinite(inv)) return fail(error, "activation scale out of range");
        store16(p, h);
        for (int j = 0; j < 32; j++) p[2 + j] = (uint8_t)(int)roundf(x[i + j] * inv);
    }
    return true;
}

bool quantize_row_q8_K(const float *x, uint64_t n, void *out, uint64_t bytes, char *error) {
    if (!activation_size(QUANT_Q8_K, x, n, bytes, error)) return false;
    uint8_t *p = out;
    for (uint64_t i = 0; i < n; i += 256, p += 292) {
        float amax = 0, max = 0;
        for (int j = 0; j < 256; j++) {
            float a = fabsf(x[i + j]);
            if (a > amax) { amax = a; max = x[i + j]; }
        }
        if (!amax) {
            /* ggml leaves zero-block bsums untouched; initialize them for dot consumers. */
            memset(p, 0, 292);
            continue;
        }
        float inv = -127.f / max;
        if (!isfinite(inv)) return fail(error, "activation scale out of range");
        for (int j = 0; j < 256; j++) {
            float rounded = x[i + j] * inv + 12582912.f;
            int q = (int)(to_bits(rounded) & 0x007fffff) - 0x00400000;
            p[4 + j] = (uint8_t)(q < 127 ? q : 127);
        }
        for (int j = 0; j < 16; j++) {
            int sum = 0;
            for (int k = 0; k < 16; k++) sum += signed8(p[4 + j * 16 + k]);
            store16(p + 260 + j * 2, (uint16_t)sum);
        }
        store_float(p, 1.f / inv);
    }
    return true;
}

bool quant_vec_dot(uint32_t type, const void *w, uint64_t w_bytes,
                    const void *x, uint64_t x_bytes, uint64_t n, float *out, char *error) {
    uint64_t need, a_need;
    uint32_t at = type == QUANT_Q8_0 ? QUANT_Q8_0 :
                  type >= QUANT_Q2_K && type <= QUANT_Q8_K ? QUANT_Q8_K : QUANT_F32;
    if (!quant_row_bytes(type, n, &need) || need > w_bytes ||
        !quant_row_bytes(at, n, &a_need) || a_need > x_bytes) return fail(error, "dot size out of bounds");
    const uint8_t *p = w, *a = x;
    float sum = 0;
    if (at == QUANT_F32) {
        const float *xf = x;
        unsigned step = type == QUANT_F32 ? 4 : 2;
        for (uint64_t i = 0; i < n; i++, p += step) {
            float value = type == QUANT_F32 ? load_float(p) :
                          type == QUANT_F16 ? quant_f16_to_f32(le16(p)) : quant_bf16_to_f32(le16(p));
            sum += value * xf[i];
        }
    } else if (at == QUANT_Q8_0) {
        for (uint64_t i = 0; i < n; i += 32, p += 34, a += 34) {
            int dot = 0;
            for (int j = 0; j < 32; j++) dot += signed8(p[2 + j]) * signed8(a[2 + j]);
            sum += dot * (quant_f16_to_f32(le16(p)) * quant_f16_to_f32(le16(a)));
        }
    } else {
        const gguf_type_info *g = gguf_ggml_type(type);
        for (uint64_t i = 0; i < n; i += 256, p += g->block_bytes, a += 292) {
            if (type == QUANT_Q8_K) {
                int dot = 0;
                for (int j = 0; j < 256; j++) dot += signed8(p[4 + j]) * signed8(a[4 + j]);
                sum += dot * (load_float(p) * load_float(a));
                continue;
            }
            int8_t q[256];
            int sc[16], mn[16], group;
            float d, dm;
            unpack_k(type, p, q, sc, mn, &d, &dm, &group);
            int sd = 0, sm = 0;
            for (int j = 0; j < 256 / group; j++) {
                int dot = 0, bsum = signed16(a + 260 + 2 * (j * group / 16));
                if (group == 32) bsum += signed16(a + 260 + 2 * (j * 2 + 1));
                for (int k = 0; k < group; k++) dot += q[j * group + k] * signed8(a[4 + j * group + k]);
                sd += sc[j] * dot;
                sm += mn[j] * bsum;
            }
            float value = d * sd;
            if (type == QUANT_Q2_K || type == QUANT_Q4_K || type == QUANT_Q5_K) value -= dm * sm;
            sum += load_float(a) * value;
        }
    }
    *out = sum;
    return true;
}

bool quant_bebel_quantize(const float *x, uint64_t n, quant_bebel_block *scratch,
                          uint64_t blocks, char *error) {
    if (n % 256 || blocks < n / 256 || !scratch)
        return fail(error, "bebelm activation scratch out of bounds");
    for (uint64_t b = 0; b < n / 256; b++) {
        quant_bebel_block *a = scratch + b;
        const float *xb = x + b * 256;
        float amax = 0;
        for (unsigned j = 0; j < 256; j++) {
            if (!isfinite(xb[j])) return fail(error, "activation must contain finite values");
            amax = fmaxf(amax, fabsf(xb[j]));
        }
        a->scale = amax / 127.f;
        float inv = amax > 0 ? 127.f / amax : 0;
        if (!isfinite(inv)) return fail(error, "activation scale out of range");
        memset(a->sums, 0, sizeof(a->sums));
        for (unsigned j = 0; j < 256; j++) {
            float q = roundf(xb[j] * inv);
            a->q[j] = (int8_t)fmaxf(-127.f, fminf(127.f, q));
            a->sums[j / 32] += a->q[j];
        }
    }
    return true;
}

static int bebel_int_dot(const int8_t *q, const int8_t *x, unsigned n) {
#if defined(__SSE2__)
    __m128i sum = _mm_setzero_si128(), zero = _mm_setzero_si128();
    for (unsigned j = 0; j < n; j += 16) {
        __m128i a = _mm_loadu_si128((const __m128i *)(q + j));
        __m128i b = _mm_loadu_si128((const __m128i *)(x + j));
        __m128i sa = _mm_cmpgt_epi8(zero, a), sb = _mm_cmpgt_epi8(zero, b);
        sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpacklo_epi8(a, sa), _mm_unpacklo_epi8(b, sb)));
        sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpackhi_epi8(a, sa), _mm_unpackhi_epi8(b, sb)));
    }
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
#else
    int sum = 0;
    for (unsigned j = 0; j < n; j++) sum += q[j] * x[j];
    return sum;
#endif
}

#if defined(__SSE2__)
static void q4_scale_min(const uint8_t *s, unsigned j, int *scale, int *min) {
    *scale = j < 4 ? s[j] & 63 : (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4);
    *min = j < 4 ? s[j + 4] & 63 : (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
}

static int packed_dot16(__m128i a, const int8_t *x, bool unsigned_a) {
    __m128i zero = _mm_setzero_si128(), b = _mm_loadu_si128((const __m128i *)x);
    __m128i sa = unsigned_a ? zero : _mm_cmpgt_epi8(zero, a), sb = _mm_cmpgt_epi8(zero, b);
    __m128i sum = _mm_add_epi32(
        _mm_madd_epi16(_mm_unpacklo_epi8(a, sa), _mm_unpacklo_epi8(b, sb)),
        _mm_madd_epi16(_mm_unpackhi_epi8(a, sa), _mm_unpackhi_epi8(b, sb)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

static float bebel_packed_dot(uint32_t type, const uint8_t *p, const quant_bebel_block *a) {
    __m128i mask = _mm_set1_epi8(15);
    int sd = 0, sm = 0;
    if (type == QUANT_Q4_K) {
        for (unsigned g = 0; g < 4; g++) {
            int lo = 0, hi = 0, sc0, sc1, mn0, mn1;
            q4_scale_min(p + 4, 2 * g, &sc0, &mn0); q4_scale_min(p + 4, 2 * g + 1, &sc1, &mn1);
            for (unsigned j = 0; j < 32; j += 16) {
                __m128i bits = _mm_loadu_si128((const __m128i *)(p + 16 + g * 32 + j));
                lo += packed_dot16(_mm_and_si128(bits, mask), a->q + g * 64 + j, true);
                hi += packed_dot16(_mm_and_si128(_mm_srli_epi16(bits, 4), mask), a->q + g * 64 + 32 + j, true);
            }
            sd += sc0 * lo + sc1 * hi;
            sm += mn0 * a->sums[2 * g] + mn1 * a->sums[2 * g + 1];
        }
        return a->scale * (quant_f16_to_f32(le16(p)) * sd - quant_f16_to_f32(le16(p + 2)) * sm);
    }
    __m128i two = _mm_set1_epi8(3), offset = _mm_set1_epi8(32);
    for (unsigned half = 0; half < 2; half++) {
        for (unsigned j = 0; j < 32; j += 16) {
            __m128i lo = _mm_loadu_si128((const __m128i *)(p + half * 64 + j));
            __m128i hi = _mm_loadu_si128((const __m128i *)(p + half * 64 + 32 + j));
            __m128i high = _mm_loadu_si128((const __m128i *)(p + 128 + half * 32 + j));
            __m128i q[4];
            q[0] = _mm_or_si128(_mm_and_si128(lo, mask), _mm_slli_epi16(_mm_and_si128(high, two), 4));
            q[1] = _mm_or_si128(_mm_and_si128(hi, mask), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(high, 2), two), 4));
            q[2] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(lo, 4), mask), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(high, 4), two), 4));
            q[3] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(hi, 4), mask), _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(high, 6), two), 4));
            for (unsigned g = 0; g < 4; g++) {
                int dot = packed_dot16(_mm_sub_epi8(q[g], offset), a->q + half * 128 + g * 32 + j, false);
                sd += signed8(p[192 + half * 8 + g * 2 + j / 16]) * dot;
            }
        }
    }
    return a->scale * (quant_f16_to_f32(le16(p + 208)) * sd);
}
#endif

static float bebel_reduce(const float *s) {
    return ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
}

static float bebel_float_dot(uint32_t type, const uint8_t *p, const float *x, uint64_t n) {
    float sum = 0;
    if (type == QUANT_Q4_K || type == QUANT_Q6_K) {
        unsigned bytes = type == QUANT_Q4_K ? 144 : 210;
        for (uint64_t i = 0; i < n; i += 256, p += bytes) {
            int8_t q[256];
            int sc[16], mn[16], group;
            float d, dm, block = 0;
            unpack_k(type, p, q, sc, mn, &d, &dm, &group);
            for (unsigned g = 0; g < 256 / (unsigned)group; g++) {
                float dot[8] = {0}, sx[8] = {0};
                for (unsigned j = 0; j < (unsigned)group; j++) {
                    unsigned k = g * group + j;
                    dot[j % 8] = fmaf(q[k], x[i + k], dot[j % 8]);
                    sx[j % 8] += x[i + k];
                }
                if (type == QUANT_Q4_K)
                    block += (d * sc[g]) * bebel_reduce(dot) - (dm * mn[g]) * bebel_reduce(sx);
                else block += sc[g] * bebel_reduce(dot);
            }
            sum += type == QUANT_Q6_K ? d * block : block;
        }
        return sum;
    }
    float lanes[4][8] = {{0}}, combined[8];
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        unsigned chain = i / 8 % 4;
        if (i >= n / 32 * 32) chain = 0;
        for (unsigned j = 0; j < 8; j++) {
            uint64_t index = i + j;
            float w = type == QUANT_F32 ? load_float(p + index * 4) :
                type == QUANT_F16 ? quant_f16_to_f32(le16(p + index * 2)) :
                quant_f16_to_f32(le16(p + index / 32 * 34)) * signed8(p[index / 32 * 34 + 2 + index % 32]);
            lanes[chain][j] = fmaf(w, x[index], lanes[chain][j]);
        }
    }
    for (unsigned j = 0; j < 8; j++) combined[j] = (lanes[0][j] + lanes[1][j]) + (lanes[2][j] + lanes[3][j]);
    sum = bebel_reduce(combined);
    for (; i < n; i++) {
        float w = type == QUANT_F32 ? load_float(p + i * 4) : quant_f16_to_f32(le16(p + i * 2));
        sum += w * x[i];
    }
    return sum;
}

bool quant_bebel_matvec(const quant_view *v, const float *x, bool integer_dot,
                        uint64_t first, uint64_t count, float *out,
                        quant_bebel_block *scratch, uint64_t blocks, char *error) {
    bool kquant = v->type == QUANT_Q4_K || v->type == QUANT_Q6_K;
    if ((!kquant && v->type != QUANT_F32 && v->type != QUANT_F16 && v->type != QUANT_Q8_0) ||
        first > v->rows || count > v->rows - first)
        return fail(error, "unsupported bebelm matvec type or rows");
    if (integer_dot && kquant) {
        if (!quant_bebel_quantize(x, v->columns, scratch, blocks, error)) return false;
    } else {
        for (uint64_t j = 0; j < v->columns; j++)
            if (!isfinite(x[j])) return fail(error, "activation must contain finite values");
    }
    for (uint64_t row = 0; row < count; row++) {
        const uint8_t *p = v->data + (first + row) * v->row_stride;
        if (!integer_dot || !kquant) {
            out[row] = bebel_float_dot(v->type, p, x, v->columns);
            continue;
        }
        float sum = 0;
        unsigned bytes = v->type == QUANT_Q4_K ? 144 : 210;
        for (uint64_t b = 0; b < v->columns / 256; b++, p += bytes) {
#if defined(__SSE2__)
            sum += bebel_packed_dot(v->type, p, scratch + b);
#else
            int8_t q[256];
            int sc[16], mn[16], group, sd = 0, sm = 0;
            float d, dm;
            const quant_bebel_block *a = scratch + b;
            unpack_k(v->type, p, q, sc, mn, &d, &dm, &group);
            for (unsigned g = 0; g < 256 / (unsigned)group; g++) {
                int dot = bebel_int_dot(q + g * group, a->q + g * group, (unsigned)group);
                sd += sc[g] * dot;
                if (v->type == QUANT_Q4_K) sm += mn[g] * a->sums[g];
            }
            float value = v->type == QUANT_Q4_K ?
                a->scale * (d * sd - dm * sm) : a->scale * (d * sd);
            sum += value;
#endif
        }
        out[row] = sum;
    }
    return true;
}

bool quant_bebel_matmul(const quant_view *v, const float *x, uint64_t tokens,
                        float *out, quant_bebel_block *scratch, uint64_t blocks, char *error) {
    if (v->type != QUANT_Q4_K && v->type != QUANT_Q6_K) {
        for (uint64_t t = 0; t < tokens; t++)
            if (!quant_bebel_matvec(v, x + t * v->columns, false, 0, v->rows, out + t * v->rows,
                                   scratch, blocks, error)) return false;
        return true;
    }
    uint64_t nb = v->columns / 256;
    if (v->columns % 256 || !nb || tokens > blocks / nb)
        return fail(error, "bebelm batch scratch out of bounds");
    for (uint64_t t = 0; t < tokens; t++)
        if (!quant_bebel_quantize(x + t * v->columns, v->columns, scratch + t * nb, nb, error)) return false;
    unsigned bytes = v->type == QUANT_Q4_K ? 144 : 210;
    for (uint64_t row = 0; row < v->rows; row++) {
        const uint8_t *p = v->data + row * v->row_stride;
        for (uint64_t t = 0; t < tokens; t++) out[t * v->rows + row] = 0;
        for (uint64_t b = 0; b < nb; b++, p += bytes) {
            int8_t q[256];
            int sc[16], mn[16], group;
            float d, dm;
            unpack_k(v->type, p, q, sc, mn, &d, &dm, &group);
            for (uint64_t t = 0; t < tokens; t++) {
                const quant_bebel_block *a = scratch + t * nb + b;
                int sd = 0, sm = 0;
                for (unsigned g = 0; g < 256 / (unsigned)group; g++) {
                    int dot = bebel_int_dot(q + g * group, a->q + g * group, (unsigned)group);
                    sd += sc[g] * dot;
                    if (v->type == QUANT_Q4_K) sm += mn[g] * a->sums[g];
                }
                float value = v->type == QUANT_Q4_K ?
                    a->scale * (d * sd - dm * sm) : a->scale * (d * sd);
                out[t * v->rows + row] += value;
            }
        }
    }
    return true;
}

bool quant_matvec_scratch(const quant_view *v, uint64_t *bytes) {
    uint32_t at = v->type == QUANT_Q8_0 ? QUANT_Q8_0 :
                  v->type >= QUANT_Q2_K && v->type <= QUANT_Q8_K ? QUANT_Q8_K : QUANT_F32;
    if (at == QUANT_F32) { *bytes = 0; return true; }
    return quant_row_bytes(at, v->columns, bytes);
}

bool quant_matvec(const quant_view *v, const float *x, uint64_t n,
                   uint64_t first, uint64_t count, float *y, uint64_t y_count,
                   void *scratch, uint64_t scratch_bytes, char *error) {
    uint64_t need;
    if (n != v->columns || n > SIZE_MAX / sizeof(float) || first > v->rows ||
        count > v->rows - first || count > y_count || count > SIZE_MAX / sizeof(float) ||
        !quant_matvec_scratch(v, &need) || need > scratch_bytes || (need && !scratch)) {
        return fail(error, "matvec dimensions or scratch out of bounds");
    }
    const void *a = x;
    uint64_t a_bytes = n * sizeof(float);
    if (need) {
        bool ok = v->type == QUANT_Q8_0 ? quantize_row_q8_0(x, n, scratch, scratch_bytes, error) :
                                        quantize_row_q8_K(x, n, scratch, scratch_bytes, error);
        if (!ok) return false;
        if (v->type == QUANT_Q8_0) {
            const uint8_t *p = scratch;
            for (uint64_t i = 0; i < n / 32; i++, p += 34) {
                if ((le16(p) & 0x7c00) == 0x7c00) return fail(error, "activation scale out of range");
            }
        }
        a = scratch;
        a_bytes = need;
    } else {
        for (uint64_t i = 0; i < n; i++) {
            if (!isfinite(x[i])) return fail(error, "activation must contain finite values");
        }
    }
    for (uint64_t row = 0; row < count; row++) {
        const uint8_t *w;
        if (!quant_view_row(v, first + row, &w, error) ||
            !quant_vec_dot(v->type, w, v->row_stride, a, a_bytes, n, y + row, error)) return false;
    }
    return true;
}
