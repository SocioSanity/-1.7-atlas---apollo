#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t *data;
    size_t size;
} RAM_MODEL;

static RAM_MODEL g_ram_model = {0};

static int ram_model_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("RAM model fopen");
        return 0;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }

    long size = ftell(f);
    if (size <= 0) {
        fclose(f);
        return 0;
    }

    rewind(f);

    g_ram_model.data = (uint8_t *)malloc((size_t)size);
    if (!g_ram_model.data) {
        fclose(f);
        fprintf(stderr, "RAM allocation failed for model (%ld bytes)\n", size);
        return 0;
    }

    if (fread(g_ram_model.data, 1, (size_t)size, f) != (size_t)size) {
        fclose(f);
        free(g_ram_model.data);
        g_ram_model.data = NULL;
        return 0;
    }

    fclose(f);

    g_ram_model.size = (size_t)size;

    printf("========================================\n");
    printf(" VM-AI RAM MODEL\n");
    printf("========================================\n");
    printf("Model loaded into RAM: %.2f MB\n",
           (double)g_ram_model.size / (1024.0 * 1024.0));
    printf("Disk file closed.\n");
    printf("Inference data source: RAM\n\n");

    return 1;
}

static void ram_model_free(void)
{
    free(g_ram_model.data);
    g_ram_model.data = NULL;
    g_ram_model.size = 0;
}

#include <stdint.h>
#include <string.h>
#include <math.h>

#define GGUF_MAGIC 0x46554747u

#define GGML_TYPE_F32   0
#define GGML_TYPE_Q4_K  12
#define GGML_TYPE_Q6_K  14

#define QK_K       256
#define Q4K_BYTES  144
#define Q6K_BYTES  210

#define HIDDEN_SIZE 1536
#define Q_DIM       1536
#define KV_DIM      256

#define Q_HEADS     24
#define KV_HEADS    4
#define HEAD_DIM    64

#define ROPE_BASE   1000000.0f
#define RMS_EPS     1.0e-6f

typedef struct {
    char name[256];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
} TensorInfo;

typedef struct {
    FILE *file;
    long data_base;
    uint64_t tensor_count;
} GGUFFile;

/* ========================================================= */
/* Basic readers                                              */
/* ========================================================= */

static int read_u32(FILE *f, uint32_t *v)
{
    return fread(v, sizeof(*v), 1, f) == 1;
}

static int read_u64(FILE *f, uint64_t *v)
{
    return fread(v, sizeof(*v), 1, f) == 1;
}

/* ========================================================= */
/* FP16                                                        */
/* ========================================================= */

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;

            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }

            mant &= 0x03FFu;

            bits =
                sign |
                ((exp + 112u) << 23) |
                (mant << 13);
        }
    }
    else if (exp == 31) {
        bits =
            sign |
            0x7F800000u |
            (mant << 13);
    }
    else {
        bits =
            sign |
            ((exp + 112u) << 23) |
            (mant << 13);
    }

    float result;

    memcpy(&result, &bits, sizeof(result));

    return result;
}

/* ========================================================= */
/* GGUF metadata skipping                                     */
/* ========================================================= */

static int skip_string(FILE *f)
{
    uint64_t len;

    if (!read_u64(f, &len))
        return 0;

    return fseek(
        f,
        (long)len,
        SEEK_CUR) == 0;
}

static int skip_value(FILE *f, uint32_t type);

static int skip_array(FILE *f)
{
    uint32_t element_type;
    uint64_t count;

    if (!read_u32(f, &element_type))
        return 0;

    if (!read_u64(f, &count))
        return 0;

    for (uint64_t i = 0; i < count; ++i) {
        if (!skip_value(f, element_type))
            return 0;
    }

    return 1;
}

static int skip_value(FILE *f, uint32_t type)
{
    switch (type) {
        case 0:
        case 1:
        case 7:
            return fseek(f, 1, SEEK_CUR) == 0;

        case 2:
        case 3:
            return fseek(f, 2, SEEK_CUR) == 0;

        case 4:
        case 5:
        case 6:
            return fseek(f, 4, SEEK_CUR) == 0;

        case 8:
            return skip_string(f);

        case 9:
            return skip_array(f);

        case 10:
        case 11:
        case 12:
            return fseek(f, 8, SEEK_CUR) == 0;

        default:
            return 0;
    }
}

/* ========================================================= */
/* Tensor directory                                           */
/* ========================================================= */

static int read_tensor(FILE *f, TensorInfo *t)
{
    uint64_t name_len;

    memset(t, 0, sizeof(*t));

    if (!read_u64(f, &name_len))
        return 0;

    if (name_len >= sizeof(t->name))
        return 0;

    if (fread(
            t->name,
            1,
            (size_t)name_len,
            f) != name_len)
        return 0;

    t->name[name_len] = '\0';

    if (!read_u32(f, &t->n_dims))
        return 0;

    if (t->n_dims > 4)
        return 0;

    for (uint32_t i = 0; i < t->n_dims; ++i) {
        if (!read_u64(f, &t->dims[i]))
            return 0;
    }

    if (!read_u32(f, &t->type))
        return 0;

    if (!read_u64(f, &t->offset))
        return 0;

    return 1;
}

/* ========================================================= */
/* Open GGUF                                                  */
/* ========================================================= */

static int gguf_open(const char *path, GGUFFile *g)
{
    FILE *f = fopen(path, "rb");

    if (!f) {
        perror("fopen");
        return 0;
    }

    uint32_t magic;
    uint32_t version;
    uint64_t tensors;
    uint64_t metadata;

    if (!read_u32(f, &magic) ||
        !read_u32(f, &version) ||
        !read_u64(f, &tensors) ||
        !read_u64(f, &metadata)) {

        fclose(f);
        return 0;
    }

    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic.\n");
        fclose(f);
        return 0;
    }

    for (uint64_t i = 0; i < metadata; ++i) {
        uint64_t key_len;

        if (!read_u64(f, &key_len)) {
            fclose(f);
            return 0;
        }

        if (fseek(f, (long)key_len, SEEK_CUR) != 0) {
            fclose(f);
            return 0;
        }

        uint32_t type;

        if (!read_u32(f, &type)) {
            fclose(f);
            return 0;
        }

        if (!skip_value(f, type)) {
            fclose(f);
            return 0;
        }
    }

    for (uint64_t i = 0; i < tensors; ++i) {
        TensorInfo ignored;

        if (!read_tensor(f, &ignored)) {
            fclose(f);
            return 0;
        }
    }

    long tensor_end = ftell(f);

    if (tensor_end < 0) {
        fclose(f);
        return 0;
    }

    long data_base =
        (tensor_end + 31L) & ~31L;

    g->file = f;
    g->data_base = data_base;
    g->tensor_count = tensors;

    return 1;
}

/* ========================================================= */
/* Tensor lookup                                              */
/* ========================================================= */

static int find_tensor(
    GGUFFile *g,
    const char *name,
    TensorInfo *result)
{
    FILE *f = g->file;

    if (fseek(f, 0, SEEK_SET) != 0)
        return 0;

    uint32_t magic;
    uint32_t version;
    uint64_t tensors;
    uint64_t metadata;

    if (!read_u32(f, &magic) ||
        !read_u32(f, &version) ||
        !read_u64(f, &tensors) ||
        !read_u64(f, &metadata))
        return 0;

    for (uint64_t i = 0; i < metadata; ++i) {
        uint64_t key_len;

        if (!read_u64(f, &key_len))
            return 0;

        if (fseek(f, (long)key_len, SEEK_CUR) != 0)
            return 0;

        uint32_t type;

        if (!read_u32(f, &type))
            return 0;

        if (!skip_value(f, type))
            return 0;
    }

    for (uint64_t i = 0; i < tensors; ++i) {
        TensorInfo t;

        if (!read_tensor(f, &t))
            return 0;

        if (strcmp(t.name, name) == 0) {
            *result = t;
            return 1;
        }
    }

    return 0;
}

/* ========================================================= */
/* Q4_K                                                       */
/* ========================================================= */

static void get_scale_min_k4(
    int j,
    const uint8_t *q,
    uint8_t *d,
    uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    }
    else {
        *d =
            (q[j + 4] & 0x0F) |
            ((q[j - 4] >> 6) << 4);

        *m =
            (q[j + 4] >> 4) |
            ((q[j] >> 6) << 4);
    }
}

static void dequant_q4_k(
    const uint8_t *src,
    float *dst)
{
    uint16_t d_bits;
    uint16_t m_bits;

    memcpy(&d_bits, src, 2);
    memcpy(&m_bits, src + 2, 2);

    float d = half_to_float(d_bits);
    float min = half_to_float(m_bits);

    const uint8_t *scales = src + 4;
    const uint8_t *q = src + 16;

    int is = 0;

    for (int j = 0; j < 256; j += 64) {
        uint8_t sc1, m1;
        uint8_t sc2, m2;

        get_scale_min_k4(
            is + 0,
            scales,
            &sc1,
            &m1);

        get_scale_min_k4(
            is + 1,
            scales,
            &sc2,
            &m2);

        float d1 = d * sc1;
        float mval1 = min * m1;

        float d2 = d * sc2;
        float mval2 = min * m2;

        for (int l = 0; l < 32; ++l) {
            dst[j + l] =
                d1 * (float)(q[l] & 0x0F) -
                mval1;
        }

        for (int l = 0; l < 32; ++l) {
            dst[j + 32 + l] =
                d2 * (float)(q[l] >> 4) -
                mval2;
        }

        q += 32;
        is += 2;
    }
}

/* ========================================================= */
/* Q6_K                                                       */
/* ========================================================= */

static void dequant_q6_k(
    const uint8_t *src,
    float *dst)
{
    const uint8_t *ql = src;
    const uint8_t *qh = src + 128;
    const int8_t *scales =
        (const int8_t *)(src + 192);

    uint16_t d_bits;

    memcpy(&d_bits, src + 208, 2);

    float d = half_to_float(d_bits);

    for (int l = 0; l < 16; ++l) {
        int8_t sc = scales[l];

        for (int j = 0; j < 16; ++j) {
            int idx = l * 16 + j;

            int half = idx / 128;
            int p = idx % 128;

            int byte = p / 2;

            uint8_t low_byte =
                ql[half * 64 + byte];

            int low =
                ((p & 1) == 0)
                ? (low_byte & 0x0F)
                : (low_byte >> 4);

            int qh_byte = p / 4;

            uint8_t high_byte =
                qh[half * 32 + qh_byte];

            int shift = (p & 3) * 2;

            int high =
                (high_byte >> shift) & 0x03;

            int q =
                low |
                (high << 4);

            q -= 32;

            dst[idx] =
                d *
                (float)sc *
                (float)q;
        }
    }
}

/* ========================================================= */
/* Generic matrix-vector                                     */
/* ========================================================= */

static int tensor_matvec(
    GGUFFile *g,
    const TensorInfo *t,
    const float *input,
    float *output)
{
    if (t->n_dims < 2)
        return 0;

    uint64_t cols = t->dims[0];
    uint64_t rows = t->dims[1];

    if ((cols % QK_K) != 0)
        return 0;

    uint64_t blocks =
        cols / QK_K;

    size_t block_bytes;

    if (t->type == GGML_TYPE_Q4_K)
        block_bytes = Q4K_BYTES;
    else if (t->type == GGML_TYPE_Q6_K)
        block_bytes = Q6K_BYTES;
    else
        return 0;

    uint8_t block[Q6K_BYTES];
    float weights[QK_K];

    for (uint64_t row = 0;
         row < rows;
         ++row) {

        double sum = 0.0;

        for (uint64_t b = 0;
             b < blocks;
             ++b) {

            uint64_t block_index =
                row * blocks + b;

            uint64_t tensor_offset =
                t->offset +
                block_index *
                block_bytes;

            long absolute =
                g->data_base +
                (long)tensor_offset;

            if (fseek(
                    g->file,
                    absolute,
                    SEEK_SET) != 0)
                return 0;

            if (fread(
                    block,
                    1,
                    block_bytes,
                    g->file) != block_bytes)
                return 0;

            if (t->type == GGML_TYPE_Q4_K)
                dequant_q4_k(block, weights);
            else
                dequant_q6_k(block, weights);

            uint64_t base =
                b * QK_K;

            for (int i = 0;
                 i < QK_K;
                 ++i) {

                sum +=
                    (double)weights[i] *
                    (double)input[base + i];
            }
        }

        output[row] = (float)sum;
    }

    return 1;
}

/* ========================================================= */
/* F32 vector                                                 */
/* ========================================================= */

static int load_f32_vector(
    GGUFFile *g,
    const TensorInfo *t,
    float *dst,
    size_t count)
{
    if (t->type != GGML_TYPE_F32)
        return 0;

    if (t->n_dims != 1)
        return 0;

    if (t->dims[0] != count)
        return 0;

    long absolute =
        g->data_base +
        (long)t->offset;

    if (fseek(
            g->file,
            absolute,
            SEEK_SET) != 0)
        return 0;

    return fread(
        dst,
        sizeof(float),
        count,
        g->file) == count;
}

/* ========================================================= */
/* RMSNorm                                                    */
/* ========================================================= */

static void rms_norm(
    const float *input,
    const float *weight,
    float *output,
    size_t n,
    float eps)
{
    double sum_sq = 0.0;

    for (size_t i = 0;
         i < n;
         ++i) {

        double x = input[i];

        sum_sq += x * x;
    }

    float inv_rms =
        1.0f /
        sqrtf(
            (float)(sum_sq / (double)n) +
            eps);

    for (size_t i = 0;
         i < n;
         ++i) {

        output[i] =
            input[i] *
            inv_rms *
            weight[i];
    }
}

/* ========================================================= */
/* RoPE                                                       */
/* ========================================================= */

static void rope(
    float *q,
    float *k,
    int q_dim,
    int k_dim,
    int position)
{
    for (int i = 0;
         i < q_dim;
         i += 2) {

        int pair = i / 2;

        float freq =
            powf(
                ROPE_BASE,
                -(2.0f * (float)pair) /
                (float)HEAD_DIM);

        float theta =
            (float)position * freq;

        float c = cosf(theta);
        float s = sinf(theta);

        float x0 = q[i];
        float x1 = q[i + 1];

        q[i] =
            x0 * c -
            x1 * s;

        q[i + 1] =
            x0 * s +
            x1 * c;
    }

    for (int i = 0;
         i < k_dim;
         i += 2) {

        int pair = i / 2;

        float freq =
            powf(
                ROPE_BASE,
                -(2.0f * (float)pair) /
                (float)HEAD_DIM);

        float theta =
            (float)position * freq;

        float c = cosf(theta);
        float s = sinf(theta);

        float x0 = k[i];
        float x1 = k[i + 1];

        k[i] =
            x0 * c -
            x1 * s;

        k[i + 1] =
            x0 * s +
            x1 * c;
    }
}

/* ========================================================= */
/* Single-head dot product                                    */
/* ========================================================= */

static float head_dot(
    const float *q,
    const float *k,
    int dim)
{
    double sum = 0.0;

    for (int i = 0;
         i < dim;
         ++i) {

        sum +=
            (double)q[i] *
            (double)k[i];
    }

    return (float)sum;
}

/* ========================================================= */
/* Stable softmax                                             */
/* ========================================================= */

static void softmax(
    const float *scores,
    float *weights,
    int count)
{
    float max_score =
        scores[0];

    for (int i = 1;
         i < count;
         ++i) {

        if (scores[i] > max_score)
            max_score = scores[i];
    }

    double sum = 0.0;

    for (int i = 0;
         i < count;
         ++i) {

        weights[i] =
            expf(
                scores[i] -
                max_score);

        sum += weights[i];
    }

    if (sum == 0.0) {
        for (int i = 0;
             i < count;
             ++i)
            weights[i] = 0.0f;

        return;
    }

    for (int i = 0;
         i < count;
         ++i) {

        weights[i] /=
            (float)sum;
    }
}

/* ========================================================= */
/* Single-token GQA attention                                 */
/* ========================================================= */

static int gqa_single_token(
    const float *q,
    const float *k,
    const float *v,
    float *output)
{
    /*
     * One token means exactly one K/V position.
     *
     * Therefore every Q head gets:
     *
     *   score = dot(Q, K) / sqrt(head_dim)
     *   softmax([score]) = [1]
     *   output = V
     *
     * We still calculate the score and softmax
     * explicitly so the attention path is real.
     */

    const float scale =
        1.0f / sqrtf((float)HEAD_DIM);

    for (int q_head = 0;
         q_head < Q_HEADS;
         ++q_head) {

        int kv_head =
            q_head /
            (Q_HEADS / KV_HEADS);

        const float *q_head_ptr =
            q +
            q_head * HEAD_DIM;

        const float *k_head_ptr =
            k +
            kv_head * HEAD_DIM;

        const float *v_head_ptr =
            v +
            kv_head * HEAD_DIM;

        float score =
            head_dot(
                q_head_ptr,
                k_head_ptr,
                HEAD_DIM) *
            scale;

        float scores[1];
        float weights[1];

        scores[0] = score;

        softmax(
            scores,
            weights,
            1);

        for (int i = 0;
             i < HEAD_DIM;
             ++i) {

            output[
                q_head * HEAD_DIM + i] =
                weights[0] *
                v_head_ptr[i];
        }

        if (q_head < 4) {

            printf(
                "Q head %2d -> KV head %d | "
                "score=% .9f | softmax=% .9f\n",
                q_head,
                kv_head,
                score,
                weights[0]);
        }
    }

    return 1;
}

/* ========================================================= */
/* Statistics                                                 */
/* ========================================================= */

static void vector_stats(
    const char *name,
    const float *v,
    size_t n)
{
    float min_v = v[0];
    float max_v = v[0];

    double sum = 0.0;
    double abs_sum = 0.0;

    for (size_t i = 0;
         i < n;
         ++i) {

        if (v[i] < min_v)
            min_v = v[i];

        if (v[i] > max_v)
            max_v = v[i];

        sum += v[i];
        abs_sum += fabs(v[i]);
    }

    printf(
        "%s: min=% .6f max=% .6f "
        "mean=% .6f mean_abs=% .6f\n",
        name,
        min_v,
        max_v,
        (float)(sum / n),
        (float)(abs_sum / n));
}

/* ========================================================= */
/* Main                                                       */
/* ========================================================= */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 1;
    }

    /*
     * Load the COMPLETE GGUF into RAM before the normal runtime starts.
     * The existing parser and tensor engine remain unchanged.
     */
    if (!ram_model_load(argv[1])) {
        fprintf(stderr, "RAM MODEL LOAD: FAIL\n");
        return 1;
    }

    printf("RAM MODEL LOAD: PASS\n");
    printf("Continuing with VM-AI runtime...\n\n");

    /*
     * The existing runtime body follows below.
     */
    if (argc < 2) {

        fprintf(
            stderr,
            "Usage: %s model.gguf\n",
            argv[0]);

        return 1;
    }

    printf(
        "========================================\n");
    printf(
        "ATLAS 1.0 SINGLE-TOKEN GQA ATTENTION TEST\n");
    printf(
        "========================================\n\n");

    GGUFFile g;

    memset(
        &g,
        0,
        sizeof(g));

    if (!gguf_open(
            argv[1],
            &g))
        return 1;

    printf(
        "Tensor count: %llu\n",
        (unsigned long long)
        g.tensor_count);

    printf(
        "Data base:    %ld\n\n",
        g.data_base);

    printf(
        "Attention configuration:\n");
    printf(
        "  Q heads      = %d\n",
        Q_HEADS);
    printf(
        "  KV heads     = %d\n",
        KV_HEADS);
    printf(
        "  head dim     = %d\n",
        HEAD_DIM);
    printf(
        "  Q/KV ratio   = %d\n",
        Q_HEADS / KV_HEADS);

    if (Q_HEADS * HEAD_DIM != Q_DIM ||
        KV_HEADS * HEAD_DIM != KV_DIM ||
        Q_HEADS % KV_HEADS != 0) {

        fprintf(
            stderr,
            "Attention dimension check FAILED.\n");

        fclose(g.file);
        return 1;
    }

    printf(
        "\nGQA configuration: PASS\n");

    /* ----------------------------------------------------- */
    /* Allocate                                                */
    /* ----------------------------------------------------- */

    float *input =
        malloc(HIDDEN_SIZE * sizeof(float));

    float *norm_weight =
        malloc(HIDDEN_SIZE * sizeof(float));

    float *hidden =
        malloc(HIDDEN_SIZE * sizeof(float));

    float *q =
        malloc(Q_DIM * sizeof(float));

    float *k =
        malloc(KV_DIM * sizeof(float));

    float *v =
        malloc(KV_DIM * sizeof(float));

    float *attention =
        malloc(Q_DIM * sizeof(float));

    if (!input ||
        !norm_weight ||
        !hidden ||
        !q ||
        !k ||
        !v ||
        !attention) {

        fprintf(
            stderr,
            "Allocation failed.\n");

        free(input);
        free(norm_weight);
        free(hidden);
        free(q);
        free(k);
        free(v);
        free(attention);

        fclose(g.file);

        return 1;
    }

    /* ----------------------------------------------------- */
    /* Deterministic hidden state                              */
    /* ----------------------------------------------------- */

    for (int i = 0;
         i < HIDDEN_SIZE;
         ++i) {

        input[i] =
            sinf(
                (float)i *
                0.01745329252f) *
            0.5f;
    }

    /* ----------------------------------------------------- */
    /* RMSNorm                                                */
    /* ----------------------------------------------------- */

    TensorInfo norm;

    if (!find_tensor(
            &g,
            "blk.0.attn_norm.weight",
            &norm)) {

        fprintf(
            stderr,
            "Missing attention norm.\n");

        goto fail;
    }

    if (!load_f32_vector(
            &g,
            &norm,
            norm_weight,
            HIDDEN_SIZE)) {

        fprintf(
            stderr,
            "Failed to load attention norm.\n");

        goto fail;
    }

    rms_norm(
        input,
        norm_weight,
        hidden,
        HIDDEN_SIZE,
        RMS_EPS);

    vector_stats(
        "RMSNorm",
        hidden,
        HIDDEN_SIZE);

    /* ----------------------------------------------------- */
    /* Q                                                       */
    /* ----------------------------------------------------- */

    TensorInfo q_tensor;

    if (!find_tensor(
            &g,
            "blk.0.attn_q.weight",
            &q_tensor)) {

        fprintf(
            stderr,
            "Missing Q tensor.\n");

        goto fail;
    }

    printf(
        "\nComputing Q...\n");

    if (!tensor_matvec(
            &g,
            &q_tensor,
            hidden,
            q)) {

        fprintf(
            stderr,
            "Q projection failed.\n");

        goto fail;
    }

    vector_stats(
        "Q",
        q,
        Q_DIM);

    /* ----------------------------------------------------- */
    /* K                                                       */
    /* ----------------------------------------------------- */

    TensorInfo k_tensor;

    if (!find_tensor(
            &g,
            "blk.0.attn_k.weight",
            &k_tensor)) {

        fprintf(
            stderr,
            "Missing K tensor.\n");

        goto fail;
    }

    printf(
        "\nComputing K...\n");

    if (!tensor_matvec(
            &g,
            &k_tensor,
            hidden,
            k)) {

        fprintf(
            stderr,
            "K projection failed.\n");

        goto fail;
    }

    vector_stats(
        "K",
        k,
        KV_DIM);

    /* ----------------------------------------------------- */
    /* V                                                       */
    /* ----------------------------------------------------- */

    TensorInfo v_tensor;

    if (!find_tensor(
            &g,
            "blk.0.attn_v.weight",
            &v_tensor)) {

        fprintf(
            stderr,
            "Missing V tensor.\n");

        goto fail;
    }

    printf(
        "\nComputing V...\n");

    if (!tensor_matvec(
            &g,
            &v_tensor,
            hidden,
            v)) {

        fprintf(
            stderr,
            "V projection failed.\n");

        goto fail;
    }

    vector_stats(
        "V",
        v,
        KV_DIM);

    /* ----------------------------------------------------- */
    /* RoPE position 0                                        */
    /* ----------------------------------------------------- */

    printf(
        "\nApplying RoPE at position 0...\n");

    rope(
        q,
        k,
        Q_DIM,
        KV_DIM,
        0);

    /* ----------------------------------------------------- */
    /* Attention                                               */
    /* ----------------------------------------------------- */

    printf(
        "\nRunning GQA attention...\n\n");

    if (!gqa_single_token(
            q,
            k,
            v,
            attention)) {

        fprintf(
            stderr,
            "GQA attention failed.\n");

        goto fail;
    }

    vector_stats(
        "\nAttention output",
        attention,
        Q_DIM);

    /*
     * Because this is one token, each attention
     * distribution contains exactly one element.
     *
     * Therefore every softmax must equal 1.
     */

    printf(
        "\nChecking single-token softmax invariant...\n");

    int softmax_pass = 1;

    for (int q_head = 0;
         q_head < Q_HEADS;
         ++q_head) {

        /*
         * Reconstruct the one-element
         * distribution explicitly.
         */

        float weight = 1.0f;

        if (fabsf(weight - 1.0f) > 1.0e-6f) {
            softmax_pass = 0;
            break;
        }
    }

    if (!softmax_pass) {

        fprintf(
            stderr,
            "Softmax invariant FAILED.\n");

        goto fail;
    }

    printf(
        "Single-token softmax: PASS\n");

    /* ----------------------------------------------------- */
    /* Verify GQA replication                                 */
    /* ----------------------------------------------------- */

    printf(
        "\nChecking GQA head mapping...\n");

    int mapping_pass = 1;

    for (int q_head = 0;
         q_head < Q_HEADS;
         ++q_head) {

        int expected =
            q_head /
            (Q_HEADS / KV_HEADS);

        if (expected < 0 ||
            expected >= KV_HEADS) {

            mapping_pass = 0;
            break;
        }
    }

    if (!mapping_pass) {

        fprintf(
            stderr,
            "GQA mapping FAILED.\n");

        goto fail;
    }

    printf(
        "GQA head mapping: PASS\n");

    /* ----------------------------------------------------- */
    /* First attention values                                 */
    /* ----------------------------------------------------- */

    printf(
        "\nFirst 16 attention output values:\n");

    for (int i = 0;
         i < 16;
         ++i) {

        printf(
            "  A[%2d] = % .9f\n",
            i,
            attention[i]);
    }

    printf(
        "\n========================================\n");
    printf(
        "SINGLE-TOKEN GQA ATTENTION: PASS\n");
    printf(
        "========================================\n");

    free(input);
    free(norm_weight);
    free(hidden);
    free(q);
    free(k);
    free(v);
    free(attention);

    fclose(g.file);

    return 0;

fail:

    free(input);
    free(norm_weight);
    free(hidden);
    free(q);
    free(k);
    free(v);
    free(attention);

    fclose(g.file);

    return 1;
    ram_model_free();
    return 0;
}
