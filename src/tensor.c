#include "tensor.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} block_q6_K;

static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)p[0] |
           ((uint16_t)p[1] << 8);
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static float f16(uint16_t h)
{
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 31u;
    uint32_t mant = h & 1023u;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;

            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --e;
            }

            mant &= 0x3ffu;

            bits = sign |
                   ((uint32_t)(e + 127) << 23) |
                   (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign |
               ((exp - 15 + 127) << 23) |
               (mant << 13);
    }

    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void q4_scales(
    const uint8_t *s,
    int j,
    uint8_t *scale,
    uint8_t *min)
{
    if (j < 4) {
        *scale = s[j] & 63u;
        *min = s[j + 4] & 63u;
    } else {
        *scale =
            (s[j + 4] & 15u) |
            (((s[8] >> (j - 4)) & 3u) << 4);

        *min =
            (s[j + 4] >> 4) |
            (((s[9] >> (j - 4)) & 3u) << 4);
    }
}

static float q4k_dot(
    const uint8_t *src,
    const float *x)
{
    const block_q4_K *q =
        (const block_q4_K *)src;

    const float d = f16(rd16((const uint8_t *)&q->d));
    const float dmin = f16(rd16((const uint8_t *)&q->dmin));

    float sum = 0.0f;

    for (int g = 0; g < 8; ++g) {
        uint8_t sc;
        uint8_t mn;

        q4_scales(
            q->scales,
            g,
            &sc,
            &mn
        );

        const float scale = d * (float)sc;
        const float minimum = dmin * (float)mn;

        const uint8_t *qs =
            q->qs + g * 16;

        const float *xx =
            x + g * 32;

        for (int j = 0; j < 16; ++j) {
            float w =
                scale * (float)(qs[j] & 15u) -
                minimum;

            sum += w * xx[j];
        }

        for (int j = 0; j < 16; ++j) {
            float w =
                scale * (float)(qs[j] >> 4) -
                minimum;

            sum += w * xx[16 + j];
        }
    }

    return sum;
}

int vm_ai_q4k_matvec(
    const uint8_t *weights,
    uint64_t rows,
    uint64_t cols,
    const float *input,
    float *output)
{
    if (!weights || !input || !output)
        return 0;

    if (!rows || !cols)
        return 0;

    if (cols % 256)
        return 0;

    const uint64_t blocks =
        cols / 256;

    const uint64_t block_bytes = 144;

    for (uint64_t r = 0; r < rows; ++r) {
        const uint8_t *row =
            weights +
            r * blocks * block_bytes;

        float sum = 0.0f;

        for (uint64_t b = 0; b < blocks; ++b) {
            sum += q4k_dot(
                row + b * block_bytes,
                input + b * 256
            );
        }

        output[r] = sum;
    }

    return 1;
}

static float q6k_dot(
    const uint8_t *src,
    const float *x)
{
    const block_q6_K *q =
        (const block_q6_K *)src;

    const float d =
        f16(rd16((const uint8_t *)&q->d));

    float sum = 0.0f;

    /*
     * Q6_K contains 256 values.
     *
     * ql contains the low 4 bits.
     * qh contains the upper 2 bits.
     * scales contains sixteen signed scales.
     */

    for (int half = 0; half < 2; ++half) {
        for (int lane = 0; lane < 32; ++lane) {
            const uint8_t h = q->qh[half * 32 + lane];
            const uint8_t q0 = q->ql[half * 64 + lane];
            const uint8_t q1 = q->ql[half * 64 + 32 + lane];
            const int scale_base = half * 8 + lane / 16;
            const int qs[4] = {
                (q0 & 15) | ((h & 3) << 4),
                (q1 & 15) | (((h >> 2) & 3) << 4),
                (q0 >> 4) | (((h >> 4) & 3) << 4),
                (q1 >> 4) | (((h >> 6) & 3) << 4)
            };
            for (int chunk = 0; chunk < 4; ++chunk) {
                const int index = half * 128 + chunk * 32 + lane;
                const float scale = d * (float)q->scales[scale_base + chunk * 2];
                sum += scale * (float)(qs[chunk] - 32) * x[index];
            }
        }
    }

    return sum;
}

int vm_ai_q6k_matvec(
    const uint8_t *weights,
    uint64_t rows,
    uint64_t cols,
    const float *input,
    float *output)
{
    if (!weights || !input || !output)
        return 0;

    if (!rows || !cols)
        return 0;

    if (cols % 256)
        return 0;

    const uint64_t blocks =
        cols / 256;

    const uint64_t block_bytes = 210;

    for (uint64_t r = 0; r < rows; ++r) {
        const uint8_t *row =
            weights +
            r * blocks * block_bytes;

        float sum = 0.0f;

        for (uint64_t b = 0; b < blocks; ++b) {
            sum += q6k_dot(
                row + b * block_bytes,
                input + b * 256
            );
        }

        output[r] = sum;
    }

    return 1;
}

int vm_ai_f32_matvec(
    const uint8_t *weights,
    uint64_t rows,
    uint64_t cols,
    const float *input,
    float *output)
{
    if (!weights || !input || !output)
        return 0;

    for (uint64_t r = 0; r < rows; ++r) {
        const uint8_t *p =
            weights + r * cols * 4;

        float sum = 0.0f;

        for (uint64_t c = 0; c < cols; ++c) {
            uint32_t bits = rd32(p + c * 4);

            float w;
            memcpy(&w, &bits, sizeof(w));

            sum += w * input[c];
        }

        output[r] = sum;
    }

    return 1;
}
