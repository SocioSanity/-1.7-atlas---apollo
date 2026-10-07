#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} block_q6_K;

static float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x03FF;
            f = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign |
            ((uint32_t)(exp - 15 + 127) << 23) |
            (mant << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

static float q6k_dot(const block_q6_K *q, const float *x)
{
    const float d = fp16_to_fp32(q->d);
    float sum = 0.0f;

    for (int i = 0; i < 256; ++i) {
        uint8_t lo_byte = q->ql[i >> 1];

        int ql = (i & 1)
            ? ((lo_byte >> 4) & 0x0F)
            : (lo_byte & 0x0F);

        uint8_t hi_byte = q->qh[i >> 2];
        int qh = (hi_byte >> ((i & 3) * 2)) & 0x03;

        int q6 = (ql | (qh << 4)) - 32;

        int scale = q->scales[i >> 4];

        float weight = d * (float)scale * (float)q6;

        sum += weight * x[i];
    }

    return sum;
}

static void q6k_decode(const block_q6_K *q, float *dst)
{
    const float d = fp16_to_fp32(q->d);

    for (int i = 0; i < 256; ++i) {
        uint8_t lo_byte = q->ql[i >> 1];

        int ql = (i & 1)
            ? ((lo_byte >> 4) & 0x0F)
            : (lo_byte & 0x0F);

        uint8_t hi_byte = q->qh[i >> 2];
        int qh = (hi_byte >> ((i & 3) * 2)) & 0x03;

        int q6 = (ql | (qh << 4)) - 32;

        int scale = q->scales[i >> 4];

        dst[i] = d * (float)scale * (float)q6;
    }
}

int main(void)
{
    printf("========================================\n");
    printf("Q6_K DOT PRODUCT TEST\n");
    printf("========================================\n");

    FILE *f = fopen(
        "model/atlas-1.0.gguf",
        "rb"
    );

    if (!f) {
        perror("model");
        return 1;
    }

    const long offset = 5950528;

    if (fseek(f, offset, SEEK_SET) != 0) {
        fprintf(stderr, "seek failed\n");
        fclose(f);
        return 1;
    }

    block_q6_K block;

    if (fread(&block, 1, sizeof(block), f) != sizeof(block)) {
        fprintf(stderr, "could not read Q6_K block\n");
        fclose(f);
        return 1;
    }

    fclose(f);

    printf("Q6_K block size: %zu bytes\n", sizeof(block));

    float x[256];
    float decoded[256];

    for (int i = 0; i < 256; ++i)
        x[i] = sinf((float)i * 0.037f);

    q6k_decode(&block, decoded);

    float reference = 0.0f;

    for (int i = 0; i < 256; ++i)
        reference += decoded[i] * x[i];

    float direct = q6k_dot(&block, x);

    float error = fabsf(reference - direct);

    printf("\nReference dot: % .12f\n", reference);
    printf("Direct Q6_K:   % .12f\n", direct);
    printf("Absolute error: %.12e\n", error);

    printf("\nFirst 8 activations:\n");
    for (int i = 0; i < 8; ++i)
        printf("[%d] % .9f\n", i, x[i]);

    printf("\nFirst 8 weights:\n");
    for (int i = 0; i < 8; ++i)
        printf("[%d] % .9f\n", i, decoded[i]);

    printf("\n========================================\n");

    if (error < 1e-5f)
        printf("Q6_K DOT PRODUCT: PASS\n");
    else
        printf("Q6_K DOT PRODUCT: FAIL\n");

    printf("========================================\n");

    return error < 1e-5f ? 0 : 1;
}
