#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
} block_q6_K;

static uint16_t read_u16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

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

static void dequant_q6_k(const uint8_t *src, float *dst)
{
    const block_q6_K *q = (const block_q6_K *)src;
    const float d = fp16_to_fp32(q->d);

    for (int l = 0; l < 16; ++l) {
        int sc = q->scales[l];

        for (int j = 0; j < 16; ++j) {
            int idx = l * 16 + j;

            uint8_t lo = q->ql[idx >> 1];

            int ql = (idx & 1)
                ? ((lo >> 4) & 0x0F)
                : (lo & 0x0F);

            uint8_t hi = q->qh[idx >> 2];
            int qh = (hi >> ((idx & 3) * 2)) & 0x03;

            int q6 = ql | (qh << 4);
            q6 -= 32;

            dst[idx] = d * (float)sc * (float)q6;
        }
    }
}

int main(void)
{
    printf("========================================\n");
    printf("Q6_K MULTI-BLOCK DECODER TEST\n");
    printf("========================================\n");

    FILE *f = fopen("model/atlas-1.0.gguf", "rb");
    if (!f) {
        perror("model/atlas-1.0.gguf");
        return 1;
    }

    const long offset = 5950528;
    const size_t blocks = 1024;
    const size_t block_bytes = 210;
    const size_t values_per_block = 256;

    if (fseek(f, offset, SEEK_SET) != 0) {
        fprintf(stderr, "seek failed\n");
        fclose(f);
        return 1;
    }

    uint8_t *raw = malloc(blocks * block_bytes);
    float *values = malloc(blocks * values_per_block * sizeof(float));

    if (!raw || !values) {
        fprintf(stderr, "allocation failed\n");
        free(raw);
        free(values);
        fclose(f);
        return 1;
    }

    size_t bytes = blocks * block_bytes;

    if (fread(raw, 1, bytes, f) != bytes) {
        fprintf(stderr, "failed to read tensor data\n");
        free(raw);
        free(values);
        fclose(f);
        return 1;
    }

    fclose(f);

    for (size_t b = 0; b < blocks; ++b)
        dequant_q6_k(
            raw + b * block_bytes,
            values + b * values_per_block
        );

    size_t total = blocks * values_per_block;

    float min = values[0];
    float max = values[0];
    double sum = 0.0;

    for (size_t i = 0; i < total; ++i) {
        if (values[i] < min) min = values[i];
        if (values[i] > max) max = values[i];
        sum += values[i];
    }

    printf("Tensor offset:    %ld\n", offset);
    printf("Blocks decoded:   %zu / %zu\n", blocks, blocks);
    printf("Values decoded:   %zu / %zu\n", total, total);
    printf("Bytes processed:  %zu\n", bytes);

    printf("\nFirst 16 values:\n");
    for (size_t i = 0; i < 16; ++i)
        printf("[%04zu] % .9f\n", i, values[i]);

    printf("\nLast 16 values:\n");
    for (size_t i = total - 16; i < total; ++i)
        printf("[%04zu] % .9f\n", i, values[i]);

    printf("\nStatistics:\n");
    printf("min  = %.9f\n", min);
    printf("max  = %.9f\n", max);
    printf("mean = %.9f\n", (float)(sum / total));

    printf("\n========================================\n");
    printf("Q6_K MULTI-BLOCK DECODE: PASS\n");
    printf("========================================\n");

    free(raw);
    free(values);

    return 0;
}

