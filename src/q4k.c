#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
} block_q4_K;

static uint16_t read_u16(const uint8_t *p)
{
    return (uint16_t)p[0] |
           ((uint16_t)p[1] << 8);
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

            f = sign |
                ((uint32_t)(e + 127) << 23) |
                (mant << 13);
        }
    }
    else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    }
    else {
        f = sign |
            ((exp - 15 + 127) << 23) |
            (mant << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

/*
 * Canonical K-quant scale/min extraction.
 *
 * Eight groups of 32 values.
 * The first four scale/min pairs are stored directly.
 * The upper four use the packed high bits in scales[8] and scales[9].
 */
static void get_scale_min_k4(
    const uint8_t *scales,
    int j,
    uint8_t *scale,
    uint8_t *min)
{
    if (j < 4) {
        *scale = scales[j] & 63;
        *min   = scales[j + 4] & 63;
    }
    else {
        *scale =
            (scales[j + 4] & 0x0F) |
            (((scales[8] >> (j - 4)) & 0x03) << 4);

        *min =
            (scales[j + 4] >> 4) |
            (((scales[9] >> (j - 4)) & 0x03) << 4);
    }
}

static void dequant_q4_k(
    const uint8_t *src,
    float *dst)
{
    const block_q4_K *q =
        (const block_q4_K *)src;

    const float d =
        fp16_to_fp32(q->d);

    const float dmin =
        fp16_to_fp32(q->dmin);

    int out = 0;

    /*
     * Q4_K stores 256 values as eight groups
     * of 32 values.
     *
     * Each group consumes 16 bytes of packed
     * 4-bit values. Low nibbles represent the
     * first 16 values, high nibbles the next 16.
     */
    for (int group = 0; group < 8; ++group) {

        uint8_t sc;
        uint8_t m;

        get_scale_min_k4(
            q->scales,
            group,
            &sc,
            &m);

        const float scale =
            d * (float)sc;

        const float minimum =
            dmin * (float)m;

        const uint8_t *qs =
            q->qs + group * 16;

        for (int j = 0; j < 16; ++j) {
            dst[out++] =
                scale * (float)(qs[j] & 0x0F)
                - minimum;
        }

        for (int j = 0; j < 16; ++j) {
            dst[out++] =
                scale * (float)(qs[j] >> 4)
                - minimum;
        }
    }
}

int main(void)
{
    const char *path =
        "model/atlas-1.0.gguf";

    FILE *f = fopen(path, "rb");

    if (!f) {
        perror(path);
        return 1;
    }

    const uint64_t tensor_data_start = 5950528ULL;
    const uint64_t token_offset      = 191439360ULL;

    const uint64_t position =
        tensor_data_start + token_offset;

    if (_fseeki64(
            f,
            (__int64)position,
            SEEK_SET) != 0) {

        fprintf(stderr, "Seek failed\n");
        fclose(f);
        return 1;
    }

    uint8_t raw[144];

    if (fread(raw, 1, sizeof(raw), f) != sizeof(raw)) {
        fprintf(stderr, "Read failed\n");
        fclose(f);
        return 1;
    }

    float values[QK_K];

    dequant_q4_k(raw, values);

    printf("========================================\n");
    printf("        CANONICAL Q4_K TEST\n");
    printf("========================================\n\n");

    printf("Tensor: token_embd.weight\n");
    printf("Tensor position: %llu\n",
           (unsigned long long)position);

    printf("Block size: %d elements\n", QK_K);
    printf("Block size: 144 bytes\n\n");

    printf("d    = %.10g\n",
           fp16_to_fp32(read_u16(raw)));

    printf("dmin = %.10g\n\n",
           fp16_to_fp32(read_u16(raw + 2)));

    printf("Raw scale bytes:\n");

    for (int i = 0; i < 12; ++i)
        printf("%02X ", raw[4 + i]);

    printf("\n\n");

    printf("First 64 dequantized values:\n\n");

    for (int i = 0; i < 64; ++i)
        printf("[%03d] % .10f\n", i, values[i]);

    float min = values[0];
    float max = values[0];
    double sum = 0.0;

    for (int i = 0; i < QK_K; ++i) {
        if (values[i] < min)
            min = values[i];

        if (values[i] > max)
            max = values[i];

        sum += values[i];
    }

    printf("\nBlock statistics:\n");
    printf("min  = %.10f\n", min);
    printf("max  = %.10f\n", max);
    printf("mean = %.10f\n", (float)(sum / QK_K));

    printf("\n========================================\n");
    printf("Q4_K block decoded: 256 / 256 values\n");
    printf("========================================\n");

    fclose(f);
    return 0;
}
