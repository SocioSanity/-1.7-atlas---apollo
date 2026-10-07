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
    uint32_t exp = (h >> 10) & 0x1F;
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
        f = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
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
        uint8_t lo = q->ql[i >> 1];
        int ql = (i & 1) ? (lo >> 4) : (lo & 0x0F);

        uint8_t hi = q->qh[i >> 2];
        int qh = (hi >> ((i & 3) * 2)) & 0x03;

        int q6 = (ql | (qh << 4)) - 32;
        int scale = q->scales[i >> 4];

        sum += d * (float)scale * (float)q6 * x[i];
    }

    return sum;
}

static void matrix_vector(
    const uint8_t *weights,
    const float *x,
    float *y,
    size_t rows,
    size_t cols)
{
    const size_t blocks_per_row = cols / 256;

    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;

        for (size_t b = 0; b < blocks_per_row; ++b) {
            const block_q6_K *q =
                (const block_q6_K *)(weights +
                (r * blocks_per_row + b) * 210);

            sum += q6k_dot(q, x + b * 256);
        }

        y[r] = sum;
    }
}

int main(void)
{
    printf("========================================\n");
    printf("Q6_K MATRIX-VECTOR TEST\n");
    printf("========================================\n");

    /*
     * Test a 4 x 1024 matrix.
     * 4 output neurons.
     * 1024 input values.
     * 4 Q6_K blocks per row.
     */
    const size_t rows = 4;
    const size_t cols = 1024;
    const size_t blocks_per_row = cols / 256;
    const size_t total_blocks = rows * blocks_per_row;
    const size_t bytes = total_blocks * 210;

    FILE *f = fopen(
        "model/atlas-1.0.gguf",
        "rb"
    );

    if (!f) {
        perror("model");
        return 1;
    }

    /*
     * Start at the known Q6_K tensor location.
     */
    if (fseek(f, 5950528, SEEK_SET) != 0) {
        fprintf(stderr, "seek failed\n");
        fclose(f);
        return 1;
    }

    uint8_t *weights = malloc(bytes);
    float *x = malloc(cols * sizeof(float));
    float *y = malloc(rows * sizeof(float));
    float *reference = malloc(rows * sizeof(float));

    if (!weights || !x || !y || !reference) {
        fprintf(stderr, "allocation failed\n");
        free(weights);
        free(x);
        free(y);
        free(reference);
        fclose(f);
        return 1;
    }

    if (fread(weights, 1, bytes, f) != bytes) {
        fprintf(stderr, "could not read matrix data\n");
        free(weights);
        free(x);
        free(y);
        free(reference);
        fclose(f);
        return 1;
    }

    fclose(f);

    for (size_t i = 0; i < cols; ++i)
        x[i] = sinf((float)i * 0.013f);

    /*
     * Actual Q6_K matrix-vector operation.
     */
    matrix_vector(weights, x, y, rows, cols);

    /*
     * Independent reference calculation.
     * This deliberately decodes each weight first.
     */
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;

        for (size_t b = 0; b < blocks_per_row; ++b) {
            const block_q6_K *q =
                (const block_q6_K *)(weights +
                (r * blocks_per_row + b) * 210);

            const float d = fp16_to_fp32(q->d);

            for (int i = 0; i < 256; ++i) {
                uint8_t lo = q->ql[i >> 1];
                int ql = (i & 1) ? (lo >> 4) : (lo & 0x0F);

                uint8_t hi = q->qh[i >> 2];
                int qh = (hi >> ((i & 3) * 2)) & 0x03;

                int q6 = (ql | (qh << 4)) - 32;
                int scale = q->scales[i >> 4];

                float w = d * (float)scale * (float)q6;

                sum += w * x[b * 256 + i];
            }
        }

        reference[r] = sum;
    }

    float max_error = 0.0f;

    printf("Rows:             %zu\n", rows);
    printf("Columns:          %zu\n", cols);
    printf("Blocks per row:   %zu\n", blocks_per_row);
    printf("Total Q6_K blocks:%zu\n", total_blocks);
    printf("Bytes processed:  %zu\n", bytes);

    printf("\nResults:\n");

    for (size_t r = 0; r < rows; ++r) {
        float error = fabsf(y[r] - reference[r]);

        if (error > max_error)
            max_error = error;

        printf(
            "row[%zu]  direct=% .12f  reference=% .12f  error=%.3e\n",
            r,
            y[r],
            reference[r],
            error
        );
    }

    printf("\n========================================\n");

    if (max_error < 1e-5f)
        printf("Q6_K MATRIX-VECTOR: PASS\n");
    else
        printf("Q6_K MATRIX-VECTOR: FAIL\n");

    printf("Maximum error: %.12e\n", max_error);
    printf("========================================\n");

    free(weights);
    free(x);
    free(y);
    free(reference);

    return max_error < 1e-5f ? 0 : 1;
}
