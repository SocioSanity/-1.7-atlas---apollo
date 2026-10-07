#include "q6k.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("========================================\n");
    printf("Q6_K FULL TENSOR TEST\n");
    printf("========================================\n");

    FILE *f = fopen("model.gguf", "rb");
    if (!f) {
        perror("model.gguf");
        return 1;
    }

    if (fseek(f, 5950528, SEEK_SET) != 0) {
        fclose(f);
        return 1;
    }

    const size_t blocks = 1024;
    const size_t values = blocks * 256;
    const size_t bytes = blocks * 210;

    unsigned char *data = malloc(bytes);
    float *out = malloc(values * sizeof(float));

    if (!data || !out) {
        fprintf(stderr, "allocation failed\n");
        free(data);
        free(out);
        fclose(f);
        return 1;
    }

    if (fread(data, 1, bytes, f) != bytes) {
        fprintf(stderr, "could not read tensor data\n");
        free(data);
        free(out);
        fclose(f);
        return 1;
    }

    fclose(f);

    size_t decoded = q6k_decode(data, out, blocks);

    printf("Blocks requested:  %zu\n", blocks);
    printf("Values expected:   %zu\n", values);
    printf("Values decoded:    %zu\n", decoded);
    printf("Bytes consumed:    %zu\n", bytes);

    float min = out[0];
    float max = out[0];
    double sum = 0.0;

    for (size_t i = 0; i < decoded; ++i) {
        if (out[i] < min) min = out[i];
        if (out[i] > max) max = out[i];
        sum += out[i];
    }

    printf("\nFirst 16 values:\n");
    for (size_t i = 0; i < 16 && i < decoded; ++i)
        printf("[%04zu] % .9f\n", i, out[i]);

    printf("\nStatistics:\n");
    printf("min  = %.9f\n", min);
    printf("max  = %.9f\n", max);
    printf("mean = %.9f\n", (float)(sum / decoded));

    printf("\n========================================\n");

    if (decoded == values)
        printf("FULL Q6_K TENSOR DECODE: PASS\n");
    else
        printf("FULL Q6_K TENSOR DECODE: FAIL\n");

    printf("========================================\n");

    free(data);
    free(out);

    return decoded == values ? 0 : 1;
}
