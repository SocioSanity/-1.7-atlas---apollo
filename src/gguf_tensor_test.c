#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint8_t u8(FILE *f)
{
    int c = fgetc(f);
    if (c == EOF) exit(1);
    return (uint8_t)c;
}

static uint16_t u16(FILE *f)
{
    uint8_t b[2];
    if (fread(b,1,2,f) != 2) exit(1);
    return (uint16_t)b[0] | ((uint16_t)b[1] << 8);
}

static uint32_t u32(FILE *f)
{
    uint8_t b[4];
    if (fread(b,1,4,f) != 4) exit(1);
    return (uint32_t)b[0] |
           ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
}

static uint64_t u64(FILE *f)
{
    uint8_t b[8];
    if (fread(b,1,8,f) != 8) exit(1);
    return (uint64_t)b[0] |
           ((uint64_t)b[1] << 8) |
           ((uint64_t)b[2] << 16) |
           ((uint64_t)b[3] << 24) |
           ((uint64_t)b[4] << 32) |
           ((uint64_t)b[5] << 40) |
           ((uint64_t)b[6] << 48) |
           ((uint64_t)b[7] << 56);
}

static float f32(FILE *f)
{
    uint32_t x = u32(f);
    float v;
    memcpy(&v, &x, sizeof(v));
    return v;
}

static double f64(FILE *f)
{
    uint64_t x = u64(f);
    double v;
    memcpy(&v, &x, sizeof(v));
    return v;
}

static void skip_bytes(FILE *f, uint64_t n)
{
    while (n > 0) {
        long chunk = n > 1048576 ? 1048576 : (long)n;
        if (fseek(f, chunk, SEEK_CUR) != 0)
            exit(1);
        n -= (uint64_t)chunk;
    }
}

static char *read_string(FILE *f)
{
    uint64_t n = u64(f);

    if (n > 10000000ULL) {
        fprintf(stderr,
                "invalid string length: %llu\n",
                (unsigned long long)n);
        exit(1);
    }

    char *s = malloc((size_t)n + 1);
    if (!s) exit(1);

    if (fread(s,1,(size_t)n,f) != n)
        exit(1);

    s[n] = '\0';
    return s;
}

/*
 * GGUF metadata value types:
 *
 * 0  UINT8
 * 1  INT8
 * 2  UINT16
 * 3  INT16
 * 4  UINT32
 * 5  INT32
 * 6  FLOAT32
 * 7  BOOL
 * 8  STRING
 * 9  ARRAY
 * 10 UINT64
 * 11 INT64
 * 12 FLOAT64
 */
static void skip_value(FILE *f, uint32_t type);

static void skip_array(FILE *f)
{
    uint32_t element_type = u32(f);
    uint64_t count = u64(f);

    if (count > 100000000ULL) {
        fprintf(stderr,
                "invalid array count: %llu\n",
                (unsigned long long)count);
        exit(1);
    }

    for (uint64_t i = 0; i < count; ++i)
        skip_value(f, element_type);
}

static void skip_value(FILE *f, uint32_t type)
{
    switch (type) {
    case 0:  u8(f);  break;
    case 1:  u8(f);  break;
    case 2:  u16(f); break;
    case 3:  u16(f); break;
    case 4:  u32(f); break;
    case 5:  u32(f); break;
    case 6:  f32(f); break;
    case 7:  u8(f);  break;
    case 8: {
        char *s = read_string(f);
        free(s);
        break;
    }
    case 9:
        skip_array(f);
        break;
    case 10:
        u64(f);
        break;
    case 11:
        u64(f);
        break;
    case 12:
        f64(f);
        break;
    default:
        fprintf(stderr,
                "unknown GGUF metadata type: %u\n",
                type);
        exit(1);
    }
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "model/atlas-1.0.gguf";

    FILE *f = fopen(path, "rb");

    if (!f) {
        perror(path);
        return 1;
    }

    uint32_t magic = u32(f);
    uint32_t version = u32(f);
    uint64_t tensor_count = u64(f);
    uint64_t metadata_count = u64(f);

    printf("========================================\n");
    printf("GGUF TENSOR DISCOVERY TEST\n");
    printf("========================================\n");

    printf("Magic:        0x%08X\n", magic);
    printf("Version:      %u\n", version);
    printf("Tensor count: %llu\n",
           (unsigned long long)tensor_count);
    printf("Metadata:     %llu\n",
           (unsigned long long)metadata_count);

    if (magic != 0x46554747) {
        printf("INVALID GGUF MAGIC\n");
        fclose(f);
        return 1;
    }

    if (version < 2 || version > 3) {
        printf("UNSUPPORTED GGUF VERSION\n");
        fclose(f);
        return 1;
    }

    /*
     * Read metadata key/value pairs correctly.
     */
    for (uint64_t i = 0; i < metadata_count; ++i) {
        char *key = read_string(f);
        uint32_t type = u32(f);

        /*
         * Print the important model metadata.
         */
        if (strstr(key, "general.") ||
            strstr(key, "block_count") ||
            strstr(key, "embedding_length") ||
            strstr(key, "context_length")) {
            printf("metadata: %-45s type=%u\n", key, type);
        }

        skip_value(f, type);
        free(key);
    }

    printf("\nTensor directory:\n");

    int q6k_count = 0;
    int first_q6k = 1;

    uint64_t first_q6k_offset = 0;
    uint64_t first_q6k_elements = 0;
    char first_q6k_name[512] = {0};

    for (uint64_t i = 0; i < tensor_count; ++i) {
        char *name = read_string(f);

        uint32_t n_dims = u32(f);

        if (n_dims > 8) {
            fprintf(stderr,
                    "invalid dimension count: %u\n",
                    n_dims);
            free(name);
            fclose(f);
            return 1;
        }

        uint64_t dims[8] = {0};
        uint64_t elements = 1;

        for (uint32_t d = 0; d < n_dims; ++d) {
            dims[d] = u64(f);
            elements *= dims[d];
        }

        uint32_t type = u32(f);
        uint64_t offset = u64(f);

        /*
         * GGML_TYPE_Q6_K = 14.
         */
        if (type == 14) {
            q6k_count++;

            if (first_q6k) {
                first_q6k = 0;
                first_q6k_offset = offset;
                first_q6k_elements = elements;

                strncpy(
                    first_q6k_name,
                    name,
                    sizeof(first_q6k_name) - 1
                );
            }

            /*
             * Show the first 20 Q6_K tensors.
             */
            if (q6k_count <= 20) {
                printf("\n[%d] Q6_K\n", q6k_count);
                printf("Name:   %s\n", name);
                printf("Dims:   ");

                for (uint32_t d = 0; d < n_dims; ++d)
                    printf(
                        "%llu%s",
                        (unsigned long long)dims[d],
                        d + 1 == n_dims ? "\n" : " x "
                    );

                printf(
                    "Elements: %llu\n",
                    (unsigned long long)elements
                );

                printf(
                    "Offset:   %llu\n",
                    (unsigned long long)offset
                );
            }
        }

        free(name);
    }

    printf("\n========================================\n");
    printf("GGUF TENSOR DISCOVERY\n");
    printf("========================================\n");

    printf("Q6_K tensors found: %d\n", q6k_count);

    if (!first_q6k) {
        printf("\nFirst Q6_K tensor:\n");
        printf("Name:     %s\n", first_q6k_name);
        printf("Elements: %llu\n",
               (unsigned long long)first_q6k_elements);
        printf("Relative offset: %llu\n",
               (unsigned long long)first_q6k_offset);
    }

    printf("\n========================================\n");

    if (q6k_count > 0)
        printf("GGUF TENSOR DISCOVERY: PASS\n");
    else
        printf("GGUF TENSOR DISCOVERY: FAIL\n");

    printf("========================================\n");

    fclose(f);
    return q6k_count > 0 ? 0 : 1;
}



