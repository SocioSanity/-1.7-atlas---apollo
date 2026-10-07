#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv1a64(
    const unsigned char *data,
    size_t size)
{
    uint64_t hash = 1469598103934665603ULL;

    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }

    return hash;
}

static int load_entire_file(
    const char *path,
    unsigned char **out_data,
    size_t *out_size)
{
    FILE *f = fopen(path, "rb");

    if (!f) {
        perror("fopen");
        return 0;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }

    long file_size = ftell(f);

    if (file_size <= 0) {
        fclose(f);
        return 0;
    }

    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return 0;
    }

    size_t size = (size_t)file_size;

    printf("Model size: %.2f MB\n",
           (double)size / (1024.0 * 1024.0));

    printf("Allocating RAM...\n");

    unsigned char *data =
        (unsigned char *)malloc(size);

    if (!data) {
        fprintf(stderr,
            "RAM allocation failed for %zu bytes.\n",
            size);

        fclose(f);
        return 0;
    }

    printf("Reading model into RAM...\n");

    size_t total = 0;

    while (total < size) {

        size_t remaining = size - total;

        size_t chunk = remaining;

        if (chunk > 64 * 1024 * 1024)
            chunk = 64 * 1024 * 1024;

        size_t got =
            fread(
                data + total,
                1,
                chunk,
                f);

        if (got == 0) {

            fprintf(stderr,
                "\nModel read failed at %zu / %zu bytes.\n",
                total,
                size);

            free(data);
            fclose(f);
            return 0;
        }

        total += got;

        double percent =
            ((double)total * 100.0) /
            (double)size;

        printf(
            "\rLoaded: %6.2f%%",
            percent);

        fflush(stdout);
    }

    printf("\n");

    fclose(f);

    *out_data = data;
    *out_size = size;

    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {

        fprintf(stderr,
            "Usage: %s model.gguf\n",
            argv[0]);

        return 1;
    }

    printf("========================================\n");
    printf("VM-AI RAM MODEL LOADER\n");
    printf("========================================\n\n");

    unsigned char *model = NULL;
    size_t model_size = 0;

    if (!load_entire_file(
            argv[1],
            &model,
            &model_size)) {

        fprintf(stderr,
            "\nMODEL LOAD: FAIL\n");

        return 1;
    }

    printf("\nMODEL LOAD: PASS\n");

    printf("RAM address: %p\n",
           (void *)model);

    printf("RAM size: %.2f MB\n",
           (double)model_size /
           (1024.0 * 1024.0));

    printf(
        "First 4 bytes: %02X %02X %02X %02X\n",
        model[0],
        model[1],
        model[2],
        model[3]);

    uint32_t magic =
        (uint32_t)model[0] |
        ((uint32_t)model[1] << 8) |
        ((uint32_t)model[2] << 16) |
        ((uint32_t)model[3] << 24);

    printf("GGUF magic: 0x%08X\n",
           magic);

    uint64_t hash =
        fnv1a64(
            model,
            model_size);

    printf(
        "RAM model checksum: %016llX\n",
        (unsigned long long)hash);

    printf("\nModel is now resident in RAM.\n");
    printf("No inference is being executed.\n");
    printf("Press ENTER to release the model.\n");

    getchar();

    printf("\nReleasing RAM...\n");

    free(model);

    printf("RAM RELEASE: PASS\n");

    return 0;
}
