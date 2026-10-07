#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static uint32_t read_u32(FILE *f)
{
    uint32_t v;
    if (fread(&v, sizeof(v), 1, f) != 1) {
        fprintf(stderr, "Failed reading uint32\n");
        exit(1);
    }
    return v;
}

static uint64_t read_u64(FILE *f)
{
    uint64_t v;
    if (fread(&v, sizeof(v), 1, f) != 1) {
        fprintf(stderr, "Failed reading uint64\n");
        exit(1);
    }
    return v;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "model/atlas-1.0.gguf";

    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return 1;
    }

    char magic[4];

    if (fread(magic, 1, 4, f) != 4) {
        fprintf(stderr, "Could not read GGUF magic\n");
        fclose(f);
        return 1;
    }

    if (magic[0] != 'G' ||
        magic[1] != 'G' ||
        magic[2] != 'U' ||
        magic[3] != 'F') {
        fprintf(stderr, "Invalid GGUF file\n");
        fclose(f);
        return 1;
    }

    uint32_t version = read_u32(f);
    uint64_t tensor_count = read_u64(f);
    uint64_t metadata_count = read_u64(f);

    printf("GGUF detected\n");
    printf("Version:        %u\n", version);
    printf("Tensors:        %llu\n",
           (unsigned long long)tensor_count);
    printf("Metadata items: %llu\n",
           (unsigned long long)metadata_count);

    fclose(f);
    return 0;
}




