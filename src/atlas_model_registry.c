#include "atlas_model_registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define GGUF_MAGIC 0x46554747u

enum {
    GGUF_UINT8 = 0,
    GGUF_INT8 = 1,
    GGUF_UINT16 = 2,
    GGUF_INT16 = 3,
    GGUF_UINT32 = 4,
    GGUF_INT32 = 5,
    GGUF_FLOAT32 = 6,
    GGUF_BOOL = 7,
    GGUF_STRING = 8,
    GGUF_ARRAY = 9,
    GGUF_UINT64 = 10,
    GGUF_INT64 = 11,
    GGUF_FLOAT64 = 12
};

static uint32_t read_u32(FILE *f)
{
    uint8_t b[4];

    if (fread(b, 1, 4, f) != 4)
        return 0;

    return ((uint32_t)b[0]) |
           ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
}

static uint64_t read_u64(FILE *f)
{
    uint8_t b[8];

    if (fread(b, 1, 8, f) != 8)
        return 0;

    return ((uint64_t)b[0]) |
           ((uint64_t)b[1] << 8) |
           ((uint64_t)b[2] << 16) |
           ((uint64_t)b[3] << 24) |
           ((uint64_t)b[4] << 32) |
           ((uint64_t)b[5] << 40) |
           ((uint64_t)b[6] << 48) |
           ((uint64_t)b[7] << 56);
}

static int read_bytes(FILE *f, void *p, size_t n)
{
    return fread(p, 1, n, f) == n;
}

static void copy_text(char *dst, size_t size, const char *src)
{
    if (!dst || size == 0)
        return;

    if (!src) {
        dst[0] = 0;
        return;
    }

    snprintf(dst, size, "%s", src);
}

static int read_string(FILE *f, char *out, size_t capacity)
{
    uint64_t length;
    uint64_t remaining;
    size_t keep;

    length = read_u64(f);

    if (length > SIZE_MAX)
        return 0;

    keep = (size_t)length;

    if (keep >= capacity)
        keep = capacity - 1;

    if (keep && !read_bytes(f, out, keep))
        return 0;

    out[keep] = 0;

    remaining = length - keep;

    while (remaining) {
        unsigned char buffer[4096];
        size_t chunk = remaining > sizeof(buffer)
                     ? sizeof(buffer)
                     : (size_t)remaining;

        if (!read_bytes(f, buffer, chunk))
            return 0;

        remaining -= chunk;
    }

    return 1;
}

static int skip_value(FILE *f, uint32_t type)
{
    uint64_t count;
    uint32_t element_type;

    switch (type) {
    case GGUF_UINT8:
    case GGUF_INT8:
    case GGUF_BOOL:
        return fseek(f, 1, SEEK_CUR) == 0;

    case GGUF_UINT16:
    case GGUF_INT16:
        return fseek(f, 2, SEEK_CUR) == 0;

    case GGUF_UINT32:
    case GGUF_INT32:
    case GGUF_FLOAT32:
        return fseek(f, 4, SEEK_CUR) == 0;

    case GGUF_UINT64:
    case GGUF_INT64:
    case GGUF_FLOAT64:
        return fseek(f, 8, SEEK_CUR) == 0;

    case GGUF_STRING:
        count = read_u64(f);
        if (count > SIZE_MAX)
            return 0;
        return fseek(f, (long)count, SEEK_CUR) == 0;

    case GGUF_ARRAY:
        element_type = read_u32(f);
        count = read_u64(f);

        while (count--) {
            if (!skip_value(f, element_type))
                return 0;
        }

        return 1;

    default:
        return 0;
    }
}

static int metadata_reserve(AtlasModel *model, size_t wanted)
{
    AtlasMetadataEntry *p;
    size_t capacity;

    if (wanted <= model->metadata_capacity)
        return 1;

    capacity = model->metadata_capacity
             ? model->metadata_capacity * 2
             : 64;

    while (capacity < wanted)
        capacity *= 2;

    p = realloc(model->metadata,
                capacity * sizeof(*p));

    if (!p)
        return 0;

    model->metadata = p;
    model->metadata_capacity = capacity;

    return 1;
}

static int tensor_reserve(AtlasModel *model, size_t wanted)
{
    AtlasTensorInfo *p;
    size_t capacity;

    if (wanted <= model->tensor_capacity)
        return 1;

    capacity = model->tensor_capacity
             ? model->tensor_capacity * 2
             : 256;

    while (capacity < wanted)
        capacity *= 2;

    p = realloc(model->tensors,
                capacity * sizeof(*p));

    if (!p)
        return 0;

    model->tensors = p;
    model->tensor_capacity = capacity;

    return 1;
}

static int add_metadata(
    AtlasModel *model,
    const char *key,
    const char *value,
    uint32_t type)
{
    AtlasMetadataEntry *entry;

    if (!metadata_reserve(
            model,
            model->metadata_count + 1))
        return 0;

    entry = &model->metadata[model->metadata_count++];

    memset(entry, 0, sizeof(*entry));

    copy_text(entry->key,
              sizeof(entry->key),
              key);

    copy_text(entry->value,
              sizeof(entry->value),
              value);

    entry->type = type;

    return 1;
}

static int add_tensor(
    AtlasModel *model,
    const char *name,
    uint32_t type,
    uint32_t dimensions,
    const uint64_t *shape,
    uint64_t offset)
{
    AtlasTensorInfo *tensor;
    uint32_t i;

    if (!tensor_reserve(
            model,
            model->tensor_count + 1))
        return 0;

    tensor = &model->tensors[model->tensor_count++];

    memset(tensor, 0, sizeof(*tensor));

    copy_text(tensor->name,
              sizeof(tensor->name),
              name);

    tensor->type = type;
    tensor->dimensions = dimensions;
    tensor->offset = offset;

    for (i = 0; i < dimensions && i < 8; ++i)
        tensor->shape[i] = shape[i];

    return 1;
}

static void learn_metadata(
    AtlasModel *model,
    const char *key,
    const char *value)
{
    if (strcmp(key, "general.architecture") == 0) {
        copy_text(model->architecture,
                  sizeof(model->architecture),
                  value);
    }

    if (strstr(key, "context_length") ||
        strstr(key, "context_size"))
        model->context_length =
            (uint32_t)strtoul(value, NULL, 10);

    if (strstr(key, "embedding_length") ||
        strstr(key, "embedding_size"))
        model->embedding_length =
            (uint32_t)strtoul(value, NULL, 10);

    if (strstr(key, "feed_forward_length") ||
        strstr(key, "feed_forward_size"))
        model->feed_forward_length =
            (uint32_t)strtoul(value, NULL, 10);

    if (strstr(key, "block_count") ||
        strstr(key, "layer_count"))
        model->block_count =
            (uint32_t)strtoul(value, NULL, 10);

    if (strstr(key, "head_count_kv"))
        model->attention_heads_kv =
            (uint32_t)strtoul(value, NULL, 10);

    else if (strstr(key, "head_count"))
        model->attention_heads =
            (uint32_t)strtoul(value, NULL, 10);
}

static int inspect_metadata(
    FILE *f,
    AtlasModel *model,
    uint64_t count)
{
    uint64_t i;

    for (i = 0; i < count; ++i) {
        char key[256];
        char value[1024];
        uint32_t type;

        memset(key, 0, sizeof(key));
        memset(value, 0, sizeof(value));

        if (!read_string(f, key, sizeof(key)))
            return 0;

        type = read_u32(f);

        switch (type) {
        case GGUF_UINT8:
            snprintf(value, sizeof(value),
                     "%u", (unsigned)fgetc(f));
            break;

        case GGUF_INT8: {
            int c = fgetc(f);
            snprintf(value, sizeof(value),
                     "%d",
                     c == EOF ? 0 : (signed char)c);
            break;
        }

        case GGUF_BOOL: {
            int c = fgetc(f);
            snprintf(value, sizeof(value),
                     "%s", c ? "true" : "false");
            break;
        }

        case GGUF_UINT16:
        case GGUF_INT16: {
            uint8_t b[2];

            if (!read_bytes(f, b, 2))
                return 0;

            snprintf(value, sizeof(value),
                     "%u",
                     (unsigned)(b[0] |
                     ((uint16_t)b[1] << 8)));
            break;
        }

        case GGUF_UINT32:
        case GGUF_INT32:
        case GGUF_FLOAT32:
            snprintf(value, sizeof(value),
                     "%u", read_u32(f));
            break;

        case GGUF_UINT64:
        case GGUF_INT64:
        case GGUF_FLOAT64:
            snprintf(value, sizeof(value),
                     "%llu",
                     (unsigned long long)read_u64(f));
            break;

        case GGUF_STRING:
            if (!read_string(f, value, sizeof(value)))
                return 0;
            break;

        case GGUF_ARRAY: {
            uint32_t element_type = read_u32(f);
            uint64_t element_count = read_u64(f);

            snprintf(value, sizeof(value),
                     "array[%llu] type=%u",
                     (unsigned long long)element_count,
                     element_type);

            while (element_count--) {
                if (!skip_value(f, element_type))
                    return 0;
            }

            break;
        }

        default:
            return 0;
        }

        if (!add_metadata(
                model,
                key,
                value,
                type))
            return 0;

        learn_metadata(model, key, value);
    }

    return 1;
}

static int inspect_tensors(
    FILE *f,
    AtlasModel *model,
    uint64_t count)
{
    uint64_t i;

    for (i = 0; i < count; ++i) {
        char name[256];
        uint64_t shape[8];
        uint32_t dimensions;
        uint32_t type;
        uint64_t offset;
        uint32_t j;

        memset(name, 0, sizeof(name));
        memset(shape, 0, sizeof(shape));

        if (!read_string(f, name, sizeof(name)))
            return 0;

        dimensions = read_u32(f);

        if (dimensions > 8)
            return 0;

        for (j = 0; j < dimensions; ++j)
            shape[j] = read_u64(f);

        type = read_u32(f);
        offset = read_u64(f);

        if (!add_tensor(
                model,
                name,
                type,
                dimensions,
                shape,
                offset))
            return 0;

        if (dimensions >= 2) {
            if (strcmp(name, "token_embd.weight") == 0 ||
                strstr(name, "token_embd") != NULL) {
                model->embedding_length =
                    (uint32_t)shape[0];

                model->vocab_size =
                    (uint32_t)shape[1];
            }

            if (strcmp(name, "output.weight") == 0) {
                model->vocab_size =
                    (uint32_t)shape[1];

                if (!model->embedding_length)
                    model->embedding_length =
                        (uint32_t)shape[0];
            }
        }
    }

    if (model->attention_heads)
        model->head_dim =
            model->embedding_length /
            model->attention_heads;

    return 1;
}

static int inspect_model(
    const char *path,
    AtlasModel *model)
{
    FILE *f;
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;

    f = fopen(path, "rb");

    if (!f)
        return 0;

    magic = read_u32(f);

    if (magic != GGUF_MAGIC) {
        fclose(f);
        return 0;
    }

    version = read_u32(f);

    if (version < 1 || version > 3) {
        fclose(f);
        return 0;
    }

    tensor_count = read_u64(f);
    metadata_count = read_u64(f);

    model->gguf_version = version;

    if (!inspect_metadata(
            f,
            model,
            metadata_count)) {
        fclose(f);
        return 0;
    }

    if (!inspect_tensors(
            f,
            model,
            tensor_count)) {
        fclose(f);
        return 0;
    }

    fclose(f);

    model->valid = 1;

    return 1;
}

static void free_model(AtlasModel *model)
{
    free(model->metadata);
    free(model->tensors);

    memset(model, 0, sizeof(*model));
}

static int registry_reserve(
    AtlasModelRegistry *registry,
    size_t wanted)
{
    AtlasModel *p;
    size_t capacity;

    if (wanted <= registry->capacity)
        return 1;

    capacity = registry->capacity
             ? registry->capacity * 2
             : 32;

    while (capacity < wanted)
        capacity *= 2;

    p = realloc(
        registry->models,
        capacity * sizeof(*p));

    if (!p)
        return 0;

    registry->models = p;
    registry->capacity = capacity;

    return 1;
}

static int same_path(
    const char *a,
    const char *b)
{
#ifdef _WIN32
    return _stricmp(a, b) == 0;
#else
    return strcmp(a, b) == 0;
#endif
}

static int already_registered(
    AtlasModelRegistry *registry,
    const char *path)
{
    size_t i;

    for (i = 0; i < registry->count; ++i) {
        if (same_path(
                registry->models[i].path,
                path))
            return 1;
    }

    return 0;
}

#ifdef _WIN32

static void scan_directory(
    AtlasModelRegistry *registry,
    const char *directory)
{
    WIN32_FIND_DATAA data;
    HANDLE handle;
    char pattern[ATLAS_MAX_PATH];

    snprintf(pattern,
             sizeof(pattern),
             "%s\\*",
             directory);

    handle = FindFirstFileA(
        pattern,
        &data);

    if (handle == INVALID_HANDLE_VALUE)
        return;

    do {
        char path[ATLAS_MAX_PATH];

        if (!strcmp(data.cFileName, ".") ||
            !strcmp(data.cFileName, ".."))
            continue;

        snprintf(path,
                 sizeof(path),
                 "%s\\%s",
                 directory,
                 data.cFileName);

        if (data.dwFileAttributes &
            FILE_ATTRIBUTE_DIRECTORY) {
            scan_directory(
                registry,
                path);
            continue;
        }

        {
            const char *dot =
                strrchr(data.cFileName, '.');

            if (!dot ||
                _stricmp(dot, ".gguf") != 0)
                continue;
        }

        if (already_registered(
                registry,
                path))
            continue;

        if (!registry_reserve(
                registry,
                registry->count + 1))
            break;

        {
            AtlasModel *model =
                &registry->models[
                    registry->count];

            LARGE_INTEGER size;

            memset(model, 0, sizeof(*model));

            copy_text(model->path,
                      sizeof(model->path),
                      path);

            copy_text(model->name,
                      sizeof(model->name),
                      data.cFileName);

            size.LowPart =
                data.nFileSizeLow;
            size.HighPart =
                data.nFileSizeHigh;

            model->file_size =
                (uint64_t)size.QuadPart;

            if (inspect_model(path, model)) {
                registry->count++;

                printf(
                    "[ATLAS] discovered: %s\n",
                    path);
            } else {
                free_model(model);

                printf(
                    "[ATLAS] invalid GGUF: %s\n",
                    path);
            }
        }

    } while (FindNextFileA(handle, &data));

    FindClose(handle);
}

#endif

void atlas_registry_init(
    AtlasModelRegistry *registry,
    const char *root)
{
    memset(registry, 0, sizeof(*registry));

    copy_text(
        registry->root,
        sizeof(registry->root),
        root ? root : "model\\specialists");
}

void atlas_registry_free(
    AtlasModelRegistry *registry)
{
    size_t i;

    if (!registry)
        return;

    for (i = 0; i < registry->count; ++i)
        free_model(&registry->models[i]);

    free(registry->models);

    memset(registry, 0, sizeof(*registry));
}

int atlas_registry_scan(
    AtlasModelRegistry *registry)
{
    if (!registry)
        return 0;

    registry->generation++;

#ifdef _WIN32
    scan_directory(
        registry,
        registry->root);

    return 1;
#else
    return 0;
#endif
}

static int contains_ci(
    const char *text,
    const char *needle)
{
    size_t i;
    size_t n;

    if (!text || !needle)
        return 0;

    n = strlen(needle);

    for (i = 0; text[i]; ++i) {
        size_t j;

        for (j = 0;
             j < n &&
             text[i + j] &&
             tolower((unsigned char)text[i + j]) ==
             tolower((unsigned char)needle[j]);
             ++j) {
        }

        if (j == n)
            return 1;
    }

    return 0;
}

static double model_score(
    const AtlasModel *model,
    const char *problem)
{
    double score = 1.0;

    if (contains_ci(problem, "code") &&
        (contains_ci(model->name, "code") ||
         contains_ci(model->path, "code")))
        score += 5.0;

    if (contains_ci(problem, "math") &&
        (contains_ci(model->name, "math") ||
         contains_ci(model->path, "math")))
        score += 5.0;

    if ((contains_ci(problem, "reason") ||
         contains_ci(problem, "logic")) &&
        (contains_ci(model->name, "reason") ||
         contains_ci(model->path, "reason")))
        score += 5.0;

    if (model->context_length >= 8192)
        score += 1.0;

    score += model->usefulness * 4.0;

    if (model->architecture[0])
        score += 0.25;

    if (model->tensor_count > 100)
        score += 0.5;

    return score;
}

size_t atlas_registry_select(
    AtlasModelRegistry *registry,
    const char *problem,
    AtlasModel **results,
    size_t max_results)
{
    size_t i;
    size_t j;
    size_t count;

    if (!registry ||
        !problem ||
        !results ||
        !max_results)
        return 0;

    /*
     * Rank the currently discovered models.
     */
    for (i = 0; i < registry->count; ++i) {
        for (j = i + 1;
             j < registry->count;
             ++j) {
            if (model_score(
                    &registry->models[j],
                    problem) >
                model_score(
                    &registry->models[i],
                    problem)) {
                AtlasModel tmp =
                    registry->models[i];

                registry->models[i] =
                    registry->models[j];

                registry->models[j] =
                    tmp;
            }
        }
    }

    count =
        registry->count < max_results
        ? registry->count
        : max_results;

    for (i = 0; i < count; ++i)
        results[i] = &registry->models[i];

    return count;
}

void atlas_model_record_result(
    AtlasModel *model,
    int useful)
{
    if (!model)
        return;

    model->successful_uses++;

    if (useful)
        model->useful_uses++;
    else
        model->failed_uses++;

    model->usefulness =
        (double)model->useful_uses /
        (double)model->successful_uses;
}

void atlas_registry_print(
    const AtlasModelRegistry *registry)
{
    size_t i;

    printf("\n");
    printf("========================================\n");
    printf(" ATLAS MODEL REGISTRY\n");
    printf("========================================\n");
    printf("Root: %s\n", registry->root);
    printf("Generation: %llu\n",
           (unsigned long long)registry->generation);
    printf("Models discovered: %zu\n",
           registry->count);

    for (i = 0; i < registry->count; ++i) {
        const AtlasModel *m =
            &registry->models[i];

        printf("\n[%zu] %s\n",
               i + 1,
               m->name);

        printf("    path       : %s\n",
               m->path);

        printf("    architecture: %s\n",
               m->architecture[0]
               ? m->architecture
               : "unknown");

        printf("    GGUF       : %u\n",
               m->gguf_version);

        printf("    tensors    : %zu\n",
               m->tensor_count);

        printf("    metadata   : %zu\n",
               m->metadata_count);

        printf("    vocab      : %u\n",
               m->vocab_size);

        printf("    context    : %u\n",
               m->context_length);

        printf("    embedding  : %u\n",
               m->embedding_length);

        printf("    layers     : %u\n",
               m->block_count);

        printf("    heads      : %u\n",
               m->attention_heads);

        printf("    usefulness : %.3f\n",
               m->usefulness);
    }

    printf("\n");
}

int atlas_solve_with_models(
    AtlasModelRegistry *registry,
    const char *problem,
    AtlasModelInvokeFn invoke,
    void *userdata,
    char *combined_output,
    size_t combined_capacity)
{
    AtlasModel *models[16];
    size_t count;
    size_t i;
    size_t used = 0;

    if (!registry ||
        !problem ||
        !invoke ||
        !combined_output ||
        !combined_capacity)
        return 0;

    combined_output[0] = 0;

    count = atlas_registry_select(
        registry,
        problem,
        models,
        16);

    for (i = 0; i < count; ++i) {
        char output[4096];
        int result;

        memset(output, 0, sizeof(output));

        result = invoke(
            models[i],
            problem,
            output,
            sizeof(output),
            userdata);

        if (result <= 0) {
            atlas_model_record_result(
                models[i],
                0);
            continue;
        }

        if (output[0]) {
            int written =
                snprintf(
                    combined_output + used,
                    combined_capacity - used,
                    "\n\n===== SPECIALIST %zu =====\n"
                    "MODEL: %s\n"
                    "%s",
                    i + 1,
                    models[i]->name,
                    output);

            if (written > 0) {
                size_t amount =
                    (size_t)written;

                if (amount >=
                    combined_capacity - used)
                    used = combined_capacity - 1;
                else
                    used += amount;
            }

            atlas_model_record_result(
                models[i],
                1);
        }
    }

    return count ? 1 : 0;
}
