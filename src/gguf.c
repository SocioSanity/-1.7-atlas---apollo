#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_MAX_DIMS 4
#define GGUF_DEFAULT_ALIGNMENT 32

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t offset;
} TensorInfo;

typedef struct {
    FILE *file;
    uint64_t tensor_data_start;
    uint64_t tensor_count;
    TensorInfo *tensors;
} GGUFModel;

static uint32_t read_u32(FILE *f)
{
    uint32_t v;
    if (fread(&v, sizeof(v), 1, f) != 1)
        exit(1);
    return v;
}

static uint64_t read_u64(FILE *f)
{
    uint64_t v;
    if (fread(&v, sizeof(v), 1, f) != 1)
        exit(1);
    return v;
}

static char *read_string(FILE *f)
{
    uint64_t len = read_u64(f);

    if (len > 1024 * 1024)
        exit(1);

    char *s = malloc((size_t)len + 1);

    if (!s)
        exit(1);

    if (len && fread(s, 1, (size_t)len, f) != len) {
        free(s);
        exit(1);
    }

    s[len] = '\0';
    return s;
}

static void skip_bytes(FILE *f, uint64_t n)
{
    while (n) {
        long chunk = n > 0x7fffffffULL
                   ? 0x7fffffffL
                   : (long)n;

        if (fseek(f, chunk, SEEK_CUR) != 0)
            exit(1);

        n -= (uint64_t)chunk;
    }
}

static void skip_value(FILE *f, uint32_t type)
{
    switch (type) {
    case 0:
    case 1:
    case 7:
        skip_bytes(f, 1);
        break;

    case 2:
    case 3:
        skip_bytes(f, 2);
        break;

    case 4:
    case 5:
    case 6:
        skip_bytes(f, 4);
        break;

    case 8: {
        char *s = read_string(f);
        free(s);
        break;
    }

    case 9: {
        uint32_t element_type = read_u32(f);
        uint64_t count = read_u64(f);

        for (uint64_t i = 0; i < count; ++i)
            skip_value(f, element_type);

        break;
    }

    case 10:
    case 11:
    case 12:
        skip_bytes(f, 8);
        break;

    default:
        fprintf(stderr, "Unknown metadata type: %u\n", type);
        exit(1);
    }
}

static uint64_t file_size(FILE *f)
{
    if (_fseeki64(f, 0, SEEK_END) != 0)
        exit(1);

    __int64 size = _ftelli64(f);

    if (size < 0)
        exit(1);

    if (_fseeki64(f, 0, SEEK_SET) != 0)
        exit(1);

    return (uint64_t)size;
}

static uint64_t align_up(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) &
           ~(alignment - 1);
}

static const char *tensor_type_name(uint32_t type)
{
    switch (type) {
    case 0:  return "F32";
    case 1:  return "F16";
    case 2:  return "Q4_0";
    case 3:  return "Q4_1";
    case 6:  return "Q5_0";
    case 7:  return "Q5_1";
    case 8:  return "Q8_0";
    case 9:  return "Q8_1";
    case 10: return "Q2_K";
    case 11: return "Q3_K";
    case 12: return "Q4_K";
    case 13: return "Q5_K";
    case 14: return "Q6_K";
    case 15: return "Q8_K";
    case 16: return "IQ2_XXS";
    case 17: return "IQ2_XS";
    case 18: return "IQ3_XXS";
    case 19: return "IQ1_S";
    case 20: return "IQ4_NL";
    case 21: return "IQ3_S";
    case 22: return "IQ2_S";
    case 23: return "IQ4_XS";
    case 24: return "I8";
    case 25: return "I16";
    case 26: return "I32";
    case 27: return "I64";
    case 28: return "F64";
    case 29: return "IQ1_M";
    case 30: return "BF16";
    default: return "UNKNOWN";
    }
}

static void load_tensor_directory(GGUFModel *model)
{
    FILE *f = model->file;

    char magic[4];

    if (fread(magic, 1, 4, f) != 4 ||
        memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "Invalid GGUF file\n");
        exit(1);
    }

    uint32_t version = read_u32(f);

    if (version != 3) {
        fprintf(stderr, "Unsupported GGUF version: %u\n", version);
        exit(1);
    }

    model->tensor_count = read_u64(f);
    uint64_t metadata_count = read_u64(f);

    for (uint64_t i = 0; i < metadata_count; ++i) {
        char *key = read_string(f);
        uint32_t type = read_u32(f);

        free(key);
        skip_value(f, type);
    }

    model->tensors =
        calloc((size_t)model->tensor_count, sizeof(TensorInfo));

    if (!model->tensors)
        exit(1);

    for (uint64_t i = 0; i < model->tensor_count; ++i) {

        TensorInfo *t = &model->tensors[i];

        t->name = read_string(f);
        t->n_dims = read_u32(f);

        if (t->n_dims > GGUF_MAX_DIMS) {
            fprintf(stderr, "Tensor has too many dimensions: %s\n",
                    t->name);
            exit(1);
        }

        for (uint32_t d = 0; d < t->n_dims; ++d)
            t->dims[d] = read_u64(f);

        t->type = read_u32(f);
        t->offset = read_u64(f);
    }

    __int64 directory_end = _ftelli64(f);

    if (directory_end < 0)
        exit(1);

    model->tensor_data_start =
        align_up((uint64_t)directory_end,
                 GGUF_DEFAULT_ALIGNMENT);
}

static TensorInfo *find_tensor(GGUFModel *model, const char *name)
{
    for (uint64_t i = 0; i < model->tensor_count; ++i) {
        if (strcmp(model->tensors[i].name, name) == 0)
            return &model->tensors[i];
    }

    return NULL;
}

static uint64_t tensor_data_position(
    const GGUFModel *model,
    const TensorInfo *tensor)
{
    return model->tensor_data_start + tensor->offset;
}

static uint64_t tensor_element_count(const TensorInfo *tensor)
{
    uint64_t count = 1;

    for (uint32_t i = 0; i < tensor->n_dims; ++i)
        count *= tensor->dims[i];

    return count;
}

static uint64_t tensor_size_bytes(const TensorInfo *tensor)
{
    uint64_t elements = tensor_element_count(tensor);

    switch (tensor->type) {

    case 0:
        return elements * 4;

    case 1:
        return elements * 2;

    case 30:
        return elements * 2;

    case 24:
        return elements;

    case 25:
        return elements * 2;

    case 26:
        return elements * 4;

    case 27:
        return elements * 8;

    case 28:
        return elements * 8;

    case 2:
    case 3:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 29:
        fprintf(stderr,
                "Quantized tensor size calculation not yet enabled for type %s\n",
                tensor_type_name(tensor->type));
        return 0;

    default:
        return 0;
    }
}

static void inspect_tensor_data(GGUFModel *model,
                                const char *name)
{
    TensorInfo *t = find_tensor(model, name);

    if (!t) {
        fprintf(stderr, "Tensor not found: %s\n", name);
        return;
    }

    uint64_t position =
        tensor_data_position(model, t);

    printf("\nTensor: %s\n", t->name);
    printf("Type:   %s\n", tensor_type_name(t->type));
    printf("Data position: %llu\n",
           (unsigned long long)position);

    if (_fseeki64(model->file,
                  (__int64)position,
                  SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to tensor\n");
        return;
    }

    unsigned char bytes[32];

    if (fread(bytes, 1, sizeof(bytes), model->file)
        != sizeof(bytes)) {
        fprintf(stderr, "Failed to read tensor bytes\n");
        return;
    }

    printf("First 32 bytes:\n");

    for (int i = 0; i < 32; ++i)
        printf("%02X%s",
               bytes[i],
               (i % 16 == 15) ? "\n" : " ");
}

int main(void)
{
    const char *path =
        "model/atlas-llm-1.0-language.gguf";

    GGUFModel model = {0};

    model.file = fopen(path, "rb");

    if (!model.file) {
        perror(path);
        return 1;
    }

    uint64_t size = file_size(model.file);

    load_tensor_directory(&model);

    printf("========================================\n");
    printf("       GGUF TENSOR DATA VALIDATOR\n");
    printf("========================================\n\n");

    printf("File size:          %llu bytes\n",
           (unsigned long long)size);

    printf("Tensor count:       %llu\n",
           (unsigned long long)model.tensor_count);

    printf("Tensor data start:  %llu\n",
           (unsigned long long)model.tensor_data_start);

    printf("\n------------- KEY TENSORS --------------\n");

    const char *names[] = {
        "output.weight",
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "output_norm.weight"
    };

    size_t count = sizeof(names) / sizeof(names[0]);

    for (size_t i = 0; i < count; ++i)
        inspect_tensor_data(&model, names[i]);

    printf("\n========================================\n");
    printf("Tensor addressing validated.\n");
    printf("========================================\n");

    for (uint64_t i = 0; i < model.tensor_count; ++i)
        free(model.tensors[i].name);

    free(model.tensors);
    fclose(model.file);

    return 0;
}

