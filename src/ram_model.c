#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_MAGIC 0x46554747u

enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
};

typedef struct {
    char *name;

    uint32_t n_dims;
    uint64_t dims[4];

    uint32_t type;

    uint64_t offset;
    uint64_t size;

    unsigned char *data;
} RamTensor;

typedef struct {
    unsigned char *data;
    size_t size;

    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;

    RamTensor *tensors;

} RamModel;

static int read_u32_mem(
    const unsigned char **p,
    const unsigned char *end,
    uint32_t *v)
{
    if ((size_t)(end - *p) < 4)
        return 0;

    memcpy(v, *p, 4);
    *p += 4;

    return 1;
}

static int read_u64_mem(
    const unsigned char **p,
    const unsigned char *end,
    uint64_t *v)
{
    if ((size_t)(end - *p) < 8)
        return 0;

    memcpy(v, *p, 8);
    *p += 8;

    return 1;
}

static int read_string_mem(
    const unsigned char **p,
    const unsigned char *end,
    char **out)
{
    uint64_t len;

    if (!read_u64_mem(p, end, &len))
        return 0;

    if (len > SIZE_MAX - 1)
        return 0;

    if ((uint64_t)(end - *p) < len)
        return 0;

    char *s =
        (char *)malloc((size_t)len + 1);

    if (!s)
        return 0;

    memcpy(s, *p, (size_t)len);

    s[len] = '\0';

    *p += len;

    *out = s;

    return 1;
}

static int skip_value_mem(
    const unsigned char **p,
    const unsigned char *end,
    uint32_t type);

static int skip_array_mem(
    const unsigned char **p,
    const unsigned char *end)
{
    uint32_t element_type;
    uint64_t count;

    if (!read_u32_mem(
            p,
            end,
            &element_type))
        return 0;

    if (!read_u64_mem(
            p,
            end,
            &count))
        return 0;

    for (uint64_t i = 0;
         i < count;
         ++i) {

        if (!skip_value_mem(
                p,
                end,
                element_type))
            return 0;
    }

    return 1;
}

static int skip_value_mem(
    const unsigned char **p,
    const unsigned char *end,
    uint32_t type)
{
    size_t bytes = 0;

    switch (type) {

        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            bytes = 1;
            break;

        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            bytes = 2;
            break;

        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            bytes = 4;
            break;

        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            bytes = 8;
            break;

        case GGUF_TYPE_STRING: {
            uint64_t len;

            if (!read_u64_mem(
                    p,
                    end,
                    &len))
                return 0;

            if ((uint64_t)(end - *p) < len)
                return 0;

            *p += len;

            return 1;
        }

        case GGUF_TYPE_ARRAY:
            return skip_array_mem(
                p,
                end);

        default:
            return 0;
    }

    if ((size_t)(end - *p) < bytes)
        return 0;

    *p += bytes;

    return 1;
}

static int load_file(
    const char *path,
    unsigned char **out,
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

    unsigned char *data =
        (unsigned char *)malloc(size);

    if (!data) {
        fprintf(stderr,
            "Unable to allocate %zu bytes.\n",
            size);
        fclose(f);
        return 0;
    }

    size_t total = 0;

    while (total < size) {

        size_t got =
            fread(
                data + total,
                1,
                size - total,
                f);

        if (got == 0) {
            free(data);
            fclose(f);
            return 0;
        }

        total += got;
    }

    fclose(f);

    *out = data;
    *out_size = size;

    return 1;
}

static int tensor_element_size(
    uint32_t type)
{
    switch (type) {

        case 0:  return 1;
        case 1:  return 1;
        case 2:  return 2;
        case 3:  return 2;
        case 4:  return 4;
        case 5:  return 4;
        case 6:  return 4;
        case 10: return 8;
        case 11: return 8;
        case 12: return 8;

        default:
            return 0;
    }
}

static int tensor_size_from_type(
    uint32_t type,
    uint64_t elements,
    uint64_t *size)
{
    /*
     * Quantized GGML types require block calculations.
     *
     * Q4_K = 256 values / 144 bytes
     * Q6_K = 256 values / 210 bytes
     *
     * F32 = 4 bytes/value.
     */

    if (type == 0) {
        if (elements > UINT64_MAX / 4)
            return 0;

        *size = elements * 4;
        return 1;
    }

    /*
     * Q4_K
     */
    if (type == 12) {
        if (elements % 256 != 0)
            return 0;

        uint64_t blocks =
            elements / 256;

        if (blocks > UINT64_MAX / 144)
            return 0;

        *size = blocks * 144;

        return 1;
    }

    /*
     * Q6_K
     */
    if (type == 14) {
        if (elements % 256 != 0)
            return 0;

        uint64_t blocks =
            elements / 256;

        if (blocks > UINT64_MAX / 210)
            return 0;

        *size = blocks * 210;

        return 1;
    }

    int bytes =
        tensor_element_size(type);

    if (bytes <= 0)
        return 0;

    if (elements >
        UINT64_MAX / (uint64_t)bytes)
        return 0;

    *size =
        elements * (uint64_t)bytes;

    return 1;
}

static int parse_model(
    RamModel *model)
{
    const unsigned char *p =
        model->data;

    const unsigned char *end =
        model->data + model->size;

    uint32_t magic;

    if (!read_u32_mem(
            &p,
            end,
            &magic))
        return 0;

    if (magic != GGUF_MAGIC) {
        fprintf(stderr,
            "Invalid GGUF magic.\n");
        return 0;
    }

    if (!read_u32_mem(
            &p,
            end,
            &model->version))
        return 0;

    if (!read_u64_mem(
            &p,
            end,
            &model->tensor_count))
        return 0;

    if (!read_u64_mem(
            &p,
            end,
            &model->metadata_count))
        return 0;

    if (model->tensor_count >
        SIZE_MAX / sizeof(RamTensor))
        return 0;

    model->tensors =
        (RamTensor *)calloc(
            (size_t)model->tensor_count,
            sizeof(RamTensor));

    if (!model->tensors)
        return 0;

    /*
     * Metadata lives immediately after
     * the GGUF header.
     */

    for (uint64_t i = 0;
         i < model->metadata_count;
         ++i) {

        char *key = NULL;
        uint32_t type;

        if (!read_string_mem(
                &p,
                end,
                &key))
            return 0;

        if (!read_u32_mem(
                &p,
                end,
                &type)) {

            free(key);
            return 0;
        }

        free(key);

        if (!skip_value_mem(
                &p,
                end,
                type))
            return 0;
    }

    /*
     * Tensor directory.
     */

    for (uint64_t i = 0;
         i < model->tensor_count;
         ++i) {

        RamTensor *t =
            &model->tensors[i];

        if (!read_string_mem(
                &p,
                end,
                &t->name))
            return 0;

        if (!read_u32_mem(
                &p,
                end,
                &t->n_dims))
            return 0;

        if (t->n_dims > 4) {
            fprintf(stderr,
                "Unsupported tensor dimensions: %u\n",
                t->n_dims);
            return 0;
        }

        uint64_t elements = 1;

        for (uint32_t d = 0;
             d < t->n_dims;
             ++d) {

            if (!read_u64_mem(
                    &p,
                    end,
                    &t->dims[d]))
                return 0;

            if (t->dims[d] != 0 &&
                elements >
                UINT64_MAX / t->dims[d])
                return 0;

            elements *= t->dims[d];
        }

        if (!read_u32_mem(
                &p,
                end,
                &t->type))
            return 0;

        if (!read_u64_mem(
                &p,
                end,
                &t->offset))
            return 0;

        if (!tensor_size_from_type(
                t->type,
                elements,
                &t->size)) {

            /*
             * Some GGML types may not be
             * needed immediately. We still
             * preserve their directory entry.
             */
            t->size = 0;
        }
    }

    /*
     * GGUF tensor offsets are relative to
     * the beginning of the tensor-data
     * section. Standard GGUF alignment is
     * 32 bytes unless metadata says otherwise.
     *
     * Locate the tensor data by aligning
     * the current parser position.
     */

    uint64_t alignment = 32;

    uint64_t pos =
        (uint64_t)(p - model->data);

    uint64_t aligned =
        (pos + alignment - 1) &
        ~(alignment - 1);

    if (aligned > model->size)
        return 0;

    unsigned char *tensor_base =
        model->data + aligned;

    /*
     * Convert every GGUF tensor offset into
     * a direct RAM pointer.
     */

    for (uint64_t i = 0;
         i < model->tensor_count;
         ++i) {

        RamTensor *t =
            &model->tensors[i];

        if (t->offset >=
            (uint64_t)(model->size - aligned)) {

            fprintf(stderr,
                "Tensor %s points outside model.\n",
                t->name);

            return 0;
        }

        t->data =
            tensor_base + t->offset;

        if (t->size != 0) {

            uint64_t available =
                (uint64_t)(model->size -
                           aligned -
                           t->offset);

            if (t->size > available) {

                fprintf(stderr,
                    "Tensor %s exceeds model.\n",
                    t->name);

                return 0;
            }
        }
    }

    return 1;
}

static RamTensor *find_tensor(
    RamModel *model,
    const char *name)
{
    for (uint64_t i = 0;
         i < model->tensor_count;
         ++i) {

        if (strcmp(
                model->tensors[i].name,
                name) == 0)
            return &model->tensors[i];
    }

    return NULL;
}

static void free_model(
    RamModel *model)
{
    if (model->tensors) {

        for (uint64_t i = 0;
             i < model->tensor_count;
             ++i) {

            free(model->tensors[i].name);
        }

        free(model->tensors);
    }

    free(model->data);

    memset(
        model,
        0,
        sizeof(*model));
}

static void print_tensor(
    RamTensor *t)
{
    printf("%-30s ",
           t->name);

    printf("type=%-3u ",
           t->type);

    printf("dims=");

    for (uint32_t d = 0;
         d < t->n_dims;
         ++d) {

        if (d)
            printf("x");

        printf("%llu",
            (unsigned long long)t->dims[d]);
    }

    printf(" offset=%llu",
        (unsigned long long)t->offset);

    printf(" RAM=%p",
        (void *)t->data);

    printf("\n");
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
    printf("VM-AI RAM MODEL BACKEND\n");
    printf("========================================\n\n");

    RamModel model = {0};

    printf("Loading entire model into RAM...\n");

    if (!load_file(
            argv[1],
            &model.data,
            &model.size)) {

        fprintf(stderr,
            "MODEL LOAD: FAIL\n");

        return 1;
    }

    printf("RAM load: PASS\n");
    printf("RAM size: %.2f MB\n",
        (double)model.size /
        (1024.0 * 1024.0));

    printf("\nParsing GGUF directly from RAM...\n");

    if (!parse_model(&model)) {

        fprintf(stderr,
            "RAM GGUF PARSE: FAIL\n");

        free_model(&model);

        return 1;
    }

    printf("RAM GGUF PARSE: PASS\n");

    printf("Version: %u\n",
        model.version);

    printf("Tensor count: %llu\n",
        (unsigned long long)
        model.tensor_count);

    printf("Metadata count: %llu\n",
        (unsigned long long)
        model.metadata_count);

    printf("\nTesting RAM tensor lookup:\n\n");

    const char *tests[] = {
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
        "output.weight"
    };

    int failures = 0;

    size_t test_count =
        sizeof(tests) /
        sizeof(tests[0]);

    for (size_t i = 0;
         i < test_count;
         ++i) {

        RamTensor *t =
            find_tensor(
                &model,
                tests[i]);

        if (!t) {

            printf(
                "FAIL  %s\n",
                tests[i]);

            failures++;
            continue;
        }

        printf(
            "PASS  ");

        print_tensor(t);
    }

    printf("\n========================================\n");

    if (failures == 0) {

        printf(
            "RAM MODEL BACKEND: PASS\n");

        printf(
            "All tested tensors resolve directly into RAM.\n");

    } else {

        printf(
            "RAM MODEL BACKEND: FAIL\n");

        printf(
            "%d tensor lookups failed.\n",
            failures);
    }

    printf(
        "========================================\n");

    printf("\nRAM model remains resident until exit.\n");

    free_model(&model);

    return failures ? 1 : 0;
}

