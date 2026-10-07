#include "atlas_model_config.h"

#include <stdio.h>
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

static int read_u32(FILE *f, uint32_t *v)
{
    return fread(v, sizeof(*v), 1, f) == 1;
}

static int read_u64(FILE *f, uint64_t *v)
{
    return fread(v, sizeof(*v), 1, f) == 1;
}

static int read_string(FILE *f, char **out)
{
    uint64_t len = 0;

    if (!read_u64(f, &len))
        return 0;

    if (len > SIZE_MAX - 1)
        return 0;

    char *s = malloc((size_t)len + 1);
    if (!s)
        return 0;

    if (len && fread(s, 1, (size_t)len, f) != len) {
        free(s);
        return 0;
    }

    s[len] = '\0';
    *out = s;
    return 1;
}

static int skip_value(FILE *f, uint32_t type)
{
    switch (type) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_BOOL:
        return fseek(f, 1, SEEK_CUR) == 0;

    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:
        return fseek(f, 2, SEEK_CUR) == 0;

    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32:
        return fseek(f, 4, SEEK_CUR) == 0;

    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64:
        return fseek(f, 8, SEEK_CUR) == 0;

    case GGUF_TYPE_STRING: {
        uint64_t len;
        if (!read_u64(f, &len))
            return 0;

        while (len) {
            long chunk = len > 0x7fffffffULL
                ? 0x7fffffffL
                : (long)len;

            if (fseek(f, chunk, SEEK_CUR) != 0)
                return 0;

            len -= (uint64_t)chunk;
        }

        return 1;
    }

    case GGUF_TYPE_ARRAY: {
        uint32_t element_type;
        uint64_t count;

        if (!read_u32(f, &element_type))
            return 0;

        if (!read_u64(f, &count))
            return 0;

        for (uint64_t i = 0; i < count; ++i)
            if (!skip_value(f, element_type))
                return 0;

        return 1;
    }

    default:
        return 0;
    }
}

static int key_is(const char *key, const char *wanted)
{
    return strcmp(key, wanted) == 0;
}

static uint32_t read_scalar_u32(FILE *f)
{
    uint32_t v = 0;
    if (!read_u32(f, &v))
        return 0;
    return v;
}

static int inspect_metadata(
    FILE *f,
    uint64_t metadata_count,
    AtlasModelConfig *c
)
{
    for (uint64_t i = 0; i < metadata_count; ++i) {
        char *key = NULL;
        uint32_t type = 0;

        if (!read_string(f, &key))
            return 0;

        if (!read_u32(f, &type)) {
            free(key);
            return 0;
        }

        if (key_is(key, "general.architecture") &&
            type == GGUF_TYPE_STRING) {

            char *value = NULL;

            if (!read_string(f, &value)) {
                free(key);
                return 0;
            }

            snprintf(
                c->architecture,
                sizeof(c->architecture),
                "%s",
                value
            );

            free(value);
        }
        else if (key_is(key, "general.context_length") &&
                 type == GGUF_TYPE_UINT32) {

            c->context_length = read_scalar_u32(f);
        }
        else if (key_is(key, "llama.context_length") &&
                 type == GGUF_TYPE_UINT32) {

            c->context_length = read_scalar_u32(f);
        }
        else if ((key_is(key, "qwen2.context_length") || key_is(key, "qwen3.context_length")) &&
                 type == GGUF_TYPE_UINT32) {

            c->context_length = read_scalar_u32(f);
        }
        else if (key_is(key, "llama.embedding_length") &&
                 type == GGUF_TYPE_UINT32) {

            c->embedding_length = read_scalar_u32(f);
        }
        else if ((key_is(key, "qwen2.embedding_length") || key_is(key, "qwen3.embedding_length")) &&
                 type == GGUF_TYPE_UINT32) {

            c->embedding_length = read_scalar_u32(f);
        }
        else if (key_is(key, "llama.feed_forward_length") &&
                 type == GGUF_TYPE_UINT32) {

            c->feed_forward_length = read_scalar_u32(f);
        }
        else if ((key_is(key, "qwen2.feed_forward_length") || key_is(key, "qwen3.feed_forward_length")) &&
                 type == GGUF_TYPE_UINT32) {

            c->feed_forward_length = read_scalar_u32(f);
        }
        else if (key_is(key, "llama.block_count") &&
                 type == GGUF_TYPE_UINT32) {

            c->block_count = read_scalar_u32(f);
        }
        else if ((key_is(key, "qwen2.block_count") || key_is(key, "qwen3.block_count")) &&
                 type == GGUF_TYPE_UINT32) {

            c->block_count = read_scalar_u32(f);
        }
        else if (key_is(key, "llama.attention.head_count") &&
                 type == GGUF_TYPE_UINT32) {

            c->attention_heads = read_scalar_u32(f);
        }
        else if ((key_is(key, "qwen2.attention.head_count") || key_is(key, "qwen3.attention.head_count")) &&
                 type == GGUF_TYPE_UINT32) {

            c->attention_heads = read_scalar_u32(f);
        }
        else if (key_is(key, "llama.attention.head_count_kv") &&
                 type == GGUF_TYPE_UINT32) {

            c->attention_heads_kv = read_scalar_u32(f);
        }
        else if ((key_is(key, "qwen2.attention.head_count_kv") || key_is(key, "qwen3.attention.head_count_kv")) &&
                 type == GGUF_TYPE_UINT32) {

            c->attention_heads_kv = read_scalar_u32(f);
        }
        else if (key_is(key, "tokenizer.ggml.tokens") &&
                 type == GGUF_TYPE_ARRAY) {

            c->has_tokenizer = 1;
            if (!skip_value(f, type)) {
                free(key);
                return 0;
            }
        }
        else if (key_is(key, "tokenizer.chat_template") &&
                 type == GGUF_TYPE_STRING) {

            c->has_chat_template = 1;
            if (!skip_value(f, type)) {
                free(key);
                return 0;
            }
        }
        else {
            if (!skip_value(f, type)) {
                free(key);
                return 0;
            }
        }

        free(key);
    }

    return 1;
}

static int inspect_tensors(
    FILE *f,
    uint64_t tensor_count,
    AtlasModelConfig *c
)
{
    for (uint64_t i = 0; i < tensor_count; ++i) {
        char *name = NULL;
        uint32_t dims = 0;
        uint64_t shape[4] = {0};
        uint32_t type = 0;
        uint64_t offset = 0;

        if (!read_string(f, &name))
            return 0;

        if (!read_u32(f, &dims) || dims > 4) {
            free(name);
            return 0;
        }

        for (uint32_t d = 0; d < dims; ++d)
            if (!read_u64(f, &shape[d])) {
                free(name);
                return 0;
            }

        if (!read_u32(f, &type) ||
            !read_u64(f, &offset)) {
            free(name);
            return 0;
        }

        (void)type;
        (void)offset;

        if (!strcmp(name, "token_embd.weight") &&
            dims == 2) {

            c->embedding_length = (uint32_t)shape[0];
            c->vocab_size = (uint32_t)shape[1];
            c->has_token_embeddings = 1;
        }

        if (!strcmp(name, "output.weight") &&
            dims == 2) {

            c->vocab_size = (uint32_t)shape[1];
            c->has_output_weights = 1;

            if (!c->embedding_length)
                c->embedding_length = (uint32_t)shape[0];
        }

        free(name);
    }

    return 1;
}

int atlas_model_inspect(
    const char *path,
    AtlasModelConfig *config
)
{
    if (!path || !config)
        return 1;

    memset(config, 0, sizeof(*config));

    FILE *f = fopen(path, "rb");
    if (!f)
        return 2;

    char magic[4];

    if (fread(magic, 1, 4, f) != 4 ||
        memcmp(magic, "GGUF", 4) != 0) {
        fclose(f);
        return 3;
    }

    uint32_t version = 0;

    if (!read_u32(f, &version) || version != 3) {
        fclose(f);
        return 4;
    }

    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;

    if (!read_u64(f, &tensor_count) ||
        !read_u64(f, &metadata_count)) {
        fclose(f);
        return 5;
    }

    if (!inspect_metadata(f, metadata_count, config) ||
        !inspect_tensors(f, tensor_count, config)) {
        fclose(f);
        return 6;
    }

    if (config->attention_heads &&
        config->embedding_length %
        config->attention_heads == 0) {

        config->head_dim =
            config->embedding_length /
            config->attention_heads;
    }

    config->valid =
        config->architecture[0] != '\0' &&
        config->embedding_length != 0 &&
        config->vocab_size != 0 &&
        config->block_count != 0 &&
        config->attention_heads != 0 &&
        config->has_token_embeddings &&
        config->has_output_weights;

    fclose(f);

    return config->valid ? 0 : 7;
}

void atlas_model_config_print(
    const AtlasModelConfig *c
)
{
    if (!c)
        return;

    printf("Atlas model configuration\n");
    printf("  architecture:       %s\n", c->architecture);
    printf("  vocabulary:         %u\n", c->vocab_size);
    printf("  context:            %u\n", c->context_length);
    printf("  embedding:          %u\n", c->embedding_length);
    printf("  feed-forward:       %u\n", c->feed_forward_length);
    printf("  layers:              %u\n", c->block_count);
    printf("  attention heads:     %u\n", c->attention_heads);
    printf("  KV heads:            %u\n", c->attention_heads_kv);
    printf("  head dimension:      %u\n", c->head_dim);
    printf("  embedded tokenizer:  %s\n",
           c->has_tokenizer ? "yes" : "no");
    printf("  chat template:       %s\n",
           c->has_chat_template ? "yes" : "no");
    printf("  token embeddings:    %s\n",
           c->has_token_embeddings ? "yes" : "no");
    printf("  output weights:      %s\n",
           c->has_output_weights ? "yes" : "no");
    printf("  valid:               %s\n",
           c->valid ? "yes" : "no");
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: atlas_model_inspect.exe <model.gguf>\n");
        return 2;
    }

    AtlasModelConfig config;

    int result = atlas_model_inspect(argv[1], &config);

    if (result != 0) {
        fprintf(
            stderr,
            "Model inspection failed: %d\n",
            result
        );
        if (result == 7) atlas_model_config_print(&config);
        return result;
    }

    atlas_model_config_print(&config);
    return 0;
}
