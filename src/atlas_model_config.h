#ifndef ATLAS_MODEL_CONFIG_H
#define ATLAS_MODEL_CONFIG_H

#include <stdint.h>

#define ATLAS_ARCH_MAX 64

typedef struct {
    char architecture[ATLAS_ARCH_MAX];

    uint32_t vocab_size;
    uint32_t context_length;

    uint32_t embedding_length;
    uint32_t feed_forward_length;

    uint32_t block_count;

    uint32_t attention_heads;
    uint32_t attention_heads_kv;

    uint32_t head_dim;

    int has_tokenizer;
    int has_chat_template;

    int has_token_embeddings;
    int has_output_weights;

    int valid;
} AtlasModelConfig;

int atlas_model_inspect(
    const char *path,
    AtlasModelConfig *config
);

void atlas_model_config_print(
    const AtlasModelConfig *config
);

#endif
