#ifndef ATLAS_MODEL_REGISTRY_H
#define ATLAS_MODEL_REGISTRY_H

#include <stddef.h>
#include <stdint.h>

#define ATLAS_MAX_MODELS 256
#define ATLAS_MAX_CAPABILITIES 64
#define ATLAS_MAX_PATH 1024
#define ATLAS_MAX_NAME 256

typedef struct {
    char key[256];
    char value[1024];
    uint32_t type;
} AtlasMetadataEntry;

typedef struct {
    char name[256];
    uint32_t type;
    uint32_t dimensions;
    uint64_t shape[8];
    uint64_t offset;
    uint64_t size;
} AtlasTensorInfo;

typedef struct {
    char path[ATLAS_MAX_PATH];
    char name[ATLAS_MAX_NAME];
    char display_name[ATLAS_MAX_NAME];

    uint64_t file_size;
    uint32_t gguf_version;

    char architecture[128];

    uint32_t vocab_size;
    uint32_t context_length;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t block_count;
    uint32_t attention_heads;
    uint32_t attention_heads_kv;
    uint32_t head_dim;

    AtlasMetadataEntry *metadata;
    size_t metadata_count;
    size_t metadata_capacity;

    AtlasTensorInfo *tensors;
    size_t tensor_count;
    size_t tensor_capacity;

    uint64_t successful_uses;
    uint64_t failed_uses;
    uint64_t useful_uses;
    double usefulness;

    int loaded;
    int valid;
} AtlasModel;

typedef struct {
    AtlasModel *models;
    size_t count;
    size_t capacity;

    char root[ATLAS_MAX_PATH];

    uint64_t generation;
} AtlasModelRegistry;

void atlas_registry_init(
    AtlasModelRegistry *registry,
    const char *root
);

void atlas_registry_free(
    AtlasModelRegistry *registry
);

int atlas_registry_scan(
    AtlasModelRegistry *registry
);

void atlas_registry_print(
    const AtlasModelRegistry *registry
);

size_t atlas_registry_select(
    AtlasModelRegistry *registry,
    const char *problem,
    AtlasModel **results,
    size_t max_results
);

void atlas_model_record_result(
    AtlasModel *model,
    int useful
);

typedef int (*AtlasModelInvokeFn)(
    AtlasModel *model,
    const char *prompt,
    char *output,
    size_t output_capacity,
    void *userdata
);

int atlas_solve_with_models(
    AtlasModelRegistry *registry,
    const char *problem,
    AtlasModelInvokeFn invoke,
    void *userdata,
    char *combined_output,
    size_t combined_capacity
);

#endif
