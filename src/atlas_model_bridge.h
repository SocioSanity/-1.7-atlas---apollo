#ifndef ATLAS_MODEL_BRIDGE_H
#define ATLAS_MODEL_BRIDGE_H

#include <stddef.h>

int atlas_model_ask(
    const char *exe,
    const char *model,
    const char *prompt,
    char *response,
    size_t response_size
);

int atlas_model_ask_with_budget(
    const char *exe,
    const char *model,
    const char *prompt,
    unsigned max_tokens,
    char *response,
    size_t response_size
);

#endif
