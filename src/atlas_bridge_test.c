#include <stdio.h>
#include <string.h>
#include "atlas_model_bridge.h"

int main(void)
{
    char response[65536];

    printf("=== ATLAS MODEL BRIDGE TEST ===\n");
    printf("Atlas -> specialist\n");

    int rc = atlas_model_ask(
        "atlas_tiny_135m_infer.exe",
        "model/specialists/atlas_tiny_135m_router.gguf",
        "Atlas is testing communication. Respond in plain English and say hello to Atlas.",
        response,
        sizeof(response)
    );

    printf("\n=== SPECIALIST -> ATLAS ===\n");
    printf("%s\n", response);
    printf("\nBridge result: %s\n", rc == 0 ? "SUCCESS" : "FAILED");

    return rc;
}
