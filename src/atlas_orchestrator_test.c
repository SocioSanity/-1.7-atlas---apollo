#include "atlas_orchestrator.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    AtlasReasoningSession *session =
        (AtlasReasoningSession *)calloc(1, sizeof(AtlasReasoningSession));

    if (!session) {
        fprintf(stderr, "Failed to allocate Atlas reasoning session.\n");
        return 1;
    }

    atlas_clear_specialists();

    atlas_register_specialist(
        "Qwen 1.5B Coder",
        "model\\specialists\\qwen\\qwen2.5-coder-1.5b-q4_k_m.gguf",
        0.75f, 1.00f, 0.40f, 0.35f, 0.60f, 0.90f);
    atlas_register_specialist(
        "Qwen 1.5B Math",
        "model\\specialists\\qwen\\qwen2.5-math-1.5b-q4_k_m.gguf",
        0.75f, 0.35f, 0.40f, 1.00f, 0.60f, 1.00f);
    atlas_register_specialist(
        "Qwen 1.5B Instruct",
        "model\\specialists\\qwen\\qwen2.5-1.5b-instruct-q4_k_m.gguf",
        0.75f, 0.35f, 0.70f, 0.35f, 0.60f, 0.90f);

    printf("=== ATLAS ORCHESTRATOR TEST ===\n\n");
    fflush(stdout);

    if (atlas_reason_staged(
            ".\\specialist_loader.exe",
            "model\\atlas-llm-1.0-compact.gguf",
            "model\\atlas-llm-1.0.gguf",
            1,
            "Write a complete C function parse_size(const char *text, size_t *out). Reject null pointers, empty input, signs, nondigits, and overflow using SIZE_MAX. Leave *out unchanged on failure. Return 1 on success and 0 on failure. Include required headers and no main.",
            session) != 0) {

        fprintf(stderr, "\nAtlas reasoning session FAILED.\n");
        free(session);
        return 1;
    }

    printf("\n=== ATLAS SYNTHESIS ===\n%s\n", session->synthesis);

    printf("\n=== ATLAS VERIFICATION ===\n%s\n", session->verification);

    printf(
        "\n=== SESSION RESULT ===\n"
        "Tasks: %d\n"
        "Rounds: %d\n"
        "Verified: %s\n",
        session->task_count,
        session->rounds,
        session->verified ? "YES" : "NO"
    );

    free(session);
    return 0;
}



