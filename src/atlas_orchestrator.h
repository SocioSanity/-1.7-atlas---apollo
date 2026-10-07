#ifndef ATLAS_ORCHESTRATOR_H
#define ATLAS_ORCHESTRATOR_H

#include <stddef.h>
#include <stdint.h>

#define ATLAS_MAX_SPECIALISTS 32
#define ATLAS_MAX_TASKS 16
#define ATLAS_MAX_RESPONSES 64
#define ATLAS_MAX_TEXT 16384

typedef struct {
    char name[64];
    char model[512];

    float reasoning;
    float coding;
    float research;
    float mathematics;
    float analysis;
    float speed;

    uint64_t successful_uses;
    uint64_t useful_uses;

    int enabled;
} AtlasSpecialist;

typedef struct {
    int id;
    char description[ATLAS_MAX_TEXT];

    int specialist_indices[8];
    int specialist_count;

    int duplicate_count;

    char responses[8][ATLAS_MAX_TEXT];
    int response_specialist_indices[8];
    int response_count;
} AtlasSubtask;

typedef struct {
    char original_problem[ATLAS_MAX_TEXT];

    AtlasSubtask tasks[ATLAS_MAX_TASKS];
    int task_count;

    char synthesis[ATLAS_MAX_TEXT];
    char verification[ATLAS_MAX_TEXT];

    int rounds;
    int disagreements;
    int verified;
} AtlasReasoningSession;

/*
 * Register a specialist Atlas can use.
 */
int atlas_register_specialist(
    const char *name,
    const char *model,
    float reasoning,
    float coding,
    float research,
    float mathematics,
    float analysis,
    float speed
);

/*
 * Remove all registered specialists.
 */
void atlas_clear_specialists(void);

/*
 * Run a complete multi-specialist reasoning session.
 *
 * exe          = model executable used by atlas_model_ask()
 * router_model = small language or coder model that selects specialists
 * coordinator_model = larger model that decomposes and synthesizes complex tasks
 * allow_reference = permit an additional coordinator review after rejection
 * problem      = user's original problem
 * session      = output reasoning session
 */
int atlas_reason(
    const char *exe,
    const char *router_model,
    const char *problem,
    AtlasReasoningSession *session
);

/* Small router chooses specialists; the larger coordinator handles complex reasoning. */
int atlas_reason_staged(
    const char *exe,
    const char *router_model,
    const char *coordinator_model,
    int allow_reference,
    const char *problem,
    AtlasReasoningSession *session
);

#endif
