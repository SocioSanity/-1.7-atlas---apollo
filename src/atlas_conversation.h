#ifndef ATLAS_CONVERSATION_H
#define ATLAS_CONVERSATION_H

#include <stddef.h>

typedef enum {
    ATLAS_SPEAKER_USER = 0,
    ATLAS_SPEAKER_ATLAS,
    ATLAS_SPEAKER_SPECIALIST,
    ATLAS_SPEAKER_TEACHER,
    ATLAS_SPEAKER_SYSTEM
} AtlasSpeaker;

typedef struct {
    AtlasSpeaker speaker;
    char model[128];
    char text[65536];
} AtlasMessage;

void atlas_conversation_init(void);

void atlas_conversation_message(
    AtlasSpeaker speaker,
    const char *model,
    const char *text
);

const char *atlas_speaker_name(AtlasSpeaker speaker);

#endif
