#include "atlas_conversation.h"
#include <stdio.h>
#include <string.h>

static FILE *g_conversation = NULL;

void atlas_conversation_init(void)
{
    if (!g_conversation)
        g_conversation = fopen("atlas_conversation.log", "ab");
}

const char *atlas_speaker_name(AtlasSpeaker speaker)
{
    switch (speaker) {
        case ATLAS_SPEAKER_USER:       return "USER";
        case ATLAS_SPEAKER_ATLAS:     return "ATLAS";
        case ATLAS_SPEAKER_SPECIALIST: return "SPECIALIST";
        case ATLAS_SPEAKER_TEACHER:    return "TEACHER";
        case ATLAS_SPEAKER_SYSTEM:     return "SYSTEM";
        default:                         return "UNKNOWN";
    }
}

void atlas_conversation_message(
    AtlasSpeaker speaker,
    const char *model,
    const char *text)
{
    if (!text)
        return;

    if (!model)
        model = "";

    printf("\n[%s%s%s]\n%s\n",
           atlas_speaker_name(speaker),
           model[0] ? " / " : "",
           model,
           text);
    fflush(stdout);

    if (g_conversation) {
        fprintf(g_conversation,
                "\n[%s%s%s]\n%s\n",
                atlas_speaker_name(speaker),
                model[0] ? " / " : "",
                model,
                text);
        fflush(g_conversation);
    }
}
