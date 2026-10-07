#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "atlas_model_bridge.h"
#include "atlas_conversation.h"

static int append_argument(char *dst, size_t capacity, size_t *used, const char *arg)
{
    size_t slashes = 0;
    if (*used + 1 >= capacity) return 0;
    dst[(*used)++] = '"';
    for (const char *p = arg; ; ++p) {
        if (*p == '\\') { ++slashes; continue; }
        if (*p == '"') {
            while (slashes) {
                if (*used + 2 >= capacity) return 0;
                dst[(*used)++] = '\\'; dst[(*used)++] = '\\'; --slashes;
            }
            if (*used + 2 >= capacity) return 0;
            dst[(*used)++] = '\\'; dst[(*used)++] = '"';
            continue;
        }
        while (slashes) {
            if (*used + 2 >= capacity) return 0;
            dst[(*used)++] = '\\'; dst[(*used)++] = '\\'; --slashes;
        }
        if (!*p) break;
        if (*used + 2 >= capacity) return 0;
        dst[(*used)++] = *p;
    }
    if (*used + 3 >= capacity) return 0;
    dst[(*used)++] = '"'; dst[(*used)++] = ' ';
    dst[*used] = 0;
    return 1;
}

int atlas_model_ask_with_budget(
    const char *exe,
    const char *model,
    const char *prompt,
    unsigned max_tokens,
    char *response,
    size_t response_size)
{
    if (!exe || !model || !prompt || !response || response_size < 2 ||
        max_tokens == 0 || max_tokens > 768)
        return 1;
    response[0] = 0;

    char token_text[16];
    snprintf(token_text, sizeof(token_text), "%u", max_tokens);
    size_t needed = strlen(exe) + strlen(model) + strlen(prompt) + sizeof(token_text) + 64;
    if (needed > (SIZE_MAX - 64) / 2) return 1;
    size_t command_capacity = needed * 2 + 64;
    char *command = malloc(command_capacity);
    if (!command) return 1;
    size_t used = 0;
    command[0] = 0;
    if (!append_argument(command, command_capacity, &used, exe) ||
        !append_argument(command, command_capacity, &used, model) ||
        !append_argument(command, command_capacity, &used, prompt) ||
        !append_argument(command, command_capacity, &used, token_text)) {
        free(command);
        return 1;
    }
    command[used - 1] = 0;

    atlas_conversation_init();
    atlas_conversation_message(ATLAS_SPEAKER_ATLAS, model, prompt);

    SECURITY_ATTRIBUTES sa = {0};
    HANDLE job = NULL;
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE read_pipe = NULL, write_pipe = NULL;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
        free(command);
        return 1;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);
    HANDLE null_stderr = CreateFileW(L"NUL", GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, NULL);
    if (null_stderr == INVALID_HANDLE_VALUE) {
        CloseHandle(read_pipe); CloseHandle(write_pipe); free(command); return 1;
    }

    STARTUPINFOA si = {0};
    PROCESS_INFORMATION pi = {0};
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_limits = {0};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = write_pipe;
    si.hStdError = null_stderr;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    job=CreateJobObjectW(NULL,NULL);
    if(!job){CloseHandle(read_pipe);CloseHandle(write_pipe);CloseHandle(null_stderr);free(command);return 1;}
    job_limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if(!SetInformationJobObject(job,JobObjectExtendedLimitInformation,
                                &job_limits,sizeof(job_limits))){
        CloseHandle(job);CloseHandle(read_pipe);CloseHandle(write_pipe);CloseHandle(null_stderr);free(command);return 1;
    }
    BOOL started = CreateProcessA(exe, command, NULL, NULL, TRUE,
        CREATE_NO_WINDOW|CREATE_SUSPENDED, NULL, NULL, &si, &pi);
    free(command);
    CloseHandle(write_pipe);
    CloseHandle(null_stderr);
    if (!started) {
        CloseHandle(job);CloseHandle(read_pipe);
        return 1;
    }
    if(!AssignProcessToJobObject(job,pi.hProcess)||ResumeThread(pi.hThread)==(DWORD)-1){
        TerminateJobObject(job,1);WaitForSingleObject(pi.hProcess,5000);
        CloseHandle(pi.hThread);CloseHandle(pi.hProcess);CloseHandle(job);CloseHandle(read_pipe);return 1;
    }

    size_t used_output = 0;
    char chunk[4096];
    WIN32_FILE_ATTRIBUTE_DATA model_attributes;
    uint64_t model_bytes=0;
    if(GetFileAttributesExA(model,GetFileExInfoStandard,&model_attributes))
        model_bytes=((uint64_t)model_attributes.nFileSizeHigh<<32)|model_attributes.nFileSizeLow;
    DWORD timeout_ms=model_bytes>UINT64_C(3000000000)?180000:60000;
    ULONGLONG deadline=GetTickCount64()+timeout_ms;
    int timed_out=0;
    for (;;) {
        DWORD available=0;
        if(!PeekNamedPipe(read_pipe,NULL,0,NULL,&available,NULL))break;
        if(available){
            DWORD got=0;
            DWORD request=available>(DWORD)sizeof(chunk)?(DWORD)sizeof(chunk):available;
            if(!ReadFile(read_pipe,chunk,request,&got,NULL)||!got)break;
            size_t keep=response_size-1-used_output;
            if(keep>got)keep=got;
            if(keep){memcpy(response+used_output,chunk,keep);used_output+=keep;}
            continue;
        }
        DWORD process_state=WaitForSingleObject(pi.hProcess,25);
        if(process_state==WAIT_OBJECT_0)break;
        if(GetTickCount64()>=deadline){timed_out=1;TerminateJobObject(job,124);break;}
    }
    if(timed_out)WaitForSingleObject(pi.hProcess,5000);
    response[used_output] = 0;
    CloseHandle(read_pipe);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(job);
    if (timed_out || exit_code != 0 || used_output == 0)
        return 1;

    atlas_conversation_message(ATLAS_SPEAKER_SPECIALIST, model, response);
    return 0;
}

int atlas_model_ask(const char *exe, const char *model, const char *prompt,
                     char *response, size_t response_size)
{
    return atlas_model_ask_with_budget(exe, model, prompt, 256,
                                        response, response_size);
}
