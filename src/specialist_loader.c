#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include "atlas_model_config.h"

static char *wide_to_utf8(const wchar_t *text)
{
    int n = WideCharToMultiByte(CP_UTF8, 0, text, -1, NULL, 0, NULL, NULL);
    if (n <= 0) return NULL;
    char *out = malloc((size_t)n);
    if (!out || !WideCharToMultiByte(CP_UTF8, 0, text, -1, out, n, NULL, NULL)) {
        free(out);
        return NULL;
    }
    return out;
}

static int append_quoted(wchar_t *dst, size_t capacity, size_t *used, const wchar_t *arg)
{
    size_t slashes = 0;
    if (*used + 1 >= capacity) return 0;
    dst[(*used)++] = L'"';
    for (const wchar_t *p = arg; ; ++p) {
        if (*p == L'\\') {
            ++slashes;
            continue;
        }
        if (*p == L'"') {
            while (slashes--) {
                if (*used + 2 >= capacity) return 0;
                dst[(*used)++] = L'\\'; dst[(*used)++] = L'\\';
            }
            if (*used + 2 >= capacity) return 0;
            dst[(*used)++] = L'\\'; dst[(*used)++] = L'"';
            slashes = 0;
            continue;
        }
        while (slashes--) {
            if (*used + 2 >= capacity) return 0;
            dst[(*used)++] = L'\\'; dst[(*used)++] = L'\\';
        }
        slashes = 0;
        if (!*p) break;
        if (*used + 2 >= capacity) return 0;
        dst[(*used)++] = *p;
    }
    if (*used + 2 >= capacity) return 0;
    dst[(*used)++] = L'"'; dst[(*used)++] = L' ';
    dst[*used] = 0;
    return 1;
}

int wmain(int argc, wchar_t **argv)
{
    if (argc < 2) {
        fwprintf(stderr, L"Usage: specialist_loader.exe <model.gguf> [prompt] [max_tokens]\n");
        return 2;
    }

    char *path = wide_to_utf8(argv[1]);
    if (!path) return 2;
    AtlasModelConfig c;
    int inspected = atlas_model_inspect(path, &c);
    free(path);
    if (inspected != 0) {
        fwprintf(stderr, L"[Lazy] Cannot inspect GGUF (error %d).\n", inspected);
        return 3;
    }

    if (strcmp(c.architecture, "qwen2") != 0 && strcmp(c.architecture, "qwen3") != 0) {
        fwprintf(stderr, L"[Lazy] Unsupported architecture: %S. The built-in C engine currently supports Qwen2 and Qwen3 tensor layouts.\n",
                 c.architecture);
        return 4;
    }
    const wchar_t *exe = L"atlas_dynamic_infer.exe";

    const wchar_t *prompt = argc >= 3 ? argv[2] : L"Hello, Atlas.";
    const wchar_t *tokens = argc >= 4 ? argv[3] : L"128";
    wchar_t command[16384];
    size_t used = 0;
    command[0] = 0;
    if (!append_quoted(command, _countof(command), &used, exe) ||
        !append_quoted(command, _countof(command), &used, argv[1]) ||
        !append_quoted(command, _countof(command), &used, prompt) ||
        !append_quoted(command, _countof(command), &used, tokens)) {
        fwprintf(stderr, L"[Lazy] Command line is too long.\n");
        return 5;
    }
    if (used) command[used - 1] = 0;

    fwprintf(stderr, L"[Lazy] Loading %ls with %ls (hidden %u, %u layers, %u attention heads).\n",
             argv[1], exe, c.embedding_length, c.block_count, c.attention_heads);
    STARTUPINFOW si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);
    if (!CreateProcessW(exe, command, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        fwprintf(stderr, L"[Lazy] Could not start %ls (Windows error %lu).\n", exe, GetLastError());
        return 6;
    }
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    fwprintf(stderr, L"[Lazy] Model unloaded; inference exit code %lu.\n", exit_code);
    return (int)exit_code;
}
