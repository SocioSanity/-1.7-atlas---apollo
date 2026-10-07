#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <string.h>

static HWND log_view;
static char *last_log;

static void refresh_log(void)
{
    HANDLE file = CreateFileA("atlas_conversation.log", GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return;
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size) || size.QuadPart < 0) { CloseHandle(file); return; }
    const DWORD limit = 1024 * 1024;
    DWORD bytes = size.QuadPart > limit ? limit : (DWORD)size.QuadPart;
    LARGE_INTEGER start;
    start.QuadPart = size.QuadPart - bytes;
    if (!SetFilePointerEx(file, start, NULL, FILE_BEGIN)) { CloseHandle(file); return; }
    char *text = malloc((size_t)bytes + 1);
    if (!text) { CloseHandle(file); return; }
    DWORD got = 0;
    BOOL ok = !bytes || ReadFile(file, text, bytes, &got, NULL);
    CloseHandle(file);
    if (!ok) { free(text); return; }
    text[got] = 0;
    if (size.QuadPart > limit) {
        char *line = strchr(text, '\n');
        if (line) memmove(text, line + 1, strlen(line + 1) + 1);
    }
    if (last_log && strcmp(last_log, text) == 0) { free(text); return; }
    free(last_log);
    last_log = text;
    int chars = MultiByteToWideChar(CP_UTF8, 0, text, -1, NULL, 0);
    if (chars <= 0) return;
    wchar_t *wide = malloc((size_t)chars * sizeof(wchar_t));
    if (!wide) return;
    if (MultiByteToWideChar(CP_UTF8, 0, text, -1, wide, chars)) {
        SetWindowTextW(log_view, wide);
        SendMessageW(log_view, EM_SETSEL, (WPARAM)-1, (LPARAM)-1);
        SendMessageW(log_view, EM_SCROLLCARET, 0, 0);
    }
    free(wide);
}

static LRESULT CALLBACK monitor_proc(HWND window, UINT message, WPARAM wp, LPARAM lp)
{
    switch (message) {
    case WM_CREATE:
        log_view = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"Waiting for Atlas's model dialogue...",
            WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL,
            12, 12, 760, 520, window, (HMENU)1, NULL, NULL);
        SendMessageW(log_view, EM_SETLIMITTEXT, 1024 * 1024, 0);
        SendMessageW(log_view, WM_SETFONT, (WPARAM)GetStockObject(DEFAULT_GUI_FONT), TRUE);
        SetTimer(window, 1, 500, NULL);
        refresh_log();
        return 0;
    case WM_SIZE:
        MoveWindow(log_view, 12, 12, LOWORD(lp) - 24, HIWORD(lp) - 24, TRUE);
        return 0;
    case WM_TIMER:
        refresh_log();
        return 0;
    case WM_DESTROY:
        KillTimer(window, 1);
        free(last_log);
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(window, message, wp, lp);
}

int WINAPI WinMain(HINSTANCE instance, HINSTANCE previous, LPSTR command, int show)
{
    (void)previous; (void)command;
    WNDCLASSW wc = {0};
    wc.lpfnWndProc = monitor_proc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"AtlasChainMonitor";
    if (!RegisterClassW(&wc)) return 1;
    HWND window = CreateWindowW(wc.lpszClassName, L"Atlas Chain Monitor — Atlas and specialists",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 820, 600,
        NULL, NULL, instance, NULL);
    if (!window) return 1;
    ShowWindow(window, show);
    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return (int)msg.wParam;
}
