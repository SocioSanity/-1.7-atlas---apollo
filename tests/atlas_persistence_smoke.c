#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <string.h>

#define ID_PROMPT 101
#define ID_SEND 102
#define ID_LOG 103

static HWND wait_for_window(DWORD timeout_ms)
{
    DWORD start=GetTickCount();
    HWND window=NULL;
    while (GetTickCount()-start<timeout_ms) {
        window=FindWindowW(L"AtlasChatWindow",NULL);
        if (window) return window;
        Sleep(100);
    }
    return NULL;
}

static int start_chat(const wchar_t *exe, PROCESS_INFORMATION *pi)
{
    wchar_t command[2*MAX_PATH];
    swprintf(command,2*MAX_PATH,L"\"%ls\"",exe);
    STARTUPINFOW si={0}; si.cb=sizeof(si);
    if (!CreateProcessW(exe,command,NULL,NULL,FALSE,0,NULL,NULL,&si,pi)) return 0;
    CloseHandle(pi->hThread);
    return 1;
}

static int wait_response(HWND window, DWORD timeout_ms)
{
    HWND send=GetDlgItem(window,ID_SEND);
    DWORD start=GetTickCount(); int observed_busy=0;
    while (GetTickCount()-start<timeout_ms) {
        if (!IsWindowEnabled(send)) observed_busy=1;
        else if (observed_busy) return 1;
        Sleep(100);
    }
    return 0;
}

static int log_contains(HWND window, const wchar_t *needle)
{
    HWND log=GetDlgItem(window,ID_LOG);
    int n=GetWindowTextLengthW(log);
    if (n<=0 || n>200000) return 0;
    wchar_t *text=calloc((size_t)n+1,sizeof(wchar_t));
    if (!text) return 0;
    GetWindowTextW(log,text,n+1);
    int found=wcsstr(text,needle)!=NULL;
    free(text); return found;
}

int wmain(int argc, wchar_t **argv)
{
    if (argc!=2) return 2;
    PROCESS_INFORMATION first={0};
    if (!start_chat(argv[1],&first)) return 3;
    HWND window=wait_for_window(15000);
    if (!window) return 4;
    SetWindowTextW(GetDlgItem(window,ID_PROMPT),L"Reply only with OK. Remember code 73142.");
    SendMessageW(window,WM_COMMAND,MAKEWPARAM(ID_SEND,BN_CLICKED),(LPARAM)GetDlgItem(window,ID_SEND));
    if (!wait_response(window,120000)) return 5;

    wchar_t memory[MAX_PATH];
    wcsncpy(memory,argv[1],MAX_PATH-1);
    wchar_t *slash=wcsrchr(memory,L'\\');
    if (!slash) return 6;
    slash[1]=L'\0'; wcscat(memory,L"Atlas_1.0.memory");
    HANDLE file=CreateFileW(memory,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL,NULL);
    if (file==INVALID_HANDLE_VALUE) return 7;
    DWORD size=GetFileSize(file,NULL);
    char *saved=size&&size<65536?malloc((size_t)size+1):NULL;
    DWORD read=0;
    int saved_ok=saved&&ReadFile(file,saved,size,&read,NULL)&&read==size;
    CloseHandle(file);
    if (!saved_ok) { free(saved); return 8; }
    saved[read]='\0'; saved_ok=strstr(saved,"73142")!=NULL; free(saved);
    if (!saved_ok) return 9;

    SendMessageW(window,WM_CLOSE,0,0);
    WaitForSingleObject(first.hProcess,10000); CloseHandle(first.hProcess);
    PROCESS_INFORMATION second={0};
    if (!start_chat(argv[1],&second)) return 10;
    window=wait_for_window(15000);
    if (!window) return 11;
    int restored=log_contains(window,L"73142");
    printf("saved=%s restored=%s\n",saved_ok?"yes":"no",restored?"yes":"no");
    CloseHandle(second.hProcess);
    return restored?0:12;
}
