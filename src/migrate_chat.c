#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { int id; HWND found; } FIND;

static BOOL CALLBACK find_log(HWND child, LPARAM param)
{
    FIND *find=(FIND *)param;
    if (GetDlgCtrlID(child)==find->id) { find->found=child; return FALSE; }
    return TRUE;
}

int wmain(int argc, wchar_t **argv)
{
    if (argc!=4) return 2;
    HWND window=FindWindowW(argv[1],argv[2]);
    if (!window) return 3;
    FIND find={103,NULL};
    EnumChildWindows(window,find_log,(LPARAM)&find);
    if (!find.found) return 4;
    int count=GetWindowTextLengthW(find.found);
    if (count<=0 || count>100000) return 5;
    wchar_t *wide=calloc((size_t)count+1,sizeof(wchar_t));
    if (!wide) return 6;
    GetWindowTextW(find.found,wide,count+1);
    int bytes=WideCharToMultiByte(CP_UTF8,0,wide,-1,NULL,0,NULL,NULL);
    char *utf8=bytes>0?malloc((size_t)bytes):NULL;
    if (!utf8 || !WideCharToMultiByte(CP_UTF8,0,wide,-1,utf8,bytes,NULL,NULL)) {
        free(wide); free(utf8); return 7;
    }
    FILE *out=_wfopen(argv[3],L"wb");
    if (!out) { free(wide); free(utf8); return 8; }
    fwrite(utf8,1,(size_t)bytes-1,out);
    fclose(out); free(wide); free(utf8);
    return 0;
}
