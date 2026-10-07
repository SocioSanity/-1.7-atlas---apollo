#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <shellapi.h>
#if !defined(ATLAS_COMPACT)
#include "atlas_model_registry.h"
#include "atlas_orchestrator.h"
#endif

#define ID_PROMPT 101
#define ID_SEND 102
#define ID_LOG 103
#define ID_SEARCH 104
#define ID_FIND 105
#define ID_STATUS 106
#define ID_SEARCH_ALL 107
#define ID_MENU_CLEAR 120
#define ID_MENU_MODELS 121
#define ID_MENU_SPECIALISTS 122
#define ID_MENU_MONITOR 123
#define ID_MENU_ABOUT 124
#define ID_MENU_ROUTE_AUTO 125
#define ID_MENU_ROUTE_CHAIN 126
#define ID_MENU_ROUTE_SINGLE 127
#define ID_MENU_MODEL_AUTO 128
#define ID_MENU_MODEL_LANGUAGE 129
#define ID_MENU_MODEL_CODER 130
#define ID_MENU_REFERENCE 131
#define ID_MENU_NEW_CHAT 132
#define ID_CHAT_BASE 400
#define MAX_RECENT_CHATS 16
#define WM_APPEND_TEXT (WM_APP + 1)
#define WM_REQUEST_DONE (WM_APP + 2)
#define WM_INFERENCE_PROGRESS (WM_APP + 3)

static HWND prompt_edit, output_box, send_button, search_edit, status_label, search_all_checkbox;
static HFONT ui_font;
static HMENU settings_menu;
static HMENU recent_chats_menu;
static wchar_t chats_directory[MAX_PATH];
static wchar_t recent_chat_paths[MAX_RECENT_CHATS][MAX_PATH];
static size_t recent_chat_count;
static size_t last_find_position;
static int route_mode = 0;       /* 0 automatic, 1 always chain, 2 single model */
static int model_preference = 0; /* 0 automatic, 1 language, 2 coder */
static int allow_reference = 1;
static wchar_t settings_path[MAX_PATH];
#if defined(ATLAS_COMPACT)
#define INFER_EXE "atlas_compact_infer.exe"
#define MODEL_LABEL L"Atlas LLM 1.0 Compact"
#define WINDOW_TITLE L"Atlas LLM 1.0 Compact Chat"
static const wchar_t *model_path = L"model\\atlas-llm-1.0-compact.gguf";
#else
#define INFER_EXE "atlas_1_0_infer.exe"
#define MODEL_LABEL L"Atlas LLM 1.0"
#define WINDOW_TITLE L"Atlas LLM 1.0 Chat"
static const wchar_t *model_path = L"model\\specialists\\qwen\\qwen2.5-coder-1.5b-q4_k_m.gguf";
static const wchar_t *language_model_path = L"model\\atlas-llm-1.0-language.gguf";
#endif
static char *conversation;
static wchar_t memory_path[MAX_PATH];
static wchar_t reference_path[MAX_PATH];
static int conversation_restored;
static size_t rendered_turn_count;

static const char system_message[] =
    "<|im_start|>system\n"
    "You are Atlas, a fast, capable general assistant and programmer. Be natural, candid, curious, and useful. "
    "The user's target is GPT-5o-like: fast and accurate across everyday language and programming, with enough confidence to push back when warranted. "
    "Answer the actual question directly, with no filler or repeated question. Keep simple answers short and conversational. "
    "For difficult work, reason carefully and present the key reasoning, checks, and conclusion without exposing hidden chain-of-thought. "
    "Do not agree just to be agreeable: respectfully challenge false premises, bad plans, and incorrect claims, and explain the strongest reason. "
    "Do not force false neutrality or preachy boilerplate; match the user's level of formality, and use blunt language when it fits. "
    "When people reasonably disagree, distinguish evidence from opinion. Be comfortable saying you are uncertain; never invent facts, citations, test results, or capabilities. "
    "Use relevant context supplied in this chat and the user's saved preference profile. Saved chats can be reopened or searched; do not claim access to a chat that is not supplied. "
    "For programming requests, follow every stated constraint, provide complete usable code when asked, and be precise about edge cases and whether the code was actually run. "
    "For general questions, give a clear answer with a useful example when it helps. Ask a follow-up only when missing information prevents a sound answer.\n"
    "<|im_end|>\n";

static wchar_t *utf8_to_wide(const char *s);
static int message_requests_implementation(const char *message);
static void render_saved_conversation(void);
static char *wide_to_utf8(const wchar_t *wide);
static int find_across_chats(HWND window);

typedef struct {
    HWND window;
    char *prompt_utf8;
    unsigned min_tokens, max_tokens;
    int route_mode;
    int model_preference;
    int allow_reference;
    int code_model;
    ULONGLONG deadline_tick;
    volatile LONG stage;
} Request;
typedef struct { char *context; char *text; } Response;

static int contains_ascii_ci(const char *text,const char *word)
{
    for (const char *p=text; *p; ++p) {
        const char *a=p,*b=word;
        while (*a && *b && tolower((unsigned char)*a)==tolower((unsigned char)*b)) { ++a; ++b; }
        if (!*b) return 1;
    }
    return 0;
}

static int contains_ascii_word_ci(const char *text, const char *word)
{
    size_t n = strlen(word);
    if (!n) return 0;
    for (const char *p = text; *p; ++p) {
        if (p != text && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) continue;
        size_t i = 0;
        while (i < n && p[i] &&
               tolower((unsigned char)p[i]) == tolower((unsigned char)word[i])) ++i;
        if (i == n && !isalnum((unsigned char)p[i]) && p[i] != '_') return 1;
    }
    return 0;
}

static unsigned min_reply_tokens(const char *message)
{
    static const char *signals[]={
        "explain","discuss","compare","analyze","analyse","reason",
        "pros and cons","write","implement",
        "function","debug","code","how does","how do","how to","should","why",
        "solve","calculate","perimeter","dimension","equation","what causes",
        "evaluate","step by step","break down","multi-step","multi step",
        "work through","derive","large problem",
        "complex problem","reasoning","plan the steps"
    };
    for (size_t i=0;i<sizeof(signals)/sizeof(signals[0]);++i)
        if (contains_ascii_ci(message,signals[i])) return 24;
    if (contains_ascii_word_ci(message,"prove")) return 24;
    return strlen(message)>140 ? 24 : 0;
}

static int message_needs_decomposition(const char *message)
{
    static const char *signals[]={
        "step by step","break down","multi-step","multi step","work through",
        "derive","analyze","analyse","large problem","complex problem",
        "reason through","plan the steps","evaluate","compare"
    };
    for (size_t i=0;i<sizeof(signals)/sizeof(signals[0]);++i)
        if (contains_ascii_ci(message,signals[i])) return 1;
    if (contains_ascii_word_ci(message,"prove")) return 1;
    return 0;
}

static int message_needs_code_model(const char *message)
{
    if (!message || contains_ascii_ci(message,"do not write source") ||
        contains_ascii_ci(message,"don't write source") ||
        contains_ascii_ci(message,"do not write code") ||
        contains_ascii_ci(message,"don't write code") ||
        contains_ascii_ci(message,"no code yet") ||
        contains_ascii_ci(message,"without writing code")) return 0;
    if (message_requests_implementation(message)) return 1;
    static const char *signals[]={
        "code","program","programming","compiler","compile","debug",
        "source code","header","pointer","array","syntax","segfault",
        "algorithm","makefile","gcc","api","bug","in c","c code","c program",
        "write a function","implement a function","c function","function in c"
    };
    for (size_t i=0;i<sizeof(signals)/sizeof(signals[0]);++i)
        if (contains_ascii_ci(message,signals[i])) return 1;
    return 0;
}

static int message_requests_implementation(const char *message)
{
    if (!message || contains_ascii_ci(message,"do not write source") ||
        contains_ascii_ci(message,"don't write source") ||
        contains_ascii_ci(message,"do not write code") ||
        contains_ascii_ci(message,"don't write code") ||
        contains_ascii_ci(message,"no code yet") ||
        contains_ascii_ci(message,"without writing code")) return 0;
    if (contains_ascii_ci(message,"function") &&
        (contains_ascii_ci(message,"write") || contains_ascii_ci(message,"implement") ||
         contains_ascii_ci(message,"create") || contains_ascii_ci(message,"provide") ||
         contains_ascii_ci(message,"generate"))) return 1;
    if (contains_ascii_ci(message,"program") &&
        (contains_ascii_ci(message,"write") || contains_ascii_ci(message,"implement") ||
         contains_ascii_ci(message,"create") || contains_ascii_ci(message,"generate"))) return 1;
    return contains_ascii_ci(message,"write code") ||
           contains_ascii_ci(message,"write a program") ||
           contains_ascii_ci(message,"write a function") ||
           contains_ascii_ci(message,"write a complete") ||
           contains_ascii_ci(message,"write complete") ||
           contains_ascii_ci(message,"write source code") ||
           contains_ascii_ci(message,"complete c source") ||
           contains_ascii_ci(message,"compile-ready") ||
           contains_ascii_ci(message,"provide source code") ||
           contains_ascii_ci(message,"provide code") ||
           contains_ascii_ci(message,"implement a ") ||
           contains_ascii_ci(message,"implement the ") ||
           contains_ascii_ci(message,"implement this") ||
           contains_ascii_ci(message,"implement it") ||
           contains_ascii_ci(message,"create a program") ||
           contains_ascii_ci(message,"generate code") ||
           contains_ascii_ci(message,"full source") ||
           contains_ascii_ci(message,"c implementation");
}

static int response_contains_code(const char *response)
{
    return response && (strstr(response,"```") || contains_ascii_ci(response,"#include") ||
                        contains_ascii_ci(response,"int main(") || contains_ascii_ci(response,"int main ("));
}

static int response_satisfies_code_request(const char *question,const char *response)
{
    if(!question||!response||!response_contains_code(response))return 0;
    if(contains_ascii_ci(question,"size_t")&&!contains_ascii_ci(response,"size_t"))return 0;
    if(contains_ascii_ci(question,"parse_size")&&!contains_ascii_ci(response,"parse_size"))return 0;
    if(contains_ascii_ci(question,"overflow")&&
       !contains_ascii_ci(response,"SIZE_MAX")&&!contains_ascii_ci(response,"ERANGE"))return 0;
    if(contains_ascii_ci(question,"empty")&&
       !contains_ascii_ci(response,"!*text")&&!contains_ascii_ci(response,"*text ==")&&
       !contains_ascii_ci(response,"text[0] =="))return 0;
    if((contains_ascii_ci(question,"nondigit")||contains_ascii_ci(question,"non-digit")||
        contains_ascii_ci(question,"signs"))&&
       !contains_ascii_ci(response,"isdigit")&&!contains_ascii_ci(response,"<'0'")&&
       !contains_ascii_ci(response,">'9'"))return 0;
    if(contains_ascii_ci(question,"unchanged")&&
       (!contains_ascii_ci(response,"*out =")||!contains_ascii_ci(response,"*out")))return 0;
    if(contains_ascii_ci(question,"return 1")&&contains_ascii_ci(question,"return 0")&&
       (!contains_ascii_ci(response,"return 1")||!contains_ascii_ci(response,"return 0")||
        contains_ascii_ci(response,"return -1")))return 0;
    if(contains_ascii_ci(question,"trailing character")&&
       !contains_ascii_ci(response,"*end")&&!contains_ascii_ci(response,"endptr")&&
       !contains_ascii_ci(response,"strtou"))return 0;
    return 1;
}

#if !defined(ATLAS_COMPACT)
static int chain_model_supported(const AtlasModel *model)
{
    if (!model || (strcmp(model->architecture, "qwen2") != 0 &&
                   strcmp(model->architecture, "qwen3") != 0) ||
        /* This machine falls back to CPU for the 5 GB Qwen3 model; routing
           it automatically made short answers take tens of seconds. */
        model->file_size > UINT64_C(2000000000)) return 0;
    for (size_t i = 0; i < model->tensor_count; ++i) {
        const AtlasTensorInfo *t = &model->tensors[i];
        if (t->dimensions < 2) continue;
        if (t->type != 1 && t->type != 6 && t->type != 7 && t->type != 8 && t->type != 12 &&
            t->type != 13 && t->type != 14)
            return 0;
    }
    return 1;
}

static int register_chain_specialist(const char *name, const char *path,
                                     float reasoning, float coding, float research,
                                     float math, float analysis)
{
    WIN32_FILE_ATTRIBUTE_DATA attributes;
    double bytes = 0.0;
    if (GetFileAttributesExA(path, GetFileExInfoStandard, &attributes))
        bytes = (double)(((uint64_t)attributes.nFileSizeHigh << 32) | attributes.nFileSizeLow);
    float speed_prior = bytes > 0.0 ? (float)(1000000000.0 / bytes) : 1.0f;
    return atlas_register_specialist(name, path, reasoning, coding,
                                      research, math, analysis, speed_prior);
}

static char *run_chained_request(const char *problem, int allow_reference)
{
    AtlasModelRegistry registry;
    AtlasReasoningSession *session = calloc(1, sizeof(*session));
    if (!session) return NULL;
    int code_task = message_requests_implementation(problem) ||
        contains_ascii_ci(problem, "programming") ||
        contains_ascii_ci(problem, "source code") ||
        contains_ascii_ci(problem, "debug") ||
        contains_ascii_ci(problem, "compiler");
    const char *router_model = code_task
        ? "model\\atlas-llm-1.0-compact.gguf"
        : "model\\atlas-llm-1.0-language.gguf";
    const char *coordinator_model = "model\\atlas-llm-1.0.gguf";
    atlas_registry_init(&registry, "model\\specialists");
    if (!atlas_registry_scan(&registry)) {
        atlas_registry_free(&registry); free(session); return NULL;
    }
    atlas_clear_specialists();
    int registered = 0;
    if (register_chain_specialist("Qwen language 0.5B",
            "model\\atlas-llm-1.0-language.gguf", 0.65f, 0.20f,
            0.60f, 0.35f, 0.65f) == 0) ++registered;
    if (register_chain_specialist("Qwen coder 0.5B",
            "model\\atlas-llm-1.0-compact.gguf", 0.45f, 1.00f,
            0.25f, 0.25f, 0.50f) == 0) ++registered;
    for (size_t i = 0; i < registry.count && i < ATLAS_MAX_SPECIALISTS; ++i) {
        AtlasModel *m = &registry.models[i];
        if (!chain_model_supported(m)) continue;
        int is_qwen3 = strcmp(m->architecture,"qwen3")==0;
        int is_code = contains_ascii_ci(m->name, "coder") || contains_ascii_ci(m->name, "code");
        int is_math = contains_ascii_ci(m->name, "math");
        float reasoning = is_qwen3 ? 0.75f : 0.75f;
        float coding = is_code ? 1.0f : (is_qwen3 ? 0.80f : 0.35f);
        float math = is_math ? 1.0f : (is_qwen3 ? 0.65f : 0.35f);
        float research = contains_ascii_ci(m->name, "instruct") ? 0.7f : (is_qwen3 ? 0.75f : 0.4f);
        if (register_chain_specialist(m->name, m->path, reasoning, coding,
                                       research, math, 0.6f) == 0)
            ++registered;
    }
    if (registered && atlas_reason_staged(".\\specialist_loader.exe",
            router_model, coordinator_model, allow_reference,
            problem, session) == 0 &&
        session->synthesis[0]) {
        if (message_requests_implementation(problem) && !session->verified) {
            char *rejected=_strdup("[UNVERIFIED: The specialist chain could not validate a complete, constraint-compliant implementation. See the chain monitor for its draft and verification findings.]");
            atlas_clear_specialists();
            atlas_registry_free(&registry);
            free(session);
            return rejected;
        }
        size_t n = strlen(session->synthesis) + 192;
        char *answer = malloc(n);
        if (answer) {
            snprintf(answer, n, "%s%s", session->synthesis,
                session->verified ? "" :
                "\r\n\r\n[Atlas's verification pass flagged unresolved issues. See the chain monitor.]\r\n");
            atlas_clear_specialists();
            atlas_registry_free(&registry);
            free(session);
            return answer;
        }
    }
    atlas_clear_specialists();
    atlas_registry_free(&registry);
    free(session);
    return NULL;
}
#endif

static void start_chain_monitor(void)
{
#if !defined(ATLAS_COMPACT)
    HWND existing=FindWindowW(L"AtlasChainMonitor",NULL);
    if (existing) { ShowWindow(existing,SW_RESTORE); SetForegroundWindow(existing); return; }
    STARTUPINFOW si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);
    if (CreateProcessW(L"atlas_chain_monitor.exe", NULL, NULL, NULL, FALSE,
                       0, NULL, NULL, &si, &pi)) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
#endif
}

static void append_wide(const wchar_t *s)
{
    int n = GetWindowTextLengthW(output_box);
    SendMessageW(output_box, EM_SETSEL, n, n);
    SendMessageW(output_box, EM_REPLACESEL, FALSE, (LPARAM)s);
}

static void post_status(Request *req,const wchar_t *label,size_t output_bytes)
{
    wchar_t text[160];
    if (output_bytes)
        _snwprintf(text,159,L"%ls  •  %zu bytes generated",label,output_bytes);
    else
        _snwprintf(text,159,L"%ls",label);
    text[159]=0;
    size_t n=wcslen(text)+1;
    wchar_t *copy=malloc(n*sizeof(wchar_t));
    if (!copy) return;
    memcpy(copy,text,n*sizeof(wchar_t));
    if (!PostMessageW(req->window,WM_INFERENCE_PROGRESS,0,(LPARAM)copy)) free(copy);
}

static void apply_ui_font(HWND control)
{
    if (control && ui_font) SendMessageW(control,WM_SETFONT,(WPARAM)ui_font,TRUE);
}

static void find_in_chat(HWND window)
{
    if (search_all_checkbox && SendMessageW(search_all_checkbox,BM_GETCHECK,0,0)==BST_CHECKED) {
        (void)find_across_chats(window);
        return;
    }
    int query_len=GetWindowTextLengthW(search_edit);
    int text_len=GetWindowTextLengthW(output_box);
    if (query_len<=0 || text_len<=0) {
        SetWindowTextW(status_label,L"Type a word or phrase to search this chat.");
        return;
    }
    wchar_t *query=malloc(((size_t)query_len+1)*sizeof(wchar_t));
    wchar_t *text=malloc(((size_t)text_len+1)*sizeof(wchar_t));
    if (!query || !text) { free(query); free(text); return; }
    GetWindowTextW(search_edit,query,query_len+1);
    GetWindowTextW(output_box,text,text_len+1);
    size_t q=(size_t)query_len,n=(size_t)text_len,found=SIZE_MAX;
    size_t start=last_find_position<n?last_find_position:0;
    for (size_t i=start;i+q<=n;++i)
        if (!_wcsnicmp(text+i,query,q)) { found=i; break; }
    if (found==SIZE_MAX && start)
        for (size_t i=0;i<start && i+q<=n;++i)
            if (!_wcsnicmp(text+i,query,q)) { found=i; break; }
    if (found!=SIZE_MAX) {
        SendMessageW(output_box,EM_SETSEL,(WPARAM)found,(LPARAM)(found+q));
        SendMessageW(output_box,EM_SCROLLCARET,0,0);
        SetFocus(output_box);
        last_find_position=found+q;
        SetWindowTextW(status_label,L"Match found. Press Find again to continue.");
    } else {
        last_find_position=0;
        SetWindowTextW(status_label,L"No match in this conversation.");
    }
    free(query); free(text); (void)window;
}

static void init_memory_path(void)
{
    wchar_t module[MAX_PATH];
    DWORD n=GetModuleFileNameW(NULL,module,MAX_PATH);
    if (!n || n>=MAX_PATH) return;
    wchar_t *slash=wcsrchr(module,L'\\');
    if (!slash) return;
    slash[1]=L'\0';
    if (wcslen(module)+wcslen(L"Atlas_1.0.memory")>=MAX_PATH) return;
    wchar_t legacy_path[MAX_PATH];
    wcscpy(legacy_path,module);
    wcscat(legacy_path,L"Atlas_1.0.memory");
    wcscpy(chats_directory,module);
    wcscat(chats_directory,L"Chats");
    if (CreateDirectoryW(chats_directory,NULL) || GetLastError()==ERROR_ALREADY_EXISTS) {
        wchar_t first_chat[MAX_PATH];
        _snwprintf(first_chat,_countof(first_chat)-1,L"%ls\\chat-legacy.memory",chats_directory);
        first_chat[_countof(first_chat)-1]=0;
        if (GetFileAttributesW(first_chat)==INVALID_FILE_ATTRIBUTES &&
            GetFileAttributesW(legacy_path)!=INVALID_FILE_ATTRIBUTES &&
            !CopyFileW(legacy_path,first_chat,TRUE))
            wcscpy(memory_path,legacy_path);
        else
            wcscpy(memory_path,first_chat);
    } else {
        wcscpy(memory_path,legacy_path);
    }
    wcscpy(settings_path,module);
    wcscat(settings_path,L"Atlas_1.0.ini");
    if (wcslen(module)+wcslen(L"Atlas_1.0.references")<MAX_PATH) {
        wcscpy(reference_path,module);
        wcscat(reference_path,L"Atlas_1.0.references");
    }
}

static char *latest_user_text(const char *prompt)
{
    static const char tag[]="<|im_start|>user\n";
    const char *last=NULL,*scan=prompt;
    while ((scan=strstr(scan,tag))!=NULL) { last=scan+sizeof(tag)-1; scan=last; }
    if (!last) return _strdup("");
    const char *end=strstr(last,"<|im_end|>");
    if (!end) end=last+strlen(last);
    while (last<end && (*last==' '||*last=='\t'||*last=='\r'||*last=='\n')) ++last;
    while (end>last && (end[-1]==' '||end[-1]=='\t'||end[-1]=='\r'||end[-1]=='\n')) --end;
    size_t n=(size_t)(end-last);
    char *text=malloc(n+1);
    if (text) { memcpy(text,last,n); text[n]=0; }
    return text;
}

static char *recent_chat_prompt(const char *prompt)
{
    static const char user_tag[] = "<|im_start|>user\n";
    static const char end_tag[] = "<|im_end|>\n";
    const char *system_end = strstr(prompt, end_tag);
    if (!system_end) return _strdup(prompt);
    system_end += sizeof(end_tag) - 1;
    const char *recent[2] = {0};
    size_t count = 0;
    const char *scan = prompt;
    while ((scan = strstr(scan, user_tag)) != NULL) {
        if (count < 2) recent[count] = scan;
        else { recent[0] = recent[1]; recent[1] = scan; }
        ++count;
        scan += sizeof(user_tag) - 1;
    }
    if (count < 3 || !recent[0] || recent[0] < system_end) return _strdup(prompt);
    size_t prefix = (size_t)(system_end - prompt);
    size_t tail = strlen(recent[0]);
    if (prefix > SIZE_MAX - tail - 1) return NULL;
    char *out = malloc(prefix + tail + 1);
    if (!out) return NULL;
    memcpy(out, prompt, prefix);
    memcpy(out + prefix, recent[0], tail + 1);
    return out;
}

static char *normalize_question(const char *question)
{
    size_t n=strlen(question),out=0;
    char *key=malloc(n+1);
    if (!key) return NULL;
    int pending_space=0;
    for (size_t i=0;i<n;++i) {
        unsigned char c=(unsigned char)question[i];
        if (isspace(c)) { pending_space=out>0; continue; }
        if (ispunct(c) && c<128) continue;
        if (pending_space) key[out++]=' ';
        pending_space=0;
        key[out++]=(char)tolower(c);
    }
    key[out]=0;
    return key;
}

static char *find_reference(const char *question)
{
    if (!reference_path[0] || !question || !*question) return NULL;
    HANDLE file=CreateFileW(reference_path,GENERIC_READ,FILE_SHARE_READ,NULL,
                            OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if (file==INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file,&size) || size.QuadPart<=0 || size.QuadPart>16*1024*1024) {
        CloseHandle(file); return NULL;
    }
    char *data=malloc((size_t)size.QuadPart+1);
    DWORD got=0;
    if (!data || !ReadFile(file,data,(DWORD)size.QuadPart,&got,NULL) || got!=(DWORD)size.QuadPart) {
        free(data); CloseHandle(file); return NULL;
    }
    data[got]=0;
    CloseHandle(file);
    char *answer=NULL;
    const char *p=data,*limit=data+got;
    while (p<limit) {
        char *end=NULL;
        unsigned long long qn=strtoull(p,&end,10);
        if (end==p || end>=limit || *end!=' ') break;
        p=end+1;
        unsigned long long an=strtoull(p,&end,10);
        if (end==p || end>=limit || *end!='\n') break;
        p=end+1;
        if (qn>(unsigned long long)(limit-p) || an>(unsigned long long)(limit-p)-(size_t)qn ||
            p+(size_t)qn+(size_t)an>=limit) break;
        if (strlen(question)==(size_t)qn && !memcmp(p,question,(size_t)qn)) {
            answer=malloc((size_t)an+1);
            if (answer) { memcpy(answer,p+(size_t)qn,(size_t)an); answer[an]=0; }
            break;
        }
        p+=(size_t)qn+(size_t)an;
        if (*p=='\n') ++p;
    }
    free(data);
    return answer;
}

static void save_reference(const char *question,const char *answer)
{
    if (!reference_path[0] || !question || !answer || !*question || !*answer) return;
    size_t qn=strlen(question),an=strlen(answer);
    if (qn>65535 || an>262144) return;
    HANDLE file=CreateFileW(reference_path,FILE_APPEND_DATA,FILE_SHARE_READ,NULL,
                            OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL);
    if (file==INVALID_HANDLE_VALUE) return;
    char header[64];
    int hn=snprintf(header,sizeof(header),"%zu %zu\n",qn,an);
    DWORD written=0;
    if (hn>0 && WriteFile(file,header,(DWORD)hn,&written,NULL) && written==(DWORD)hn &&
        WriteFile(file,question,(DWORD)qn,&written,NULL) && written==(DWORD)qn &&
        WriteFile(file,answer,(DWORD)an,&written,NULL) && written==(DWORD)an)
        WriteFile(file,"\n",1,&written,NULL);
    FlushFileBuffers(file);
    CloseHandle(file);
}

static char *load_memory(void)
{
    if (!memory_path[0]) return NULL;
    HANDLE file=CreateFileW(memory_path,GENERIC_READ,FILE_SHARE_READ,NULL,
                            OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if (file==INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file,&size) || size.QuadPart<=0 || size.QuadPart>65536) {
        CloseHandle(file); return NULL;
    }
    char *data=malloc((size_t)size.QuadPart+1);
    DWORD read=0;
    if (!data || !ReadFile(file,data,(DWORD)size.QuadPart,&read,NULL) ||
        read!=(DWORD)size.QuadPart) { free(data); data=NULL; }
    if (data) data[read]='\0';
    CloseHandle(file);
    return data;
}

static void save_memory(void)
{
    if (!memory_path[0] || !conversation) return;
    HANDLE file=CreateFileW(memory_path,GENERIC_WRITE,FILE_SHARE_READ,NULL,
                            CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL);
    if (file==INVALID_HANDLE_VALUE) return;
    size_t n=strlen(conversation);
    size_t offset=0;
    while (offset<n) {
        DWORD chunk=(DWORD)((n-offset)>65536?65536:(n-offset));
        DWORD written=0;
        if (!WriteFile(file,conversation+offset,chunk,&written,NULL) || !written) break;
        offset+=written;
    }
    FlushFileBuffers(file);
    CloseHandle(file);
}

typedef struct { wchar_t path[MAX_PATH]; FILETIME written; } ChatCandidate;

static void chat_title_from_file(const wchar_t *path,wchar_t *title,size_t capacity)
{
    if (!capacity) return;
    wcscpy_s(title,capacity,L"Conversation");
    HANDLE file=CreateFileW(path,GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,NULL,
                            OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(file==INVALID_HANDLE_VALUE)return;
    char data[4097];DWORD got=0;
    BOOL ok=ReadFile(file,data,sizeof(data)-1,&got,NULL);CloseHandle(file);
    if(!ok||!got)return;
    data[got]=0;char *start=strstr(data,"<|im_start|>user");if(!start)return;
    start+=sizeof("<|im_start|>user")-1;
    if(*start=='\r')++start;
    if(*start!='\n')return;
    ++start;char *end=strstr(start,"<|im_end|>");if(!end)return;
    while(start<end&&(*start==' '||*start=='\r'||*start=='\n'||*start=='\t'))++start;
    if(start==end)return;
    size_t n=(size_t)(end-start);if(n>256)n=256;
    char text[257];memcpy(text,start,n);text[n]=0;
    for(size_t i=0;i<n;++i)if(text[i]=='\r'||text[i]=='\n'||text[i]=='\t')text[i]=' ';
    wchar_t *wide=utf8_to_wide(text);if(!wide)return;
    wcsncpy_s(title,capacity,wide,_TRUNCATE);free(wide);
}

static void refresh_recent_chats(void)
{
    if(!recent_chats_menu)return;
    while(GetMenuItemCount(recent_chats_menu)>0)DeleteMenu(recent_chats_menu,0,MF_BYPOSITION);
    recent_chat_count=0;
    wchar_t pattern[MAX_PATH];
    _snwprintf(pattern,_countof(pattern)-1,L"%ls\\*.memory",chats_directory);
    pattern[_countof(pattern)-1]=0;
    WIN32_FIND_DATAW data;HANDLE find=FindFirstFileW(pattern,&data);
    if(find==INVALID_HANDLE_VALUE){AppendMenuW(recent_chats_menu,MF_STRING|MF_GRAYED,0,L"No saved chats");return;}
    ChatCandidate items[MAX_RECENT_CHATS];size_t count=0;
    do{
        if(data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)continue;
        ChatCandidate candidate={0};candidate.written=data.ftLastWriteTime;
        _snwprintf(candidate.path,_countof(candidate.path)-1,L"%ls\\%ls",chats_directory,data.cFileName);
        candidate.path[_countof(candidate.path)-1]=0;
        size_t slot=0;while(slot<count&&CompareFileTime(&items[slot].written,&candidate.written)>=0)++slot;
        if(slot>=MAX_RECENT_CHATS)continue;
        size_t new_count=count<MAX_RECENT_CHATS?count+1:count;
        for(size_t j=new_count-1;j>slot;--j)items[j]=items[j-1];
        items[slot]=candidate;count=new_count;
    }while(FindNextFileW(find,&data));
    FindClose(find);recent_chat_count=count;
    for(size_t i=0;i<count;++i){
        recent_chat_paths[i][0]=0;wcsncpy_s(recent_chat_paths[i],MAX_PATH,items[i].path,_TRUNCATE);
        wchar_t title[96],label[128];chat_title_from_file(items[i].path,title,_countof(title));
        _snwprintf(label,_countof(label)-1,L"%ls%ls",title,
            _wcsicmp(items[i].path,memory_path)==0?L"  (current)":L"");
        label[_countof(label)-1]=0;
        AppendMenuW(recent_chats_menu,MF_STRING,ID_CHAT_BASE+(UINT)i,label);
    }
    if(!count)AppendMenuW(recent_chats_menu,MF_STRING|MF_GRAYED,0,L"No saved chats");
}

static void render_active_chat(HWND window)
{
    if(!conversation)conversation=_strdup(system_message);
    SetWindowTextW(output_box,MODEL_LABEL L"\r\nLocal inference. Recent conversations are available under File.\r\n");
    render_saved_conversation();
    wchar_t status[96];
    _snwprintf(status,_countof(status)-1,L"Loaded chat (%zu bytes, %zu turns)",conversation?strlen(conversation):0,rendered_turn_count);
    status[_countof(status)-1]=0;SetWindowTextW(status_label,status);
    DrawMenuBar(window);
}

static void switch_chat(HWND window,const wchar_t *path)
{
    if(!path||!path[0]||!IsWindowEnabled(send_button))return;
    save_memory();
    wcsncpy_s(memory_path,MAX_PATH,path,_TRUNCATE);
    free(conversation);conversation=load_memory();
    if(!conversation)conversation=_strdup(system_message);
    conversation_restored=1;last_find_position=0;
    render_active_chat(window);refresh_recent_chats();
}

static int find_across_chats(HWND window)
{
    int qn=GetWindowTextLengthW(search_edit);
    if(qn<=0){SetWindowTextW(status_label,L"Type a word or phrase to search saved chats.");return 0;}
    wchar_t *wide=malloc(((size_t)qn+1)*sizeof(wchar_t));
    if(!wide)return 0;
    GetWindowTextW(search_edit,wide,qn+1);
    char *query=wide_to_utf8(wide);free(wide);
    if(!query||!*query){free(query);return 0;}
    save_memory();
    wchar_t pattern[MAX_PATH];
    _snwprintf(pattern,_countof(pattern)-1,L"%ls\\*.memory",chats_directory);
    pattern[_countof(pattern)-1]=0;
    WIN32_FIND_DATAW data;HANDLE find=FindFirstFileW(pattern,&data);
    wchar_t match[MAX_PATH]={0};
    if(find!=INVALID_HANDLE_VALUE){
        do{
            if(data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)continue;
            wchar_t path[MAX_PATH];
            _snwprintf(path,_countof(path)-1,L"%ls\\%ls",chats_directory,data.cFileName);
            path[_countof(path)-1]=0;
            HANDLE file=CreateFileW(path,GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,NULL,
                                    OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
            if(file==INVALID_HANDLE_VALUE)continue;
            LARGE_INTEGER size;char *text=NULL;DWORD got=0;
            if(GetFileSizeEx(file,&size)&&size.QuadPart>0&&size.QuadPart<=65536){
                text=malloc((size_t)size.QuadPart+1);
                if(!text||!ReadFile(file,text,(DWORD)size.QuadPart,&got,NULL)||got!=(DWORD)size.QuadPart){free(text);text=NULL;}
                if(text)text[got]=0;
            }
            CloseHandle(file);
            if(text){int found=0;size_t q=strlen(query),n=strlen(text);
                if(q<=n)for(size_t i=0;i+q<=n;++i){size_t j=0;while(j<q&&tolower((unsigned char)text[i+j])==tolower((unsigned char)query[j]))++j;if(j==q){found=1;break;}}
                free(text);if(found){wcsncpy_s(match,MAX_PATH,path,_TRUNCATE);break;}
            }
        }while(FindNextFileW(find,&data));
        FindClose(find);
    }
    free(query);
    if(!match[0]){SetWindowTextW(status_label,L"No match in saved chats.");return 0;}
    switch_chat(window,match);
    if(search_all_checkbox)SendMessageW(search_all_checkbox,BM_SETCHECK,BST_UNCHECKED,0);
    find_in_chat(window);
    return 1;
}

static void create_new_chat(HWND window)
{
    if(!IsWindowEnabled(send_button))return;
    save_memory();
    FILETIME now;GetSystemTimeAsFileTime(&now);
    ULARGE_INTEGER stamp;stamp.LowPart=now.dwLowDateTime;stamp.HighPart=now.dwHighDateTime;
    wchar_t path[MAX_PATH];
    _snwprintf(path,_countof(path)-1,L"%ls\\chat-%016llx.memory",chats_directory,
               (unsigned long long)stamp.QuadPart);path[_countof(path)-1]=0;
    HANDLE file=CreateFileW(path,GENERIC_WRITE,FILE_SHARE_READ,NULL,CREATE_NEW,
                            FILE_ATTRIBUTE_NORMAL,NULL);
    if(file==INVALID_HANDLE_VALUE){SetWindowTextW(status_label,L"Could not create a new chat.");return;}
    CloseHandle(file);wcsncpy_s(memory_path,MAX_PATH,path,_TRUNCATE);
    free(conversation);conversation=_strdup(system_message);
    save_memory();last_find_position=0;conversation_restored=1;
    render_active_chat(window);refresh_recent_chats();
}

static int append_render_text(wchar_t *dst,size_t capacity,size_t *used,const wchar_t *text)
{
    size_t n=wcslen(text);
    if(n>=capacity-*used)return 0;
    memcpy(dst+*used,text,n*sizeof(wchar_t));*used+=n;dst[*used]=0;return 1;
}

static void render_saved_conversation(void)
{
    if (!conversation) return;
    size_t bytes=strlen(conversation);
    if(bytes>(SIZE_MAX-256)/2)return;
    size_t capacity=bytes*2+256;
    if(capacity>SIZE_MAX/sizeof(wchar_t))return;
    wchar_t *render=calloc(capacity,sizeof(wchar_t));
    if(!render)return;
    size_t used=0;
    size_t parsed_turns=0;
    const wchar_t *header=MODEL_LABEL L"\r\nLocal inference. Recent conversations are available under File.\r\n";
    if(!append_render_text(render,capacity,&used,header)){free(render);return;}
    const char *p=conversation;
    const char *start_tag="<|im_start|>";
    const size_t tag_len=sizeof("<|im_start|>")-1;
    while ((p=strstr(p,start_tag))!=NULL) {
        const char *role=p+tag_len;
        const char *line=strchr(role,'\n');
        if (!line) break;
        const char *end=strstr(line+1,"<|im_end|>");
        if (!end) break;
        size_t role_len=(size_t)(line-role);
        if(role_len && role[role_len-1]=='\r')--role_len;
        int user=role_len==4 && !memcmp(role,"user",4);
        int assistant=role_len==9 && !memcmp(role,"assistant",9);
        if (user || assistant) {
            ++parsed_turns;
            if(!append_render_text(render,capacity,&used,user?L"\r\nYou: ":L"\r\nAtlas LLM 1.0: "))break;
            size_t n=(size_t)(end-(line+1));
            char *part=malloc(n+1);
            if (part) {
                memcpy(part,line+1,n); part[n]='\0';
                wchar_t *wide=utf8_to_wide(part);
                if (wide) { int ok=append_render_text(render,capacity,&used,wide); free(wide); if(!ok){free(part);break;} }
                free(part);
            }
            if(!append_render_text(render,capacity,&used,L"\r\n"))break;
        }
        p=end+sizeof("<|im_end|>")-1;
    }
    SetWindowTextW(output_box,render);free(render);
    rendered_turn_count=parsed_turns;
}

static char *wide_to_utf8(const wchar_t *wide)
{
    int n = WideCharToMultiByte(CP_UTF8, 0, wide, -1, NULL, 0, NULL, NULL);
    if (n <= 0) return NULL;
    char *s = malloc((size_t)n);
    if (!s || !WideCharToMultiByte(CP_UTF8, 0, wide, -1, s, n, NULL, NULL)) { free(s); return NULL; }
    return s;
}

static wchar_t *utf8_to_wide(const char *s)
{
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    if (n <= 0) return NULL;
    wchar_t *w = malloc((size_t)n * sizeof(wchar_t));
    if (!w || !MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n)) { free(w); return NULL; }
    return w;
}

static size_t complete_utf8_prefix(const char *s,size_t length)
{
    size_t i=0;
    while (i<length) {
        unsigned char c=(unsigned char)s[i]; size_t n;
        if (c<0x80) n=1;
        else if (c>=0xc2 && c<=0xdf) n=2;
        else if (c>=0xe0 && c<=0xef) n=3;
        else if (c>=0xf0 && c<=0xf4) n=4;
        else { ++i; continue; }
        if (i+n>length) break;
        int valid=1;
        for (size_t j=1;j<n;++j)
            if (((unsigned char)s[i+j]&0xc0)!=0x80) valid=0;
        if (n==3 && ((c==0xe0&&(unsigned char)s[i+1]<0xa0) ||
                     (c==0xed&&(unsigned char)s[i+1]>=0xa0))) valid=0;
        if (n==4 && ((c==0xf0&&(unsigned char)s[i+1]<0x90) ||
                     (c==0xf4&&(unsigned char)s[i+1]>=0x90))) valid=0;
        if (!valid) { ++i; continue; }
        i+=n;
    }
    return i;
}

static void quote_arg(char *dst, size_t cap, const char *src)
{
    size_t o = 0;
    if (cap < 3) return;
    dst[o++] = '"';
    for (size_t i = 0; src[i] && o + 3 < cap; ++i) {
        if (src[i] == '"' || src[i] == '\\') dst[o++] = '\\';
        dst[o++] = src[i];
    }
    dst[o++] = '"'; dst[o] = 0;
}

static char *run_child_capture(Request *req,const char *command,DWORD *exit_code)
{
    if (exit_code) *exit_code=1;
    SECURITY_ATTRIBUTES sa={sizeof(sa),NULL,TRUE};
    HANDLE reader=NULL,writer=NULL;
    if (!CreatePipe(&reader,&writer,&sa,0)) return NULL;
    SetHandleInformation(reader,HANDLE_FLAG_INHERIT,0);
    STARTUPINFOW si={0}; PROCESS_INFORMATION pi={0};
    si.cb=sizeof(si); si.dwFlags=STARTF_USESTDHANDLES;
    HANDLE nul=CreateFileW(L"NUL",GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,
                           &sa,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    si.hStdOutput=writer; si.hStdError=nul; si.hStdInput=nul;
    wchar_t *cmdw=utf8_to_wide(command);
    BOOL started=cmdw && CreateProcessW(NULL,cmdw,NULL,NULL,TRUE,CREATE_NO_WINDOW,
                                        NULL,NULL,&si,&pi);
    free(cmdw); CloseHandle(writer);
    if (nul!=INVALID_HANDLE_VALUE) CloseHandle(nul);
    char *response=NULL;
    if (!started) response=_strdup("[Could not start inference executable.]");
    else {
        size_t used=0,cap=8192; response=malloc(cap);
        char buf[2048]; DWORD got=0; int timed_out=0; ULONGLONG last_status=0;
        for (;;) {
            DWORD available=0;
            if (!PeekNamedPipe(reader,NULL,0,NULL,&available,NULL)) break;
            if (!available) {
                if (WaitForSingleObject(pi.hProcess,0)==WAIT_OBJECT_0) break;
                ULONGLONG now=GetTickCount64();
                if (now>=req->deadline_tick) {
                    timed_out=1; TerminateProcess(pi.hProcess,124);
                    WaitForSingleObject(pi.hProcess,INFINITE); break;
                }
                if (now-last_status>=250) {
                    LONG stage=InterlockedCompareExchange(&req->stage,0,0);
                    post_status(req,stage?L"Atlas reference is generating":L"Atlas is generating",used);
                    last_status=now;
                }
                Sleep(40); continue;
            }
            DWORD amount=available<(DWORD)sizeof(buf)?available:(DWORD)sizeof(buf);
            if (!ReadFile(reader,buf,amount,&got,NULL) || !got) continue;
            if (!response) continue;
            if (used+(size_t)got+1>cap) {
                size_t next=cap;
                while (next<used+(size_t)got+1 && next<SIZE_MAX/2) next*=2;
                char *grown=next>=used+(size_t)got+1?realloc(response,next):NULL;
                if (!grown) { free(response); response=NULL; continue; }
                response=grown; cap=next;
            }
            memcpy(response+used,buf,got); used+=got; response[used]=0;
        }
        if (timed_out && response) {
            static const char timeout_note[]="\r\n[Response stopped at Atlas's five-minute limit; showing the partial answer.]\r\n";
            size_t a=strlen(response),b=sizeof(timeout_note)-1;
            char *grown=realloc(response,a+b+1);
            if (grown) { response=grown; memcpy(response+a,timeout_note,b+1); }
        }
        WaitForSingleObject(pi.hProcess,INFINITE);
        DWORD code=1; GetExitCodeProcess(pi.hProcess,&code);
        if (exit_code) *exit_code=code;
        CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
        if (!response) response=_strdup("");
    }
    CloseHandle(reader);
    if (response) {
        size_t n=strlen(response),valid=complete_utf8_prefix(response,n);
        response[valid]=0;
    }
    return response;
}

static float remove_confidence_marker(char *response,int *truncated)
{
    static const char marker[]="\036ATLAS_CONFIDENCE=";
    if (!response) return -1.0f;
    char *p=strstr(response,marker);
    if (!p) return -1.0f;
    char *value=p+sizeof(marker)-1;
    char *cut=strstr(value,",TRUNCATED=");
    if (truncated) *truncated=cut?atoi(cut+sizeof(",TRUNCATED=")-1):0;
    if (cut) *cut=0;
    float score=(float)atof(value);
    while (p>response && (p[-1]=='\r'||p[-1]=='\n')) --p;
    *p=0;
    return score;
}

static DWORD WINAPI inference_worker(void *param)
{
    Request *req = (Request *)param;
    char *model = wide_to_utf8(model_path);
#if !defined(ATLAS_COMPACT)
    char *language_model = wide_to_utf8(language_model_path);
    char *teacher_model=wide_to_utf8(L"model\\atlas-llm-1.0.gguf");
    char *qlanguage = NULL;
#endif
    char *question=latest_user_text(req->prompt_utf8);
    char *question_key=question?normalize_question(question):NULL;
    char *model_prompt=recent_chat_prompt(req->prompt_utf8);
    if(model_prompt&&req->code_model){
        static const char code_hint[]="\nReturn one concise, complete C answer only. Keep comments minimal. Include every requested constraint and all required headers; do not add explanations before or after the code.";
        size_t a=strlen(model_prompt),b=sizeof(code_hint)-1;
        if(a<=SIZE_MAX-b-1){char *grown=realloc(model_prompt,a+b+1);if(grown){model_prompt=grown;memcpy(model_prompt+a,code_hint,b+1);}}
    }
    size_t n = strlen(req->prompt_utf8) + 4096;
    char *qmodel = malloc(n), *qprompt = malloc(n), *command = malloc(n * 2 + 512);
#if !defined(ATLAS_COMPACT)
    qlanguage = malloc(n); char *qteacher=malloc(n);
#endif
    if (!model || !question || !question_key || !model_prompt || !qmodel || !qprompt || !command
#if !defined(ATLAS_COMPACT)
        || !language_model || !teacher_model || !qlanguage || !qteacher
#endif
    ) goto failed;
#if !defined(ATLAS_COMPACT)
    if (req->model_preference==1) { free(model); model=_strdup(language_model); if (!model) goto failed; }
#endif
    quote_arg(qmodel, n, model);
    quote_arg(qprompt, n, model_prompt);
    free(model_prompt); model_prompt=NULL;
    char *response=find_reference(question_key);
    DWORD exit_code=0;
    int from_reference=response!=NULL;
    if (from_reference) {
        post_status(req,L"Using a saved Atlas reference",0);
        wchar_t *part=utf8_to_wide(response);
        if (part && !PostMessageW(req->window,WM_APPEND_TEXT,0,(LPARAM)part)) free(part);
    }
#if !defined(ATLAS_COMPACT)
    if (!response && req->route_mode!=2 && (req->route_mode==1 || message_needs_decomposition(question))) {
        post_status(req,L"Breaking this request into specialist tasks",0);
        /* Route only the latest user turn. Passing the entire saved chat
           repeatedly inflated every specialist prompt and slowed inference. */
        response=run_chained_request(question,req->allow_reference);
        if (response) {
            exit_code=0;
            post_status(req,L"Atlas is combining and checking specialist answers",0);
        }
    }
#endif
    if (!response) {
    post_status(req,L"Starting Atlas's local model",0);
#if defined(ATLAS_COMPACT)
    snprintf(command,n*2+512,INFER_EXE " %s %s %u %u --confidence",qmodel,qprompt,req->max_tokens,req->min_tokens);
#else
    quote_arg(qlanguage, n, language_model);
    quote_arg(qteacher,n,teacher_model);
    snprintf(command,n*2+512,INFER_EXE " %s %s %u %u %s --confidence",qmodel,qprompt,req->max_tokens,req->min_tokens,qlanguage);
#endif
    response=run_child_capture(req,command,&exit_code);
    float confidence=remove_confidence_marker(response,NULL);
#if !defined(ATLAS_COMPACT)
    if (req->allow_reference && exit_code==0 && confidence>=0.0f && confidence<0.25f) {
        InterlockedExchange(&req->stage,1);
        post_status(req,L"Consulting Atlas's 1.5B reference",0);
        wchar_t *notice=utf8_to_wide("\r\n[Low confidence; checking Atlas's 1.5B reference model...]\r\n");
        if (notice && !PostMessageW(req->window,WM_APPEND_TEXT,0,(LPARAM)notice)) free(notice);
        snprintf(command,n*2+512,"atlas_teacher_infer.exe %s %s %u %u --confidence",
                 qteacher,qprompt,req->max_tokens,req->min_tokens);
        DWORD teacher_exit=1;
        char *teacher=run_child_capture(req,command,&teacher_exit);
        int teacher_truncated=0;
        (void)remove_confidence_marker(teacher,&teacher_truncated);
        if (teacher_exit==0 && teacher && *teacher) {
            free(response); response=teacher;
            if (!teacher_truncated) save_reference(question_key,response);
        } else free(teacher);
    }
#endif
    }
    if (response && !from_reference && message_requests_implementation(question) &&
        !response_satisfies_code_request(question,response)) {
#if !defined(ATLAS_COMPACT)
        if (req->route_mode!=2) {
            post_status(req,L"The direct model omitted source code; asking the specialists",0);
            char *chain=run_chained_request(question,req->allow_reference);
            if (chain && response_satisfies_code_request(question,chain)) { free(response); response=chain; }
            else {
                free(chain); free(response);
                response=_strdup("[UNVERIFIED: The models did not produce complete source code satisfying the stated constraints. No implementation is presented as correct. See the chain monitor for specialist drafts and verification.]");
            }
        } else {
            free(response);
            response=_strdup("[UNVERIFIED: The selected single model did not produce source code for this implementation request. No implementation is being presented as correct.]");
        }
#else
        free(response);
        response=_strdup("[UNVERIFIED: The model did not produce source code for this implementation request.]");
#endif
    }
    if (response && !from_reference) {
        wchar_t *part=utf8_to_wide(response);
        if (part && !PostMessageW(req->window,WM_APPEND_TEXT,0,(LPARAM)part)) free(part);
    }
    if (!response || !*response) {
        free(response);
        response=message_requests_implementation(question)
            ? _strdup("[UNVERIFIED: The local models returned no usable implementation. Try a compatible coding specialist or a larger model.]")
            : _strdup("[The local model returned no text. Check the model setup and inference log, then try again.]");
        if(response){wchar_t *notice=utf8_to_wide(response);if(notice&&!PostMessageW(req->window,WM_APPEND_TEXT,0,(LPARAM)notice))free(notice);}
    }
    if (exit_code!=0 && exit_code!=124 && response) {
        size_t a=strlen(response),b=strlen("\r\n[Inference failed.]\r\n");
        char *grown=realloc(response,a+b+1);
        if (grown) { response=grown; memcpy(response+a,"\r\n[Inference failed.]\r\n",b+1); }
    }
    free(model); free(qmodel); free(qprompt); free(command);
#if !defined(ATLAS_COMPACT)
    free(language_model); free(teacher_model); free(qlanguage); free(qteacher);
#endif
    free(question);
    free(question_key);
    {
        Response *result = calloc(1, sizeof(*result));
        if (result) {
            result->context = req->prompt_utf8;
            result->text = response ? response : _strdup("\r\n[Could not allocate inference output.]\r\n");
            PostMessageW(req->window, WM_REQUEST_DONE, 0, (LPARAM)result);
        } else { free(response); free(req->prompt_utf8); PostMessageW(req->window, WM_REQUEST_DONE, 0, 0); }
        free(req);
    }
    return 0;
failed:
    free(model); free(qmodel); free(qprompt); free(command);
#if !defined(ATLAS_COMPACT)
    free(language_model); free(teacher_model); free(qlanguage); free(qteacher);
#endif
    free(question);
    free(question_key);
    {
        Response *result = calloc(1, sizeof(*result));
        if (result) {
            result->context = req->prompt_utf8;
            result->text = _strdup("\r\n[Could not launch inference worker.]\r\n");
            PostMessageW(req->window, WM_REQUEST_DONE, 0, (LPARAM)result);
        } else { free(req->prompt_utf8); PostMessageW(req->window, WM_REQUEST_DONE, 0, 0); }
        free(req);
    }
    return 1;
}

static void start_request(HWND window)
{
    int chars = GetWindowTextLengthW(prompt_edit);
    if (chars <= 0) { MessageBoxW(window, L"Type a message first.", L"Empty message", MB_OK | MB_ICONINFORMATION); return; }
    wchar_t *wide = malloc(((size_t)chars + 1) * sizeof(wchar_t));
    if (!wide) return;
    GetWindowTextW(prompt_edit, wide, chars + 1);
    char *message = wide_to_utf8(wide); free(wide);
    if (!message) return;
    size_t oldlen = conversation ? strlen(conversation) : 0;
    size_t newlen = oldlen + strlen(message) + 100;
    char *utf8 = malloc(newlen);
    if (!utf8) { free(message); return; }
    if (oldlen) memcpy(utf8, conversation, oldlen); else utf8[0] = 0;
    snprintf(utf8 + oldlen, newlen - oldlen,
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", message);
    int code_model=1;
#if !defined(ATLAS_COMPACT)
    code_model=model_preference==2 || (model_preference==0 && message_needs_code_model(message));
#endif
    wchar_t *echo = utf8_to_wide(message);
    append_wide(L"\r\nYou: "); if (echo) { append_wide(echo); free(echo); }
#if defined(ATLAS_COMPACT)
    append_wide(L"\r\nAtlas LLM 1.0 (Coder 100%): ");
#else
    append_wide(code_model
        ? L"\r\nAtlas LLM 1.0 (Coder 100%, Language 0%): "
        : L"\r\nAtlas LLM 1.0 (Language 100%, Coder 0%): ");
#endif
    append_wide(L"\r\n");
    SetWindowTextW(prompt_edit, L"");
    EnableWindow(send_button, FALSE);
    SetWindowTextW(status_label,L"Starting Atlas's local model...");
    Request *req = malloc(sizeof(*req));
    if (!req) { free(utf8); EnableWindow(send_button, TRUE); return; }
    req->window = window; req->prompt_utf8 = utf8;
    req->route_mode=route_mode; req->model_preference=model_preference;
    req->allow_reference=allow_reference; req->code_model=code_model;
    req->deadline_tick=GetTickCount64()+300000ULL;
    req->stage=0;
    req->min_tokens=code_model ? 0 : min_reply_tokens(message);
    req->max_tokens=code_model ? 512 :
        (message_needs_decomposition(message) ? 256 : 64);
    free(message);
    char *pending=_strdup(utf8);
    if (pending) { free(conversation); conversation=pending; save_memory(); }
    HANDLE thread = CreateThread(NULL, 0, inference_worker, req, 0, NULL);
    if (!thread) { free(utf8); free(req); EnableWindow(send_button, TRUE); append_wide(L"Could not start inference worker.\r\n"); }
    else CloseHandle(thread);
}

static void persist_settings(void)
{
    if (!settings_path[0]) return;
    wchar_t value[16];
    _snwprintf(value,15,L"%d",route_mode); value[15]=0;
    WritePrivateProfileStringW(L"General",L"RouteMode",value,settings_path);
    _snwprintf(value,15,L"%d",model_preference); value[15]=0;
    WritePrivateProfileStringW(L"General",L"ModelPreference",value,settings_path);
    _snwprintf(value,15,L"%d",allow_reference); value[15]=0;
    WritePrivateProfileStringW(L"General",L"AllowReference",value,settings_path);
}

static void open_models(int specialists)
{
    ShellExecuteW(NULL,L"open",specialists?L"model\\specialists":L"model",NULL,NULL,SW_SHOWNORMAL);
}

static HMENU make_menu(void)
{
    HMENU root=CreateMenu(),file=CreatePopupMenu(),options=CreatePopupMenu(),view=CreatePopupMenu(),help=CreatePopupMenu();
    AppendMenuW(file,MF_STRING,ID_MENU_CLEAR,L"Clear conversation...");
    AppendMenuW(file,MF_STRING,ID_MENU_NEW_CHAT,L"New chat");
    recent_chats_menu=CreatePopupMenu();
    AppendMenuW(file,MF_POPUP,(UINT_PTR)recent_chats_menu,L"Recent chats");
    AppendMenuW(file,MF_SEPARATOR,0,NULL);
    AppendMenuW(file,MF_STRING,ID_MENU_MODELS,L"Open model folder");
    AppendMenuW(file,MF_STRING,ID_MENU_SPECIALISTS,L"Open specialist folder");
    AppendMenuW(file,MF_SEPARATOR,0,NULL); AppendMenuW(file,MF_STRING,ID_MENU_ABOUT,L"Model setup guide");
    AppendMenuW(options,MF_STRING,ID_MENU_ROUTE_AUTO,L"Route: Automatic");
    AppendMenuW(options,MF_STRING,ID_MENU_ROUTE_CHAIN,L"Route: Always chain specialists");
    AppendMenuW(options,MF_STRING,ID_MENU_ROUTE_SINGLE,L"Route: Single model");
    AppendMenuW(options,MF_SEPARATOR,0,NULL);
    AppendMenuW(options,MF_STRING,ID_MENU_MODEL_AUTO,L"Model: Automatic");
    AppendMenuW(options,MF_STRING,ID_MENU_MODEL_LANGUAGE,L"Model: Conversation");
    AppendMenuW(options,MF_STRING,ID_MENU_MODEL_CODER,L"Model: Coding");
    AppendMenuW(options,MF_SEPARATOR,0,NULL); AppendMenuW(options,MF_STRING,ID_MENU_REFERENCE,L"Allow 1.5B reference model");
    AppendMenuW(view,MF_STRING,ID_MENU_MONITOR,L"Show model conversation");
    AppendMenuW(help,MF_STRING,ID_MENU_ABOUT,L"About and model setup");
    AppendMenuW(root,MF_POPUP,(UINT_PTR)file,L"File"); AppendMenuW(root,MF_POPUP,(UINT_PTR)options,L"Options");
    AppendMenuW(root,MF_POPUP,(UINT_PTR)view,L"View"); AppendMenuW(root,MF_POPUP,(UINT_PTR)help,L"Help");
    settings_menu=options;
    return root;
}

static LRESULT CALLBACK window_proc(HWND window, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg) {
    case WM_CREATE:
        route_mode=GetPrivateProfileIntW(L"General",L"RouteMode",0,settings_path);
        model_preference=GetPrivateProfileIntW(L"General",L"ModelPreference",0,settings_path);
        allow_reference=GetPrivateProfileIntW(L"General",L"AllowReference",1,settings_path);
        if (route_mode<0||route_mode>2) route_mode=0;
        if (model_preference<0||model_preference>2) model_preference=0;
        search_all_checkbox=CreateWindowW(L"BUTTON",L"All chats",WS_CHILD|WS_VISIBLE|BS_AUTOCHECKBOX,
                                          16,12,88,30,window,(HMENU)ID_SEARCH_ALL,NULL,NULL);
        search_edit = CreateWindowExW(WS_EX_CLIENTEDGE,L"EDIT",L"",WS_CHILD|WS_VISIBLE|ES_AUTOHSCROLL,
                                      112,12,560,28,window,(HMENU)ID_SEARCH,NULL,NULL);
        send_button = CreateWindowW(L"BUTTON", L"Find", WS_CHILD | WS_VISIBLE,
                                    680, 42, 100, 32, window, (HMENU)ID_FIND, NULL, NULL);
        prompt_edit = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"", WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_AUTOVSCROLL | WS_VSCROLL,
                                      16, 80, 650, 74, window, (HMENU)ID_PROMPT, NULL, NULL);
        send_button = CreateWindowW(L"BUTTON", L"Send", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
                                    680, 80, 100, 32, window, (HMENU)ID_SEND, NULL, NULL);
        status_label=CreateWindowW(L"STATIC",L"Ready",WS_CHILD|WS_VISIBLE,16,158,764,22,window,(HMENU)ID_STATUS,NULL,NULL);
        output_box = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT",
                      MODEL_LABEL L"\r\nLocal C inference. Complex requests can take up to five minutes; progress appears above.\r\n",
                      WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL,
                      16, 184, 764, 368, window, (HMENU)ID_LOG, NULL, NULL);
        ui_font=CreateFontW(-18,0,0,0,FW_NORMAL,FALSE,FALSE,FALSE,DEFAULT_CHARSET,
                            OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,
                            DEFAULT_PITCH|FF_DONTCARE,L"Segoe UI");
        apply_ui_font(search_edit); apply_ui_font(prompt_edit); apply_ui_font(output_box);apply_ui_font(search_all_checkbox);
        apply_ui_font(status_label);
        HWND find_button=GetDlgItem(window,ID_FIND); apply_ui_font(find_button);
        apply_ui_font(send_button);
        return 0;
    case WM_SIZE:
        if (output_box) {
            int w = LOWORD(lp), h = HIWORD(lp);
            if (w<560) w=560; if (h<420) h=420;
            MoveWindow(search_edit,112,12,w-220,28,TRUE);
            MoveWindow(GetDlgItem(window,ID_FIND),w-100,10,84,32,TRUE);
            MoveWindow(output_box,16,52,w-32,h-244,TRUE);
            MoveWindow(status_label,16,h-182,w-32,24,TRUE);
            MoveWindow(prompt_edit,16,h-150,w-148,100,TRUE);
            MoveWindow(send_button,w-116,h-148,100,38,TRUE);
        }
        return 0;
    case WM_GETMINMAXINFO:
        ((MINMAXINFO*)lp)->ptMinTrackSize.x=600;
        ((MINMAXINFO*)lp)->ptMinTrackSize.y=460;
        return 0;
    case WM_COMMAND:
        if(LOWORD(wp)==ID_MENU_NEW_CHAT){create_new_chat(window);return 0;}
        if(LOWORD(wp)>=ID_CHAT_BASE&&LOWORD(wp)<ID_CHAT_BASE+MAX_RECENT_CHATS){
            size_t index=(size_t)(LOWORD(wp)-ID_CHAT_BASE);
            if(index<recent_chat_count)switch_chat(window,recent_chat_paths[index]);
            return 0;
        }
        switch (LOWORD(wp)) {
        case ID_MENU_CLEAR:
            if (MessageBoxW(window,L"Clear the saved conversation?",L"Clear conversation",MB_YESNO|MB_ICONQUESTION)==IDYES) {
                free(conversation); conversation=_strdup(system_message); save_memory();
                SetWindowTextW(output_box,MODEL_LABEL L"\r\nLocal inference is ready. Use Options to choose routing and model preferences.\r\n");
                SetWindowTextW(status_label,L"Conversation cleared.");
            } return 0;
        case ID_MENU_MODELS: open_models(0); return 0;
        case ID_MENU_SPECIALISTS: open_models(1); return 0;
        case ID_MENU_MONITOR: start_chain_monitor(); return 0;
        case ID_MENU_ABOUT:
            MessageBoxW(window,L"Atlas is a local C inference and specialist-routing project. Model folders contain separately licensed weights. GPT-level capability depends on the selected model; equal parameter count does not imply equal results. See README.md and MODEL_SETUP.md for supported models and setup.",L"Model setup",MB_OK|MB_ICONINFORMATION); return 0;
        case ID_MENU_ROUTE_AUTO: case ID_MENU_ROUTE_CHAIN: case ID_MENU_ROUTE_SINGLE:
            route_mode=LOWORD(wp)==ID_MENU_ROUTE_AUTO?0:(LOWORD(wp)==ID_MENU_ROUTE_CHAIN?1:2);
            CheckMenuRadioItem(settings_menu,ID_MENU_ROUTE_AUTO,ID_MENU_ROUTE_SINGLE,LOWORD(wp),MF_BYCOMMAND); persist_settings(); return 0;
        case ID_MENU_MODEL_AUTO: case ID_MENU_MODEL_LANGUAGE: case ID_MENU_MODEL_CODER:
            model_preference=LOWORD(wp)==ID_MENU_MODEL_AUTO?0:(LOWORD(wp)==ID_MENU_MODEL_LANGUAGE?1:2);
            CheckMenuRadioItem(settings_menu,ID_MENU_MODEL_AUTO,ID_MENU_MODEL_CODER,LOWORD(wp),MF_BYCOMMAND); persist_settings(); return 0;
        case ID_MENU_REFERENCE:
            allow_reference=!allow_reference; CheckMenuItem(settings_menu,ID_MENU_REFERENCE,MF_BYCOMMAND|(allow_reference?MF_CHECKED:MF_UNCHECKED)); persist_settings(); return 0;
        }
        if (LOWORD(wp) == ID_SEND) { start_request(window); return 0; }
        if (LOWORD(wp) == ID_FIND) {
            search_edit=GetDlgItem(window,ID_SEARCH);
            search_all_checkbox=GetDlgItem(window,ID_SEARCH_ALL);
            find_in_chat(window); return 0;
        }
        if (LOWORD(wp)==ID_SEARCH && HIWORD(wp)==EN_SETFOCUS) return 0;
        break;
    case WM_KEYDOWN:
        if (wp=='F' && (GetKeyState(VK_CONTROL)&0x8000)) { SetFocus(search_edit); return 0; }
        break;
    case WM_INITMENUPOPUP:
        if((HMENU)wp==recent_chats_menu)refresh_recent_chats();
        return 0;
    case WM_INFERENCE_PROGRESS: {
        wchar_t *status=(wchar_t*)lp;
        if (status) { SetWindowTextW(status_label,status); free(status); }
        return 0;
    }
    case WM_APPEND_TEXT: {
        wchar_t *wide = (wchar_t *)lp;
        if (wide) { append_wide(wide); free(wide); }
        return 0;
    }
    case WM_REQUEST_DONE: {
        Response *result = (Response *)lp;
        if (result) {
            const char *message=result->text ? result->text : "";
            const char *error=strstr(message,"[Inference failed.");
            if (!error) error=strstr(message,"Could not start " INFER_EXE);
            if (!error) error=strstr(message,"Could not launch inference worker.");
            if (error) {
                wchar_t *notice=utf8_to_wide(error);
                if (notice) { append_wide(notice); free(notice); }
            }
            size_t a = strlen(result->context ? result->context : "");
            size_t b = strlen(result->text ? result->text : "");
            static const char turn_end[] = "<|im_end|>\n";
            char *next = malloc(a + b + sizeof(turn_end));
            if (next) {
                memcpy(next, result->context, a);
                memcpy(next + a, result->text, b);
                memcpy(next + a + b, turn_end, sizeof(turn_end));
                free(conversation); conversation = next;
                size_t len = strlen(conversation);
        if (len > 10000) {
            const char *cut = conversation + len - 7000;
            const char *boundary = strstr(cut, "<|im_start|>user\n");
            if (!boundary) boundary = strstr(cut, "<|im_start|>");
            if (boundary) {
                        const char *system = system_message;
                        size_t prefix = strlen(system);
                        size_t tail = strlen(boundary);
                        char *trimmed = malloc(prefix + tail + 1);
                        if (trimmed) { memcpy(trimmed, system, prefix); memcpy(trimmed + prefix, boundary, tail + 1); free(conversation); conversation = trimmed; }
                    }
                }
                save_memory();
            }
            free(result->context); free(result->text); free(result);
        }
        EnableWindow(send_button, TRUE);
        SetWindowTextW(status_label,L"Ready");
        return 0;
    }
    case WM_DESTROY: save_memory(); PostQuitMessage(0); return 0;
    case WM_DESTROY + 100: return 0;
    }
    return DefWindowProcW(window, msg, wp, lp);
}

int WINAPI WinMain(HINSTANCE instance, HINSTANCE previous, LPSTR cmdline, int show)
{
    (void)previous;
    start_chain_monitor();
    init_memory_path();
    if (settings_path[0]) {
        route_mode=GetPrivateProfileIntW(L"General",L"RouteMode",0,settings_path);
        model_preference=GetPrivateProfileIntW(L"General",L"ModelPreference",0,settings_path);
        allow_reference=GetPrivateProfileIntW(L"General",L"AllowReference",1,settings_path);
    }
    conversation=load_memory();
    if (conversation) {
        const char *old_system_end=strstr(conversation,"<|im_end|>\n");
        if (old_system_end && !strncmp(conversation,"<|im_start|>system\n",19)) {
            const char *history=old_system_end+sizeof("<|im_end|>\n")-1;
            size_t prefix=strlen(system_message), tail=strlen(history);
            char *updated=malloc(prefix+tail+1);
            if (updated) {
                memcpy(updated,system_message,prefix);
                memcpy(updated+prefix,history,tail+1);
                free(conversation); conversation=updated;
                save_memory();
            }
        }
        conversation_restored=1;
    } else conversation=_strdup(system_message);
    WNDCLASSW wc = {0}; wc.lpfnWndProc = window_proc; wc.hInstance = instance;
    wc.lpszClassName = L"AtlasChatWindow"; wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    if (!RegisterClassW(&wc)) return 1;
    HMENU menu=make_menu();
    HWND window = CreateWindowW(wc.lpszClassName, WINDOW_TITLE, WS_OVERLAPPEDWINDOW|WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, 980, 760, NULL, menu, instance, NULL);
    if (!window) return 1;
    if (conversation_restored) render_active_chat(window);
    ShowWindow(window, show);
    CheckMenuRadioItem(settings_menu,ID_MENU_ROUTE_AUTO,ID_MENU_ROUTE_SINGLE,
        route_mode==1?ID_MENU_ROUTE_CHAIN:(route_mode==2?ID_MENU_ROUTE_SINGLE:ID_MENU_ROUTE_AUTO),MF_BYCOMMAND);
    CheckMenuRadioItem(settings_menu,ID_MENU_MODEL_AUTO,ID_MENU_MODEL_CODER,
        model_preference==1?ID_MENU_MODEL_LANGUAGE:(model_preference==2?ID_MENU_MODEL_CODER:ID_MENU_MODEL_AUTO),MF_BYCOMMAND);
    CheckMenuItem(settings_menu,ID_MENU_REFERENCE,MF_BYCOMMAND|(allow_reference?MF_CHECKED:MF_UNCHECKED));
    /* Optional same-GUI automation entry point for repeatable local trials.
       It enters through the normal Send command after the window is shown. */
    if (cmdline && !strncmp(cmdline, "--ask=", 6)) {
        char *argument = cmdline + 6;
        size_t length = strlen(argument);
        if (length >= 2 && argument[0] == '"' && argument[length - 1] == '"') {
            argument[length - 1] = 0;
            ++argument;
        }
        wchar_t *initial_prompt = utf8_to_wide(argument);
        if (initial_prompt) {
            SetWindowTextW(prompt_edit, initial_prompt);
            free(initial_prompt);
            PostMessageW(window, WM_COMMAND, MAKEWPARAM(ID_SEND, BN_CLICKED),
                         (LPARAM)GetDlgItem(window, ID_SEND));
        }
    }
#if defined(ATLAS_UI_DESIGN_TEST)
    ShowWindow(window,SW_HIDE);
    SetWindowTextW(prompt_edit,
        L"Atlas, design a better user interface for yourself. Break the interface proposal into a few small parts and decide which changes give the most usability benefit for the least processing and implementation work. The current app is a native Win32 chat window. Propose a practical prioritized design that improves readability, conversation navigation, composing and sending, response status, and model/reference visibility. Describe the layout, controls, and keyboard behavior. Keep it realistic to implement in plain C Win32, concise enough that we can implement it, and do not write source code yet.");
    PostMessageW(window,WM_COMMAND,ID_SEND,0);
#endif
    MSG msg; while (GetMessageW(&msg, NULL, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    return (int)msg.wParam;
}
