#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>

typedef struct {
    char *path;
    uint8_t *data;
    size_t size;
} RamFile;

typedef struct {
    RamFile *files;
    size_t count;
    size_t capacity;
    size_t total_bytes;
} RamFS;

static void die(const char *msg)
{
    fprintf(stderr, "\nFATAL: %s\n", msg);
    exit(1);
}

static void ramfs_init(RamFS *fs)
{
    memset(fs, 0, sizeof(*fs));
}

static void ramfs_add(RamFS *fs, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return;
    }

    long len = ftell(f);
    if (len < 0) {
        fclose(f);
        return;
    }

    rewind(f);

    if (len == 0) {
        fclose(f);
        return;
    }

    uint8_t *data = (uint8_t *)malloc((size_t)len);
    if (!data) {
        fclose(f);
        die("RAM allocation failed");
    }

    size_t got = fread(data, 1, (size_t)len, f);
    fclose(f);

    if (got != (size_t)len) {
        free(data);
        return;
    }

    if (fs->count == fs->capacity) {
        size_t newcap = fs->capacity ? fs->capacity * 2 : 256;

        RamFile *newfiles =
            (RamFile *)realloc(fs->files, newcap * sizeof(RamFile));

        if (!newfiles) {
            free(data);
            die("RAM filesystem expansion failed");
        }

        fs->files = newfiles;
        fs->capacity = newcap;
    }

    fs->files[fs->count].path = _strdup(path);
    fs->files[fs->count].data = data;
    fs->files[fs->count].size = (size_t)len;

    fs->count++;
    fs->total_bytes += (size_t)len;

    printf("RAM  %8.2f MB  %s\n",
           (double)len / (1024.0 * 1024.0),
           path);
}

static int is_runtime_file(const char *name)
{
    const char *ext = strrchr(name, '.');

    if (!ext)
        return 0;

    return
        _stricmp(ext, ".dat") == 0 ||
        _stricmp(ext, ".tsv") == 0 ||
        _stricmp(ext, ".txt") == 0 ||
        _stricmp(ext, ".json") == 0 ||
        _stricmp(ext, ".bin") == 0 ||
        _stricmp(ext, ".gguf") == 0 ||
        _stricmp(ext, ".model") == 0 ||
        _stricmp(ext, ".vocab") == 0;
}

static void scan_directory(RamFS *fs, const char *dir)
{
    char pattern[MAX_PATH * 4];
    snprintf(pattern, sizeof(pattern), "%s\\*", dir);

    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);

    if (h == INVALID_HANDLE_VALUE)
        return;

    do {
        if (!strcmp(fd.cFileName, ".") ||
            !strcmp(fd.cFileName, ".."))
            continue;

        char path[MAX_PATH * 4];
        snprintf(path, sizeof(path), "%s\\%s", dir, fd.cFileName);

        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            scan_directory(fs, path);
        } else if (is_runtime_file(fd.cFileName)) {
            ramfs_add(fs, path);
        }

    } while (FindNextFileA(h, &fd));

    FindClose(h);
}

static void print_memory_status(void)
{
    MEMORYSTATUSEX ms;
    memset(&ms, 0, sizeof(ms));
    ms.dwLength = sizeof(ms);

    if (!GlobalMemoryStatusEx(&ms))
        return;

    printf("\nSYSTEM RAM\n");
    printf("==========\n");
    printf("Total RAM:     %.2f GB\n",
           (double)ms.ullTotalPhys / (1024.0 * 1024.0 * 1024.0));

    printf("Available RAM: %.2f GB\n",
           (double)ms.ullAvailPhys / (1024.0 * 1024.0 * 1024.0));

    printf("RAM used:      %lu%%\n", ms.dwMemoryLoad);
}

static void free_ramfs(RamFS *fs)
{
    for (size_t i = 0; i < fs->count; i++) {
        free(fs->files[i].path);
        free(fs->files[i].data);
    }

    free(fs->files);
}

int main(int argc, char **argv)
{
    printf("========================================\n");
    printf(" VM-AI WHOLE-RUNTIME RAM LOADER\n");
    printf("========================================\n\n");

    const char *root = ".";

    if (argc > 1)
        root = argv[1];

    RamFS fs;
    ramfs_init(&fs);

    print_memory_status();

    printf("\nLOADING VM-AI DATA INTO RAM\n");
    printf("============================\n");

    scan_directory(&fs, root);

    printf("\n========================================\n");
    printf(" RAM RESIDENCY COMPLETE\n");
    printf("========================================\n");

    printf("Files loaded:       %zu\n", fs.count);
    printf("RAM resident data:  %.2f GB\n",
           (double)fs.total_bytes /
           (1024.0 * 1024.0 * 1024.0));

    print_memory_status();

    printf("\nATLAS 1.0 MODEL STATUS\n");
    printf("=================\n");

    int found_model = 0;

    for (size_t i = 0; i < fs.count; i++) {
        const char *p = fs.files[i].path;

        if (strstr(p, ".gguf")) {
            printf("MODEL IN RAM:       %s\n", p);
            printf("MODEL SIZE:         %.2f GB\n",
                   (double)fs.files[i].size /
                   (1024.0 * 1024.0 * 1024.0));
            found_model = 1;
        }
    }

    if (!found_model)
        printf("WARNING: No GGUF model found.\n");

    printf("\nThe loaded buffers are now resident in this process.\n");
    printf("Press ENTER to release RAM and exit.\n");

    getchar();

    free_ramfs(&fs);

    return 0;
}
