#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint8_t u8(FILE *f){int c=fgetc(f);if(c==EOF)exit(1);return(uint8_t)c;}
static uint16_t u16(FILE *f){uint8_t b[2];if(fread(b,1,2,f)!=2)exit(1);return(uint16_t)b[0]|((uint16_t)b[1]<<8);}
static uint32_t u32(FILE *f){uint8_t b[4];if(fread(b,1,4,f)!=4)exit(1);return(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);}
static uint64_t u64(FILE *f){uint8_t b[8];if(fread(b,1,8,f)!=8)exit(1);return(uint64_t)b[0]|((uint64_t)b[1]<<8)|((uint64_t)b[2]<<16)|((uint64_t)b[3]<<24)|((uint64_t)b[4]<<32)|((uint64_t)b[5]<<40)|((uint64_t)b[6]<<48)|((uint64_t)b[7]<<56);}

static char *read_string(FILE *f)
{
    uint64_t n=u64(f);
    if(n>10000000ULL)exit(1);
    char *s=malloc((size_t)n+1);
    if(!s)exit(1);
    if(fread(s,1,(size_t)n,f)!=(size_t)n)exit(1);
    s[n]=0;
    return s;
}

static void skip_value(FILE *f,uint32_t t);

static void skip_array(FILE *f)
{
    uint32_t t=u32(f);
    uint64_t n=u64(f);
    for(uint64_t i=0;i<n;i++)skip_value(f,t);
}

static void skip_value(FILE *f,uint32_t t)
{
    switch(t){
        case 0:case 1:case 7:u8(f);break;
        case 2:case 3:u16(f);break;
        case 4:case 5:case 6:u32(f);break;
        case 8:{char *s=read_string(f);free(s);break;}
        case 9:skip_array(f);break;
        case 10:case 11:case 12:u64(f);break;
        default:exit(1);
    }
}

static uint64_t align_to(uint64_t x,uint64_t alignment)
{
    return (x+alignment-1)/alignment*alignment;
}

int main(void)
{
    const char *path="model/atlas-1.0.gguf";
    FILE *f=fopen(path,"rb");

    if(!f){
        perror(path);
        return 1;
    }

    uint32_t magic=u32(f);
    uint32_t version=u32(f);
    uint64_t tensors=u64(f);
    uint64_t metadata=u64(f);

    if(magic!=0x46554747 || version!=3){
        printf("INVALID GGUF\n");
        fclose(f);
        return 1;
    }

    uint64_t alignment=32;

    for(uint64_t i=0;i<metadata;i++){
        char *key=read_string(f);
        uint32_t type=u32(f);

        if(strcmp(key,"general.alignment")==0){
            if(type!=4){
                printf("general.alignment has unexpected type %u\n",type);
                free(key);
                fclose(f);
                return 1;
            }

            alignment=u32(f);

            printf("general.alignment = %llu\n",
                   (unsigned long long)alignment);
        }else{
            skip_value(f,type);
        }

        free(key);
    }

    uint64_t directory_end=(uint64_t)ftell(f);
    uint64_t data_start=align_to(directory_end,alignment);

    printf("\n========================================\n");
    printf("GGUF ALIGNMENT TEST\n");
    printf("========================================\n");

    printf("Directory end: %llu\n",
           (unsigned long long)directory_end);

    printf("Alignment:     %llu\n",
           (unsigned long long)alignment);

    printf("Data start:    %llu\n",
           (unsigned long long)data_start);

    printf("Known offset:  5950528\n");

    printf("Difference:    %lld\n",
           (long long)data_start-5950528LL);

    if(data_start==5950528ULL)
        printf("\nDATA OFFSET MATCH: PASS\n");
    else
        printf("\nDATA OFFSET MATCH: FAIL\n");

    printf("========================================\n");

    fclose(f);
    return data_start==5950528ULL?0:1;
}
