#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint8_t u8(FILE *f){int c=fgetc(f);if(c==EOF)exit(1);return(uint8_t)c;}
static uint16_t u16(FILE *f){uint8_t b[2];if(fread(b,1,2,f)!=2)exit(1);return(uint16_t)b[0]|((uint16_t)b[1]<<8);}
static uint32_t u32(FILE *f){uint8_t b[4];if(fread(b,1,4,f)!=4)exit(1);return(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);}
static uint64_t u64(FILE *f){uint8_t b[8];if(fread(b,1,8,f)!=8)exit(1);return(uint64_t)b[0]|((uint64_t)b[1]<<8)|((uint64_t)b[2]<<16)|((uint64_t)b[3]<<24)|((uint64_t)b[4]<<32)|((uint64_t)b[5]<<40)|((uint64_t)b[6]<<48)|((uint64_t)b[7]<<56);}

static char *str(FILE *f){
    uint64_t n=u64(f);
    if(n>10000000ULL){fprintf(stderr,"bad string length\n");exit(1);}
    char *s=malloc((size_t)n+1);
    if(!s)exit(1);
    if(fread(s,1,(size_t)n,f)!=(size_t)n)exit(1);
    s[n]=0;
    return s;
}

static void skip_value(FILE *f,uint32_t t);

static void skip_array(FILE *f){
    uint32_t t=u32(f);
    uint64_t n=u64(f);
    for(uint64_t i=0;i<n;i++)skip_value(f,t);
}

static void skip_value(FILE *f,uint32_t t){
    switch(t){
        case 0:case 1:case 7:u8(f);break;
        case 2:case 3:u16(f);break;
        case 4:case 5:case 6:u32(f);break;
        case 8:{char *s=str(f);free(s);break;}
        case 9:skip_array(f);break;
        case 10:case 11:case 12:u64(f);break;
        default:fprintf(stderr,"bad metadata type %u\n",t);exit(1);
    }
}

static uint64_t align32(uint64_t x){
    return (x+31ULL)&~31ULL;
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

    printf("========================================\n");
    printf("GGUF DATA OFFSET TEST\n");
    printf("========================================\n");
    printf("Magic:        0x%08X\n",magic);
    printf("Version:      %u\n",version);
    printf("Tensors:      %llu\n",(unsigned long long)tensors);
    printf("Metadata:     %llu\n",(unsigned long long)metadata);

    for(uint64_t i=0;i<metadata;i++){
        char *key=str(f);
        uint32_t type=u32(f);
        skip_value(f,type);
        free(key);
    }

    uint64_t directory_end=(uint64_t)ftell(f);
    uint64_t data_start=align32(directory_end);

    printf("\nTensor directory end: %llu\n",
           (unsigned long long)directory_end);
    printf("Aligned data start:   %llu\n",
           (unsigned long long)data_start);

    int found=0;

    for(uint64_t i=0;i<tensors;i++){
        char *name=str(f);
        uint32_t dims_n=u32(f);

        uint64_t elements=1;
        for(uint32_t d=0;d<dims_n;d++)
            elements*=u64(f);

        uint32_t type=u32(f);
        uint64_t relative=u64(f);

        if(strcmp(name,"output.weight")==0){
            uint64_t absolute=data_start+relative;

            printf("\nTARGET TENSOR FOUND\n");
            printf("Name:              %s\n",name);
            printf("Elements:          %llu\n",
                   (unsigned long long)elements);
            printf("GGUF type:         %u\n",type);
            printf("Relative offset:   %llu\n",
                   (unsigned long long)relative);
            printf("Absolute offset:   %llu\n",
                   (unsigned long long)absolute);

            printf("\nKnown test offset: 5950528\n");
            printf("Offset difference: %lld\n",
                   (long long)absolute-5950528LL);

            if(absolute==5950528ULL)
                printf("OFFSET MATCH: PASS\n");
            else
                printf("OFFSET MATCH: FAIL\n");

            found=1;
        }

        free(name);
    }

    printf("\n========================================\n");

    if(found)
        printf("GGUF DATA OFFSET: PASS\n");
    else
        printf("output.weight NOT FOUND\n");

    printf("========================================\n");

    fclose(f);
    return found?0:1;
}
