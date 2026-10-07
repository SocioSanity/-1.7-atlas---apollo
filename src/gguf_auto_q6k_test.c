#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} block_q6_K;

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

    for(uint64_t i=0;i<n;i++)
        skip_value(f,t);
}

static void skip_value(FILE *f,uint32_t t)
{
    switch(t){
        case 0:case 1:case 7:u8(f);break;
        case 2:case 3:u16(f);break;
        case 4:case 5:case 6:u32(f);break;
        case 8:{
            char *s=read_string(f);
            free(s);
            break;
        }
        case 9:skip_array(f);break;
        case 10:case 11:case 12:u64(f);break;
        default:exit(1);
    }
}

static uint64_t align32(uint64_t x)
{
    return (x+31ULL)&~31ULL;
}

static float fp16_to_fp32(uint16_t h)
{
    uint32_t sign=(uint32_t)(h&0x8000)<<16;
    uint32_t exp=(h>>10)&0x1F;
    uint32_t mant=h&0x3FF;

    uint32_t out;

    if(exp==0){
        if(mant==0){
            out=sign;
        }else{
            exp=1;
            while((mant&0x400)==0){
                mant<<=1;
                exp--;
            }
            mant&=0x3FF;
            out=sign|((exp+112)<<23)|(mant<<13);
        }
    }else if(exp==31){
        out=sign|0x7F800000|(mant<<13);
    }else{
        out=sign|((exp+112)<<23)|(mant<<13);
    }

    float v;
    memcpy(&v,&out,sizeof(v));
    return v;
}

static void dequant_q6_k(const uint8_t *src,float *dst)
{
    const block_q6_K *b=(const block_q6_K *)src;

    float d=fp16_to_fp32(b->d);

    for(int l=0;l<16;l++){
        int8_t sc=b->scales[l];

        for(int j=0;j<16;j++){
            int idx=l*16+j;

            uint8_t ql=b->ql[idx>>1];
            int q=(idx&1)?(ql>>4):(ql&0x0F);

            uint8_t qh=b->qh[idx>>2];
            int high=(qh>>((idx&3)*2))&3;

            int q6=(q|(high<<4))-32;

            dst[idx]=d*(float)sc*(float)q6;
        }
    }
}

int main(void)
{
    const char *path=
        "model/atlas-1.0.gguf";

    FILE *f=fopen(path,"rb");

    if(!f){
        perror(path);
        return 1;
    }

    uint32_t magic=u32(f);
    uint32_t version=u32(f);
    uint64_t tensor_count=u64(f);
    uint64_t metadata_count=u64(f);

    if(magic!=0x46554747 || version!=3){
        printf("INVALID GGUF\n");
        fclose(f);
        return 1;
    }

    for(uint64_t i=0;i<metadata_count;i++){
        char *key=read_string(f);
        uint32_t type=u32(f);
        skip_value(f,type);
        free(key);
    }

    uint64_t directory_end=(uint64_t)ftell(f);
    uint64_t data_start=align32(directory_end);

    uint64_t output_relative=0;
    uint64_t output_elements=0;
    uint32_t output_type=0;
    int found=0;

    for(uint64_t i=0;i<tensor_count;i++){
        char *name=read_string(f);
        uint32_t dims_n=u32(f);

        uint64_t elements=1;

        for(uint32_t d=0;d<dims_n;d++)
            elements*=u64(f);

        uint32_t type=u32(f);
        uint64_t relative=u64(f);

        if(strcmp(name,"output.weight")==0){
            output_relative=relative;
            output_elements=elements;
            output_type=type;
            found=1;
        }

        free(name);
    }

    if(!found){
        printf("output.weight NOT FOUND\n");
        fclose(f);
        return 1;
    }

    uint64_t absolute=data_start+output_relative;

    printf("========================================\n");
    printf("AUTOMATIC Q6_K TENSOR LOAD TEST\n");
    printf("========================================\n");

    printf("Tensor:          output.weight\n");
    printf("Type:            %u\n",output_type);
    printf("Elements:        %llu\n",
           (unsigned long long)output_elements);
    printf("Data start:      %llu\n",
           (unsigned long long)data_start);
    printf("Relative offset: %llu\n",
           (unsigned long long)output_relative);
    printf("Absolute offset: %llu\n",
           (unsigned long long)absolute);

    if(output_type!=14){
        printf("EXPECTED Q6_K TYPE 14\n");
        fclose(f);
        return 1;
    }

    if(fseek(f,(long)absolute,SEEK_SET)!=0){
        printf("SEEK FAILED\n");
        fclose(f);
        return 1;
    }

    uint8_t raw[210];
    float values[256];

    if(fread(raw,1,sizeof(raw),f)!=sizeof(raw)){
        printf("BLOCK READ FAILED\n");
        fclose(f);
        return 1;
    }

    dequant_q6_k(raw,values);

    float min=values[0];
    float max=values[0];
    double sum=0.0;

    for(int i=0;i<256;i++){
        if(values[i]<min)min=values[i];
        if(values[i]>max)max=values[i];
        sum+=values[i];
    }

    printf("\nFirst Q6_K block:\n");
    printf("Bytes: 210\n");
    printf("Values: 256\n");
    printf("d: %.9g\n",fp16_to_fp32(((block_q6_K *)raw)->d));
    printf("Min: %.9f\n",min);
    printf("Max: %.9f\n",max);
    printf("Mean: %.9f\n",(float)(sum/256.0));

    printf("\n========================================\n");
    printf("AUTOMATIC Q6_K LOAD: PASS\n");
    printf("========================================\n");

    fclose(f);
    return 0;
}
