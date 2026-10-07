#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint8_t d[2];
    uint8_t dmin[2];
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

static uint8_t read_u8(FILE *f)
{
    int c=fgetc(f);
    if(c==EOF) exit(1);
    return (uint8_t)c;
}

static uint16_t read_u16(FILE *f)
{
    uint8_t b[2];
    if(fread(b,1,2,f)!=2) exit(1);
    return (uint16_t)b[0] | ((uint16_t)b[1]<<8);
}

static uint32_t read_u32(FILE *f)
{
    uint8_t b[4];
    if(fread(b,1,4,f)!=4) exit(1);

    return (uint32_t)b[0] |
           ((uint32_t)b[1]<<8) |
           ((uint32_t)b[2]<<16) |
           ((uint32_t)b[3]<<24);
}

static uint64_t read_u64(FILE *f)
{
    uint8_t b[8];

    if(fread(b,1,8,f)!=8) exit(1);

    return (uint64_t)b[0] |
           ((uint64_t)b[1]<<8) |
           ((uint64_t)b[2]<<16) |
           ((uint64_t)b[3]<<24) |
           ((uint64_t)b[4]<<32) |
           ((uint64_t)b[5]<<40) |
           ((uint64_t)b[6]<<48) |
           ((uint64_t)b[7]<<56);
}

static uint16_t mem_u16(const uint8_t *p)
{
    return (uint16_t)p[0] |
           ((uint16_t)p[1]<<8);
}

static char *read_string(FILE *f)
{
    uint64_t n=read_u64(f);

    if(n>10000000ULL) exit(1);

    char *s=malloc((size_t)n+1);
    if(!s) exit(1);

    if(fread(s,1,(size_t)n,f)!=(size_t)n)
        exit(1);

    s[n]=0;
    return s;
}

static void skip_value(FILE *f,uint32_t type);

static void skip_array(FILE *f)
{
    uint32_t type=read_u32(f);
    uint64_t count=read_u64(f);

    for(uint64_t i=0;i<count;i++)
        skip_value(f,type);
}

static void skip_value(FILE *f,uint32_t type)
{
    switch(type)
    {
        case 0:
        case 1:
        case 7:
            read_u8(f);
            break;

        case 2:
        case 3:
            read_u16(f);
            break;

        case 4:
        case 5:
        case 6:
            read_u32(f);
            break;

        case 8:
        {
            char *s=read_string(f);
            free(s);
            break;
        }

        case 9:
            skip_array(f);
            break;

        case 10:
        case 11:
        case 12:
            read_u64(f);
            break;

        default:
            fprintf(stderr,"bad metadata type %u\n",type);
            exit(1);
    }
}

static uint64_t align_to(uint64_t x,uint64_t a)
{
    return ((x+a-1)/a)*a;
}

static float fp16_to_fp32(uint16_t h)
{
    uint32_t sign=(uint32_t)(h&0x8000)<<16;
    uint32_t exp=(h>>10)&0x1F;
    uint32_t mant=h&0x3FF;
    uint32_t out;

    if(exp==0)
    {
        if(mant==0)
        {
            out=sign;
        }
        else
        {
            exp=1;

            while((mant&0x400)==0)
            {
                mant<<=1;
                exp--;
            }

            mant&=0x3FF;

            out=sign |
                ((exp+112)<<23) |
                (mant<<13);
        }
    }
    else if(exp==31)
    {
        out=sign|0x7F800000|(mant<<13);
    }
    else
    {
        out=sign |
            ((exp+112)<<23) |
            (mant<<13);
    }

    float v;
    memcpy(&v,&out,sizeof(v));
    return v;
}

static void get_scale_min(
    const block_q4_K *b,
    int group,
    int *scale,
    int *min)
{
    if(group<4)
    {
        *scale=b->scales[group]&0x3F;
        *min=b->scales[group+4]&0x3F;
    }
    else
    {
        *scale=
            (b->scales[group+4]&0x0F) |
            ((b->scales[group-4]>>6)<<4);

        *min=
            (b->scales[group+4]>>4) |
            ((b->scales[group]>>6)<<4);
    }
}

static void dequant_q4_k(
    const uint8_t *src,
    float *dst)
{
    const block_q4_K *b=
        (const block_q4_K *)src;

    float d=fp16_to_fp32(mem_u16(b->d));
    float dmin=fp16_to_fp32(mem_u16(b->dmin));

    for(int group=0;group<8;group++)
    {
        int scale;
        int min;

        get_scale_min(
            b,
            group,
            &scale,
            &min);

        float ds=d*(float)scale;
        float dm=dmin*(float)min;

        const uint8_t *q=
            b->qs+group*16;

        for(int i=0;i<16;i++)
        {
            dst[group*32+i]=
                ds*(float)(q[i]&0x0F)-dm;

            dst[group*32+i+16]=
                ds*(float)(q[i]>>4)-dm;
        }
    }
}

int main(void)
{
    const char *path=
        "model/atlas-1.0.gguf";

    FILE *f=fopen(path,"rb");

    if(!f)
    {
        perror(path);
        return 1;
    }

    uint32_t magic=read_u32(f);
    uint32_t version=read_u32(f);
    uint64_t tensor_count=read_u64(f);
    uint64_t metadata_count=read_u64(f);

    if(magic!=0x46554747 || version!=3)
    {
        printf("INVALID GGUF\n");
        fclose(f);
        return 1;
    }

    uint64_t alignment=32;

    for(uint64_t i=0;i<metadata_count;i++)
    {
        char *key=read_string(f);
        uint32_t type=read_u32(f);

        if(strcmp(key,"general.alignment")==0 && type==4)
            alignment=read_u32(f);
        else
            skip_value(f,type);

        free(key);
    }

    uint64_t q4_count=0;
    uint64_t first_offset=0;
    uint64_t first_elements=0;
    char first_name[512]={0};

    for(uint64_t i=0;i<tensor_count;i++)
    {
        char *name=read_string(f);
        uint32_t dims=read_u32(f);

        uint64_t elements=1;

        for(uint32_t d=0;d<dims;d++)
            elements*=read_u64(f);

        uint32_t type=read_u32(f);
        uint64_t relative=read_u64(f);

        if(type==12)
        {
            q4_count++;

            if(q4_count==1)
            {
                first_offset=relative;
                first_elements=elements;

                strncpy(
                    first_name,
                    name,
                    sizeof(first_name)-1);
            }
        }

        free(name);
    }

    uint64_t directory_end=(uint64_t)ftell(f);
    uint64_t data_start=align_to(directory_end,alignment);

    printf("========================================\n");
    printf("REAL Q4_K TENSOR TEST\n");
    printf("========================================\n");

    printf("Tensor count:     %llu\n",
        (unsigned long long)tensor_count);

    printf("Q4_K tensors:     %llu\n",
        (unsigned long long)q4_count);

    printf("Directory end:    %llu\n",
        (unsigned long long)directory_end);

    printf("Data start:       %llu\n",
        (unsigned long long)data_start);

    if(q4_count==0)
    {
        printf("\nNO Q4_K TENSOR FOUND\n");
        fclose(f);
        return 1;
    }

    uint64_t absolute=data_start+first_offset;

    printf("\nFirst Q4_K tensor:\n");
    printf("Name:             %s\n",first_name);
    printf("Elements:         %llu\n",
        (unsigned long long)first_elements);
    printf("Relative offset:  %llu\n",
        (unsigned long long)first_offset);
    printf("Absolute offset:  %llu\n",
        (unsigned long long)absolute);

    if(fseek(f,(long)absolute,SEEK_SET)!=0)
    {
        printf("SEEK FAILED\n");
        fclose(f);
        return 1;
    }

    uint8_t block[144];

    if(fread(block,1,sizeof(block),f)!=sizeof(block))
    {
        printf("BLOCK READ FAILED\n");
        fclose(f);
        return 1;
    }

    float values[256];

    dequant_q4_k(block,values);

    float min=values[0];
    float max=values[0];
    double sum=0.0;
    int finite=1;

    for(int i=0;i<256;i++)
    {
        if(!isfinite(values[i]))
            finite=0;

        if(values[i]<min)
            min=values[i];

        if(values[i]>max)
            max=values[i];

        sum+=values[i];
    }

    printf("\nFirst Q4_K block:\n");
    printf("Bytes:            144\n");
    printf("Values:           256\n");

    printf("d:                %.9g\n",
        fp16_to_fp32(mem_u16(block)));

    printf("dmin:             %.9g\n",
        fp16_to_fp32(mem_u16(block+2)));

    printf("Min:              %.9f\n",min);
    printf("Max:              %.9f\n",max);
    printf("Mean:             %.9f\n",
        (float)(sum/256.0));

    printf("\nFirst 32 values:\n");

    for(int i=0;i<32;i++)
    {
        printf("% .8f%s",
            values[i],
            i%8==7?"\n":" ");
    }

    printf("\n========================================\n");

    if(finite && min!=max)
        printf("REAL Q4_K DEQUANTIZATION: PASS\n");
    else
        printf("REAL Q4_K DEQUANTIZATION: FAIL\n");

    printf("========================================\n");

    fclose(f);
    return finite ? 0 : 1;
}
