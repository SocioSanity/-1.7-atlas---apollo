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

static uint16_t mem_u16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float fp16_to_fp32(uint16_t h)
{
    uint32_t sign=(uint32_t)(h&0x8000)<<16;
    uint32_t exp=(h>>10)&0x1F;
    uint32_t mant=h&0x3FF;
    uint32_t out;

    if(exp==0)
    {
        if(mant==0) out=sign;
        else
        {
            exp=1;
            while((mant&0x400)==0)
            {
                mant<<=1;
                exp--;
            }
            mant&=0x3FF;
            out=sign|((exp+112)<<23)|(mant<<13);
        }
    }
    else if(exp==31)
        out=sign|0x7F800000|(mant<<13);
    else
        out=sign|((exp+112)<<23)|(mant<<13);

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
        *scale=(b->scales[group+4]&0x0F) |
               ((b->scales[group-4]>>6)<<4);

        *min=(b->scales[group+4]>>4) |
             ((b->scales[group]>>6)<<4);
    }
}

static void dequant_q4_k(
    const uint8_t *src,
    float *dst)
{
    const block_q4_K *b=(const block_q4_K *)src;

    float d=fp16_to_fp32(mem_u16(b->d));
    float dmin=fp16_to_fp32(mem_u16(b->dmin));

    for(int g=0;g<8;g++)
    {
        int sc,mn;
        get_scale_min(b,g,&sc,&mn);

        float ds=d*(float)sc;
        float dm=dmin*(float)mn;

        const uint8_t *q=b->qs+g*16;

        for(int i=0;i<16;i++)
        {
            dst[g*32+i]=
                ds*(float)(q[i]&0x0F)-dm;

            dst[g*32+i+16]=
                ds*(float)(q[i]>>4)-dm;
        }
    }
}

static float reference_dot(
    const uint8_t *src,
    const float *x)
{
    const block_q4_K *b=(const block_q4_K *)src;

    float d=fp16_to_fp32(mem_u16(b->d));
    float dmin=fp16_to_fp32(mem_u16(b->dmin));

    float sum=0.0f;

    for(int g=0;g<8;g++)
    {
        int sc,mn;
        get_scale_min(b,g,&sc,&mn);

        float ds=d*(float)sc;
        float dm=dmin*(float)mn;

        const uint8_t *q=b->qs+g*16;

        for(int i=0;i<16;i++)
        {
            float a=ds*(float)(q[i]&0x0F)-dm;
            float bval=ds*(float)(q[i]>>4)-dm;

            sum+=a*x[g*32+i];
            sum+=bval*x[g*32+i+16];
        }
    }

    return sum;
}

static float direct_dot(
    const uint8_t *src,
    const float *x)
{
    const block_q4_K *b=(const block_q4_K *)src;

    float d=fp16_to_fp32(mem_u16(b->d));
    float dmin=fp16_to_fp32(mem_u16(b->dmin));

    float sum=0.0f;

    for(int g=0;g<8;g++)
    {
        int sc,mn;
        get_scale_min(b,g,&sc,&mn);

        float ds=d*(float)sc;
        float dm=dmin*(float)mn;

        const uint8_t *q=b->qs+g*16;

        for(int i=0;i<16;i++)
        {
            sum +=
                (ds*(float)(q[i]&0x0F)-dm) *
                x[g*32+i];

            sum +=
                (ds*(float)(q[i]>>4)-dm) *
                x[g*32+i+16];
        }
    }

    return sum;
}

int main(void)
{
    const char *path=
        "model/atlas-1.0.gguf";

    /*
     * Verified absolute offset of the first real Q4_K tensor:
     * token_embd.weight = 197389888.
     */
    const uint64_t offset=197389888ULL;

    FILE *f=fopen(path,"rb");

    if(!f)
    {
        perror(path);
        return 1;
    }

    if(fseek(f,(long)offset,SEEK_SET)!=0)
    {
        printf("SEEK FAILED\n");
        fclose(f);
        return 1;
    }

    uint8_t block[144];

    if(fread(block,1,144,f)!=144)
    {
        printf("READ FAILED\n");
        fclose(f);
        return 1;
    }

    fclose(f);

    float x[256];

    /*
     * Deterministic test vector.
     */
    for(int i=0;i<256;i++)
        x[i]=((float)((i*37)%101)-50.0f)/50.0f;

    float a[256];

    dequant_q4_k(block,a);

    float reference=0.0f;

    for(int i=0;i<256;i++)
        reference+=a[i]*x[i];

    float direct=direct_dot(block,x);
    float independent=reference_dot(block,x);

    float err=fabsf(direct-reference);
    float err2=fabsf(independent-reference);

    printf("========================================\n");
    printf("Q4_K REAL DOT PRODUCT TEST\n");
    printf("========================================\n");

    printf("Tensor:           token_embd.weight\n");
    printf("Offset:           %llu\n",
        (unsigned long long)offset);

    printf("Block bytes:      144\n");
    printf("Values:           256\n");

    printf("\nReference dot:    %.12f\n",reference);
    printf("Direct Q4_K:      %.12f\n",direct);
    printf("Independent:      %.12f\n",independent);

    printf("\nDirect error:      %.12g\n",err);
    printf("Independent error: %.12g\n",err2);

    printf("\n========================================\n");

    if(err<1e-5f && err2<1e-5f)
        printf("Q4_K DOT PRODUCT: PASS\n");
    else
        printf("Q4_K DOT PRODUCT: FAIL\n");

    printf("========================================\n");

    return (err<1e-5f && err2<1e-5f)?0:1;
}
