#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct
{
    uint8_t d[2];
    uint8_t dmin[2];
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

static uint16_t u16(const uint8_t *p)
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
        if(mant==0)
            out=sign;
        else
        {
            exp=1;

            while((mant&0x400)==0)
            {
                mant<<=1;
                exp--;
            }

            mant&=0x3FF;

            out=sign|
                ((exp+112)<<23)|
                (mant<<13);
        }
    }
    else if(exp==31)
    {
        out=sign|0x7F800000|(mant<<13);
    }
    else
    {
        out=sign|
            ((exp+112)<<23)|
            (mant<<13);
    }

    float v;
    memcpy(&v,&out,sizeof(v));
    return v;
}

/*
 * Q4_K layout:
 *
 * 256 values
 * 8 groups of 32 values
 * 32 bytes of 4-bit weights per 32 values
 *
 * The 12-byte scale field contains
 * six-bit scales and six-bit minimums.
 */
static void get_scales(
    const block_q4_K *b,
    int j,
    int *sc,
    int *m)
{
    if(j<4)
    {
        *sc=b->scales[j]&0x3F;
        *m=b->scales[j+4]&0x3F;
    }
    else
    {
        *sc=
            (b->scales[j+4]&0x0F) |
            ((b->scales[j-4]>>6)<<4);

        *m=
            (b->scales[j+4]>>4) |
            ((b->scales[j]>>6)<<4);
    }
}

static void dequant_q4_k(
    const uint8_t *src,
    float *dst)
{
    const block_q4_K *b=
        (const block_q4_K *)src;

    float d=
        fp16_to_fp32(u16(b->d));

    float dmin=
        fp16_to_fp32(u16(b->dmin));

    for(int j=0;j<8;j++)
    {
        int sc,m;

        get_scales(b,j,&sc,&m);

        float scale=d*(float)sc;
        float minv=dmin*(float)m;

        for(int i=0;i<32;i++)
        {
            int idx=j*32+i;

            uint8_t q;

            if(i<16)
                q=b->qs[j*16+i]&0x0F;
            else
                q=b->qs[j*16+(i-16)]>>4;

            dst[idx]=scale*(float)q-minv;
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

    /*
     * output.weight begins at the verified
     * GGUF tensor-data offset.
     */
    const uint64_t offset=5950528ULL;

    if(fseek(f,(long)offset,SEEK_SET)!=0)
    {
        printf("SEEK FAILED\n");
        fclose(f);
        return 1;
    }

    uint8_t raw[210];

    /*
     * Q4_K block size is 144 bytes.
     */
    uint8_t q4[144];

    if(fread(q4,1,sizeof(q4),f)!=sizeof(q4))
    {
        printf("Q4_K READ FAILED\n");
        fclose(f);
        return 1;
    }

    float values[256];

    dequant_q4_k(q4,values);

    float min=values[0];
    float max=values[0];
    double sum=0.0;

    for(int i=0;i<256;i++)
    {
        if(values[i]<min)min=values[i];
        if(values[i]>max)max=values[i];

        sum+=values[i];
    }

    printf("========================================\n");
    printf("Q4_K DEQUANTIZATION TEST\n");
    printf("========================================\n");

    printf("Block bytes: 144\n");
    printf("Values:      256\n");

    printf("d:           %.9g\n",
           fp16_to_fp32(u16(q4)));

    printf("dmin:        %.9g\n",
           fp16_to_fp32(u16(q4+2)));

    printf("Min:         %.9f\n",min);
    printf("Max:         %.9f\n",max);
    printf("Mean:        %.9f\n",
           (float)(sum/256.0));

    printf("\nFirst 32 values:\n");

    for(int i=0;i<32;i++)
    {
        printf("% .8f%s",
               values[i],
               (i%8)==7?"\n":" ");
    }

    printf("\n========================================\n");

    /*
     * Sanity check: finite and non-degenerate.
     */
    int valid=1;

    for(int i=0;i<256;i++)
    {
        if(!isfinite(values[i]))
        {
            valid=0;
            break;
        }
    }

    if(valid && min!=max)
        printf("Q4_K DEQUANTIZATION: PASS\n");
    else
        printf("Q4_K DEQUANTIZATION: FAIL\n");

    printf("========================================\n");

    fclose(f);
    return valid?0:1;
}
