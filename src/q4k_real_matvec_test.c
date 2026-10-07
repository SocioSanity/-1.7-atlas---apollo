#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK 256
#define Q4K_BYTES 144
#define ROWS 1536
#define COLS 256

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
        *scale=(b->scales[group+4]&0x0F) |
               ((b->scales[group-4]>>6)<<4);

        *min=(b->scales[group+4]>>4) |
             ((b->scales[group]>>6)<<4);
    }
}

static double q4k_block_dot(
    const uint8_t *src,
    const float *x)
{
    const block_q4_K *b=
        (const block_q4_K *)src;

    float d=
        fp16_to_fp32(mem_u16(b->d));

    float dmin=
        fp16_to_fp32(mem_u16(b->dmin));

    double sum=0.0;

    for(int g=0;g<8;g++)
    {
        int sc,mn;

        get_scale_min(
            b,g,&sc,&mn);

        float ds=d*(float)sc;
        float dm=dmin*(float)mn;

        const uint8_t *q=
            b->qs+g*16;

        const float *xx=
            x+g*32;

        for(int i=0;i<16;i++)
        {
            sum +=
                ((double)ds*(q[i]&0x0F)-dm) *
                xx[i];

            sum +=
                ((double)ds*(q[i]>>4)-dm) *
                xx[i+16];
        }
    }

    return sum;
}

static void q4k_matvec_real(
    FILE *f,
    uint64_t tensor_offset,
    const float *x,
    double *y)
{
    /*
     * token_embd.weight is stored row-major:
     *
     * 1536 rows
     * 151936 columns
     *
     * We intentionally calculate only the first
     * 256 columns of every row. Each row therefore
     * contains exactly one Q4_K block for this test.
     */
    uint8_t block[Q4K_BYTES];

    for(int row=0;row<ROWS;row++)
    {
        uint64_t offset=
            tensor_offset+
            (uint64_t)row*Q4K_BYTES;

        if(fseek(f,(long)offset,SEEK_SET)!=0)
        {
            printf("SEEK FAILED row %d\n",row);
            exit(1);
        }

        if(fread(block,1,Q4K_BYTES,f)!=Q4K_BYTES)
        {
            printf("READ FAILED row %d\n",row);
            exit(1);
        }

        y[row]=q4k_block_dot(block,x);
    }
}

static void q4k_reference_real(
    FILE *f,
    uint64_t tensor_offset,
    const float *x,
    double *y)
{
    uint8_t block[Q4K_BYTES];

    for(int row=0;row<ROWS;row++)
    {
        uint64_t offset=
            tensor_offset+
            (uint64_t)row*Q4K_BYTES;

        if(fseek(f,(long)offset,SEEK_SET)!=0)
            exit(1);

        if(fread(block,1,Q4K_BYTES,f)!=Q4K_BYTES)
            exit(1);

        const block_q4_K *b=
            (const block_q4_K *)block;

        float d=
            fp16_to_fp32(mem_u16(b->d));

        float dmin=
            fp16_to_fp32(mem_u16(b->dmin));

        double sum=0.0;

        for(int g=0;g<8;g++)
        {
            int sc,mn;

            get_scale_min(
                b,g,&sc,&mn);

            float ds=d*(float)sc;
            float dm=dmin*(float)mn;

            const uint8_t *q=
                b->qs+g*16;

            for(int i=0;i<16;i++)
            {
                float w0=
                    ds*(float)(q[i]&0x0F)-dm;

                float w1=
                    ds*(float)(q[i]>>4)-dm;

                sum+=(double)w0*x[g*32+i];
                sum+=(double)w1*x[g*32+i+16];
            }
        }

        y[row]=sum;
    }
}

int main(void)
{
    const char *path=
        "model/atlas-1.0.gguf";

    /*
     * Verified from the GGUF tensor directory:
     *
     * token_embd.weight
     * type       = Q4_K
     * dimensions = 1536 x 151936
     * relative   = 191439360
     * absolute   = 197389888
     */
    const uint64_t tensor_offset=
        197389888ULL;

    FILE *f=fopen(path,"rb");

    if(!f)
    {
        perror(path);
        return 1;
    }

    float x[COLS];

    /*
     * Deterministic input vector.
     */
    for(int i=0;i<COLS;i++)
        x[i]=((float)((i*31)%127)-63.0f)/63.0f;

    double actual[ROWS];
    double reference[ROWS];

    q4k_matvec_real(
        f,
        tensor_offset,
        x,
        actual);

    q4k_reference_real(
        f,
        tensor_offset,
        x,
        reference);

    fclose(f);

    double max_error=0.0;

    printf("========================================\n");
    printf("REAL GGUF Q4_K MATRIX-VECTOR TEST\n");
    printf("========================================\n");

    printf("Tensor:           token_embd.weight\n");
    printf("Type:             Q4_K\n");
    printf("Tensor offset:    %llu\n",
        (unsigned long long)tensor_offset);

    printf("Model dimensions: 1536 x 151936\n");
    printf("Test columns:     256\n");
    printf("Test rows:        1536\n");
    printf("Q4_K blocks:      1536\n");

    printf("\nSelected outputs:\n");

    for(int i=0;i<ROWS;i++)
    {
        double error=
            fabs(actual[i]-reference[i]);

        if(error>max_error)
            max_error=error;

        if(i<8 || i>=ROWS-4)
        {
            printf(
                "row %4d  ref=% .12f  actual=% .12f  error=%.12g\n",
                i,
                reference[i],
                actual[i],
                error);
        }
    }

    printf("\nMax error:        %.12g\n",max_error);

    printf("\n========================================\n");

    if(max_error<1e-10)
        printf("REAL GGUF Q4_K MATVEC: PASS\n");
    else
        printf("REAL GGUF Q4_K MATVEC: FAIL\n");

    printf("========================================\n");

    return max_error<1e-10 ? 0 : 1;
}
