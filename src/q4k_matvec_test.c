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

static void q4k_matvec(
    const uint8_t *matrix,
    const float *x,
    double *y,
    int rows,
    int cols)
{
    int blocks_per_row=cols/256;

    for(int row=0;row<rows;row++)
    {
        const uint8_t *row_data=
            matrix+(size_t)row*
            (size_t)blocks_per_row*144;

        double sum=0.0;

        for(int block=0;block<blocks_per_row;block++)
        {
            const block_q4_K *b=
                (const block_q4_K *)
                (row_data+block*144);

            float d=
                fp16_to_fp32(mem_u16(b->d));

            float dmin=
                fp16_to_fp32(mem_u16(b->dmin));

            const float *xx=x+block*256;

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
                        ((double)ds*(q[i]&0x0F)-dm) *
                        xx[g*32+i];

                    sum +=
                        ((double)ds*(q[i]>>4)-dm) *
                        xx[g*32+i+16];
                }
            }
        }

        y[row]=sum;
    }
}

static void reference_matvec(
    const uint8_t *matrix,
    const float *x,
    double *y,
    int rows,
    int cols)
{
    int blocks_per_row=cols/256;

    for(int row=0;row<rows;row++)
    {
        double sum=0.0;

        const uint8_t *row_data=
            matrix+(size_t)row*
            (size_t)blocks_per_row*144;

        for(int block=0;block<blocks_per_row;block++)
        {
            float w[256];

            dequant_q4_k(
                row_data+block*144,
                w);

            for(int i=0;i<256;i++)
                sum+=(double)w[i]*x[block*256+i];
        }

        y[row]=sum;
    }
}

int main(void)
{
    const int rows=4;
    const int cols=1024;
    const int blocks_per_row=cols/256;

    const size_t matrix_bytes=
        (size_t)rows*
        blocks_per_row*
        144;

    uint8_t *matrix=calloc(1,matrix_bytes);
    float *x=malloc(sizeof(float)*cols);
    double *reference=malloc(sizeof(double)*rows);
    double *actual=malloc(sizeof(double)*rows);

    if(!matrix || !x || !reference || !actual)
    {
        printf("ALLOCATION FAILED\n");
        free(matrix);
        free(x);
        free(reference);
        free(actual);
        return 1;
    }

    for(size_t i=0;i<matrix_bytes;i++)
        matrix[i]=(uint8_t)((i*73+19)&0xFF);

    for(int row=0;row<rows;row++)
    {
        for(int block=0;block<blocks_per_row;block++)
        {
            uint8_t *p=
                matrix+
                ((size_t)row*
                 blocks_per_row+
                 block)*144;

            p[0]=0x00;
            p[1]=0x24;

            p[2]=0x00;
            p[3]=0x20;
        }
    }

    for(int i=0;i<cols;i++)
        x[i]=((float)((i*29)%113)-56.0f)/56.0f;

    reference_matvec(
        matrix,x,reference,
        rows,cols);

    q4k_matvec(
        matrix,x,actual,
        rows,cols);

    double max_error=0.0;

    printf("========================================\n");
    printf("Q4_K MATRIX-VECTOR TEST\n");
    printf("========================================\n");

    printf("Rows:             %d\n",rows);
    printf("Columns:          %d\n",cols);
    printf("Blocks/row:       %d\n",blocks_per_row);
    printf("Total blocks:     %d\n",rows*blocks_per_row);
    printf("Bytes:            %llu\n",
        (unsigned long long)matrix_bytes);

    printf("\nResults:\n");

    for(int i=0;i<rows;i++)
    {
        double error=
            fabs(actual[i]-reference[i]);

        if(error>max_error)
            max_error=error;

        printf(
            "row %d  reference=% .12f  actual=% .12f  error=%.12g\n",
            i,
            reference[i],
            actual[i],
            error);
    }

    printf("\nMax error:        %.12g\n",max_error);

    printf("\n========================================\n");

    if(max_error<1e-10)
        printf("Q4_K MATRIX-VECTOR: PASS\n");
    else
        printf("Q4_K MATRIX-VECTOR: FAIL\n");

    printf("========================================\n");

    free(matrix);
    free(x);
    free(reference);
    free(actual);

    return max_error<1e-10 ? 0 : 1;
}
