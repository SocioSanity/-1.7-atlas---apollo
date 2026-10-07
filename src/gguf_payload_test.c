#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint16_t u16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t u32(const uint8_t *p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void dump(FILE *f,uint64_t pos)
{
    uint8_t b[210];

    if(fseek(f,(long)pos,SEEK_SET)!=0)
        return;

    if(fread(b,1,sizeof(b),f)!=sizeof(b))
        return;

    printf("\nOffset %llu\n",(unsigned long long)pos);

    printf("d bytes: %02X %02X\n",b[208],b[209]);
    printf("d raw:   0x%04X\n",u16(b+208));

    printf("first 32: ");
    for(int i=0;i<32;i++)
        printf("%02X ",b[i]);
    printf("\n");

    printf("last  16: ");
    for(int i=194;i<210;i++)
        printf("%02X ",b[i]);
    printf("\n");
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

    printf("========================================\n");
    printf("GGUF PAYLOAD STRUCTURE TEST\n");
    printf("========================================\n");

    printf("\nDirectory end:       5931202\n");
    printf("Calculated start:    5931232\n");
    printf("Known valid region:  5950528\n");
    printf("Gap:                 19296 bytes\n");

    dump(f,5931232ULL);
    dump(f,5950528ULL);

    printf("\nSearching for matching block boundary...\n");

    int found=0;

    for(uint64_t p=5931232ULL;p<5950528ULL;p+=32){
        if(fseek(f,(long)p,SEEK_SET)!=0)
            break;

        uint8_t b[210];

        if(fread(b,1,sizeof(b),f)!=sizeof(b))
            break;

        uint16_t d=u16(b+208);

        /*
         * Q6_K FP16 scale should be a finite,
         * reasonably small positive/negative value.
         *
         * Search for the known valid block signature:
         * the bytes at the known location have
         * d = 0x0000?  We therefore report candidates
         * rather than assuming a particular value.
         */

        if(d!=0){
            printf("candidate: %llu  d=0x%04X\n",
                   (unsigned long long)p,d);

            found++;

            if(found>=20)
                break;
        }
    }

    printf("\nCandidates: %d\n",found);

    printf("\n========================================\n");

    fclose(f);
    return 0;
}
