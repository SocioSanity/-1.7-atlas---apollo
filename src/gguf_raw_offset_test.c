#include <stdio.h>
#include <stdint.h>

static uint16_t u16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static void dump(FILE *f, uint64_t pos, const char *label)
{
    uint8_t b[32];

    if(fseek(f,(long)pos,SEEK_SET)!=0){
        printf("SEEK FAILED: %s\n",label);
        return;
    }

    if(fread(b,1,sizeof(b),f)!=sizeof(b)){
        printf("READ FAILED: %s\n",label);
        return;
    }

    printf("\n%s\n",label);
    printf("Offset: %llu\n",(unsigned long long)pos);

    printf("Bytes:  ");
    for(int i=0;i<32;i++)
        printf("%02X ",b[i]);

    printf("\n");

    printf("First u16: 0x%04X (%u)\n",
           u16(b),u16(b));

    printf("Last  u16: 0x%04X (%u)\n",
           u16(b+30),u16(b+30));
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
    printf("GGUF RAW DATA LOCATION TEST\n");
    printf("========================================\n");

    dump(f,5931232ULL,
         "GGUF-CALCULATED DATA START");

    dump(f,5950528ULL,
         "PREVIOUS Q6_K TEST LOCATION");

    fclose(f);

    printf("\n========================================\n");
    return 0;
}
