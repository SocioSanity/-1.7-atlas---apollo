#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint8_t u8(FILE *f)
{
    int c=fgetc(f);
    if(c==EOF)exit(1);
    return(uint8_t)c;
}

static uint16_t u16(FILE *f)
{
    uint8_t b[2];
    if(fread(b,1,2,f)!=2)exit(1);
    return(uint16_t)b[0]|((uint16_t)b[1]<<8);
}

static uint32_t u32(FILE *f)
{
    uint8_t b[4];
    if(fread(b,1,4,f)!=4)exit(1);

    return(uint32_t)b[0] |
           ((uint32_t)b[1]<<8) |
           ((uint32_t)b[2]<<16) |
           ((uint32_t)b[3]<<24);
}

static uint64_t u64(FILE *f)
{
    uint8_t b[8];

    if(fread(b,1,8,f)!=8)exit(1);

    return(uint64_t)b[0] |
           ((uint64_t)b[1]<<8) |
           ((uint64_t)b[2]<<16) |
           ((uint64_t)b[3]<<24) |
           ((uint64_t)b[4]<<32) |
           ((uint64_t)b[5]<<40) |
           ((uint64_t)b[6]<<48) |
           ((uint64_t)b[7]<<56);
}

static char *read_string(FILE *f)
{
    uint64_t n=u64(f);

    if(n>10000000ULL)
        exit(1);

    char *s=malloc((size_t)n+1);

    if(!s)
        exit(1);

    if(fread(s,1,(size_t)n,f)!=(size_t)n)
        exit(1);

    s[n]=0;

    return s;
}

static void skip_value(FILE *f,uint32_t type);

static void skip_array(FILE *f)
{
    uint32_t element_type=u32(f);
    uint64_t count=u64(f);

    for(uint64_t i=0;i<count;i++)
        skip_value(f,element_type);
}

static void skip_value(FILE *f,uint32_t type)
{
    switch(type)
    {
        case 0:
        case 1:
        case 7:
            u8(f);
            break;

        case 2:
        case 3:
            u16(f);
            break;

        case 4:
        case 5:
        case 6:
            u32(f);
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
            u64(f);
            break;

        default:
            fprintf(stderr,"unknown metadata type %u\n",type);
            exit(1);
    }
}

static uint64_t align_to(uint64_t x,uint64_t alignment)
{
    return (x+alignment-1)/alignment*alignment;
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

    uint32_t magic=u32(f);
    uint32_t version=u32(f);
    uint64_t tensor_count=u64(f);
    uint64_t metadata_count=u64(f);

    printf("========================================\n");
    printf("GGUF TRUE DATA OFFSET TEST\n");
    printf("========================================\n");

    printf("Magic:       0x%08X\n",magic);
    printf("Version:     %u\n",version);
    printf("Tensors:     %llu\n",
           (unsigned long long)tensor_count);
    printf("Metadata:    %llu\n",
           (unsigned long long)metadata_count);

    if(magic!=0x46554747 || version!=3)
    {
        printf("INVALID GGUF\n");
        fclose(f);
        return 1;
    }

    uint64_t alignment=32;

    /*
     * Read metadata.
     */
    for(uint64_t i=0;i<metadata_count;i++)
    {
        char *key=read_string(f);
        uint32_t type=u32(f);

        if(strcmp(key,"general.alignment")==0 &&
           type==4)
        {
            alignment=u32(f);
        }
        else
        {
            skip_value(f,type);
        }

        free(key);
    }

    /*
     * IMPORTANT:
     * The tensor directory comes AFTER metadata.
     * We must consume ALL tensor descriptors before
     * calculating the tensor-data start.
     */
    uint64_t output_relative=0;
    uint32_t output_type=0;
    uint64_t output_elements=0;
    int found_output=0;

    for(uint64_t i=0;i<tensor_count;i++)
    {
        char *name=read_string(f);

        uint32_t n_dims=u32(f);

        if(n_dims>8)
        {
            fprintf(stderr,"invalid dimensions: %u\n",n_dims);
            free(name);
            fclose(f);
            return 1;
        }

        uint64_t elements=1;

        for(uint32_t d=0;d<n_dims;d++)
        {
            uint64_t dim=u64(f);
            elements*=dim;
        }

        uint32_t type=u32(f);
        uint64_t relative=u64(f);

        if(strcmp(name,"output.weight")==0)
        {
            output_relative=relative;
            output_type=type;
            output_elements=elements;
            found_output=1;
        }

        free(name);
    }

    /*
     * NOW the file pointer is at the actual end
     * of the tensor directory.
     */
    uint64_t directory_end=(uint64_t)ftell(f);

    uint64_t data_start=
        align_to(directory_end,alignment);

    uint64_t output_absolute=
        data_start+output_relative;

    printf("\nTensor directory end: %llu\n",
           (unsigned long long)directory_end);

    printf("Alignment:            %llu\n",
           (unsigned long long)alignment);

    printf("Tensor data start:    %llu\n",
           (unsigned long long)data_start);

    printf("\noutput.weight:\n");

    printf("Found:                %s\n",
           found_output?"YES":"NO");

    printf("Type:                 %u\n",
           output_type);

    printf("Elements:             %llu\n",
           (unsigned long long)output_elements);

    printf("Relative offset:      %llu\n",
           (unsigned long long)output_relative);

    printf("Absolute offset:      %llu\n",
           (unsigned long long)output_absolute);

    printf("\nPrevious valid test:  5950528\n");

    printf("Difference:           %lld\n",
           (long long)output_absolute-5950528LL);

    printf("\n========================================\n");

    if(found_output &&
       output_absolute==5950528ULL)
    {
        printf("TRUE DATA OFFSET: PASS\n");
    }
    else
    {
        printf("TRUE DATA OFFSET: FAIL\n");
    }

    printf("========================================\n");

    fclose(f);

    return(found_output &&
           output_absolute==5950528ULL)?0:1;
}
