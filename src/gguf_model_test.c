#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_MAGIC 0x46554747u
#define GGUF_MAX_DIMS 8

typedef struct
{
    char *name;
    uint32_t n_dims;
    uint64_t dims[GGUF_MAX_DIMS];
    uint64_t elements;
    uint32_t type;
    uint64_t relative_offset;
    uint64_t absolute_offset;
} GGUF_Tensor;

typedef struct
{
    FILE *file;

    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t alignment;
    uint64_t data_start;

    GGUF_Tensor *tensors;
} GGUF_Model;

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

    return(uint16_t)b[0] |
           ((uint16_t)b[1]<<8);
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
    {
        fprintf(stderr,"invalid GGUF string length\n");
        exit(1);
    }

    char *s=malloc((size_t)n+1);

    if(!s)exit(1);

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
            fprintf(stderr,"unknown GGUF metadata type %u\n",type);
            exit(1);
    }
}

static uint64_t align_to(uint64_t x,uint64_t alignment)
{
    return(x+alignment-1)/alignment*alignment;
}

static int gguf_open(GGUF_Model *model,const char *path)
{
    memset(model,0,sizeof(*model));

    model->file=fopen(path,"rb");

    if(!model->file)
    {
        perror(path);
        return 0;
    }

    uint32_t magic=u32(model->file);

    if(magic!=GGUF_MAGIC)
    {
        fprintf(stderr,"invalid GGUF magic\n");
        fclose(model->file);
        return 0;
    }

    model->version=u32(model->file);
    model->tensor_count=u64(model->file);
    model->metadata_count=u64(model->file);

    if(model->version<2 || model->version>3)
    {
        fprintf(stderr,"unsupported GGUF version %u\n",
                model->version);
        fclose(model->file);
        return 0;
    }

    model->alignment=32;

    for(uint64_t i=0;i<model->metadata_count;i++)
    {
        char *key=read_string(model->file);
        uint32_t type=u32(model->file);

        if(strcmp(key,"general.alignment")==0 &&
           type==4)
        {
            model->alignment=u32(model->file);
        }
        else
        {
            skip_value(model->file,type);
        }

        free(key);
    }

    model->tensors=
        calloc((size_t)model->tensor_count,
               sizeof(GGUF_Tensor));

    if(!model->tensors)
    {
        fclose(model->file);
        return 0;
    }

    for(uint64_t i=0;i<model->tensor_count;i++)
    {
        GGUF_Tensor *t=&model->tensors[i];

        t->name=read_string(model->file);

        t->n_dims=u32(model->file);

        if(t->n_dims>GGUF_MAX_DIMS)
        {
            fprintf(stderr,
                    "tensor %s has too many dimensions\n",
                    t->name);

            return 0;
        }

        t->elements=1;

        for(uint32_t d=0;d<t->n_dims;d++)
        {
            t->dims[d]=u64(model->file);
            t->elements*=t->dims[d];
        }

        t->type=u32(model->file);
        t->relative_offset=u64(model->file);
    }

    uint64_t directory_end=
        (uint64_t)ftell(model->file);

    model->data_start=
        align_to(directory_end,model->alignment);

    for(uint64_t i=0;i<model->tensor_count;i++)
    {
        model->tensors[i].absolute_offset=
            model->data_start+
            model->tensors[i].relative_offset;
    }

    return 1;
}

static GGUF_Tensor *gguf_find_tensor(
    GGUF_Model *model,
    const char *name)
{
    for(uint64_t i=0;i<model->tensor_count;i++)
    {
        if(strcmp(model->tensors[i].name,name)==0)
            return &model->tensors[i];
    }

    return NULL;
}

static void gguf_close(GGUF_Model *model)
{
    if(model->tensors)
    {
        for(uint64_t i=0;i<model->tensor_count;i++)
            free(model->tensors[i].name);

        free(model->tensors);
    }

    if(model->file)
        fclose(model->file);

    memset(model,0,sizeof(*model));
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "model/atlas-1.0.gguf";

    GGUF_Model model;

    if(!gguf_open(&model,path))
    {
        printf("GGUF OPEN: FAIL\n");
        return 1;
    }

    printf("========================================\n");
    printf("GGUF MODEL LOADER TEST\n");
    printf("========================================\n");

    printf("Version:       %u\n",model.version);
    printf("Tensors:       %llu\n",
           (unsigned long long)model.tensor_count);
    printf("Metadata:      %llu\n",
           (unsigned long long)model.metadata_count);
    printf("Alignment:     %llu\n",
           (unsigned long long)model.alignment);
    printf("Data start:    %llu\n",
           (unsigned long long)model.data_start);

    GGUF_Tensor *output=
        gguf_find_tensor(&model,"output.weight");

    if(!output)
    {
        printf("\noutput.weight: NOT FOUND\n");
        gguf_close(&model);
        return 1;
    }

    printf("\noutput.weight\n");
    printf("Type:          %u\n",output->type);
    printf("Elements:      %llu\n",
           (unsigned long long)output->elements);
    printf("Dims:          ");

    for(uint32_t d=0;d<output->n_dims;d++)
        printf("%llu%s",
               (unsigned long long)output->dims[d],
               d+1==output->n_dims?"\n":" x ");

    printf("Relative:      %llu\n",
           (unsigned long long)output->relative_offset);

    printf("Absolute:      %llu\n",
           (unsigned long long)output->absolute_offset);

    GGUF_Tensor *v=
        gguf_find_tensor(&model,"blk.0.attn_v.weight");

    if(!v)
    {
        printf("\nblk.0.attn_v.weight: NOT FOUND\n");
        gguf_close(&model);
        return 1;
    }

    printf("\nblk.0.attn_v.weight\n");
    printf("Type:          %u\n",v->type);
    printf("Elements:      %llu\n",
           (unsigned long long)v->elements);
    printf("Absolute:      %llu\n",
           (unsigned long long)v->absolute_offset);

    printf("\n========================================\n");

    if(output->absolute_offset >= model.data_start &&
       v->absolute_offset >= model.data_start &&
       output->elements > 0 &&
       v->elements > 0)
    {
        printf("GGUF MODEL LOADER: PASS\n");
    }
    else
    {
        printf("GGUF MODEL LOADER: FAIL\n");
    }

    printf("========================================\n");

    gguf_close(&model);

    return 0;
}




