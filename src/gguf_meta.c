#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int r32(FILE*f,uint32_t*v){return fread(v,4,1,f)==1;}
static int r64(FILE*f,uint64_t*v){return fread(v,8,1,f)==1;}

static char* rs(FILE*f){
    uint64_t n;
    if(!r64(f,&n)||n>1000000)return NULL;
    char*s=malloc((size_t)n+1);
    if(!s)return NULL;
    if(fread(s,1,(size_t)n,f)!=n){free(s);return NULL;}
    s[n]=0;
    return s;
}

static const char* mt(uint32_t t){
    switch(t){
        case 0:return"UINT8";case 1:return"INT8";case 2:return"UINT16";
        case 3:return"INT16";case 4:return"UINT32";case 5:return"INT32";
        case 6:return"FLOAT32";case 7:return"BOOL";case 8:return"STRING";
        case 9:return"ARRAY";case 10:return"UINT64";case 11:return"INT64";
        case 12:return"FLOAT64";default:return"UNKNOWN";
    }
}

static const char* tt(uint32_t t){
    switch(t){
        case 0:return"F32";case 1:return"F16";case 2:return"Q4_0";
        case 3:return"Q4_1";case 6:return"Q5_0";case 7:return"Q5_1";
        case 8:return"Q8_0";case 9:return"Q8_1";case 10:return"Q2_K";
        case 11:return"Q3_K_S";case 12:return"Q3_K_M";case 13:return"Q3_K_L";
        case 14:return"Q4_K_S";case 15:return"Q4_K_M";case 16:return"Q5_K_S";
        case 17:return"Q5_K_M";case 18:return"Q6_K";case 19:return"IQ2_XXS";
        case 20:return"IQ2_XS";case 21:return"IQ3_XXS";case 22:return"IQ1_S";
        case 23:return"IQ4_NL";case 24:return"IQ3_S";case 25:return"IQ2_S";
        case 26:return"IQ4_XS";case 27:return"I8";case 28:return"I16";
        case 29:return"I32";case 30:return"I64";case 31:return"F64";
        case 32:return"IQ1_M";case 33:return"BF16";
        default:return"UNKNOWN";
    }
}

static int skip(FILE*f,uint32_t t){
    switch(t){
        case 0:case 1:case 7:{uint8_t x;return fread(&x,1,1,f)==1;}
        case 2:case 3:{uint16_t x;return fread(&x,2,1,f)==1;}
        case 4:case 5:case 6:{uint32_t x;return fread(&x,4,1,f)==1;}
        case 8:{char*s=rs(f);if(!s)return 0;free(s);return 1;}
        case 9:{
            uint32_t e;uint64_t n;
            if(!r32(f,&e)||!r64(f,&n))return 0;
            for(uint64_t i=0;i<n;i++)if(!skip(f,e))return 0;
            return 1;
        }
        case 10:case 11:{uint64_t x;return fread(&x,8,1,f)==1;}
        case 12:{double x;return fread(&x,8,1,f)==1;}
        default:return 0;
    }
}

static void inspect(const char*p){
    FILE*f=fopen(p,"rb");
    if(!f){printf("OPEN FAILED: %s\n",p);return;}

    uint32_t magic,ver;
    uint64_t nt,nm;

    if(!r32(f,&magic)||!r32(f,&ver)||!r64(f,&nt)||!r64(f,&nm)){
        printf("HEADER FAILED\n");fclose(f);return;
    }

    printf("\n============================================================\n%s\n",p);
    printf("GGUF version: %u\nTensors:      %llu\nMetadata:     %llu\n",
           ver,(unsigned long long)nt,(unsigned long long)nm);
    printf("------------------------------------------------------------\n");

    for(uint64_t i=0;i<nm;i++){
        char*k=rs(f);uint32_t t;
        if(!k||!r32(f,&t)){free(k);printf("METADATA FAILED\n");fclose(f);return;}
        printf("META [%llu] %-45s %-10s",
               (unsigned long long)i,k,mt(t));

        if(t==9){
            uint32_t e;uint64_t n;
            if(!r32(f,&e)||!r64(f,&n)){free(k);fclose(f);return;}
            printf(" elem=%s count=%llu\n",mt(e),(unsigned long long)n);
            for(uint64_t j=0;j<n;j++)if(!skip(f,e)){free(k);fclose(f);return;}
        }else{
            printf("\n");
            if(!skip(f,t)){free(k);fclose(f);return;}
        }
        free(k);
    }

    printf("\n------------------------------------------------------------\nTENSORS\n------------------------------------------------------------\n");

    for(uint64_t i=0;i<nt;i++){
        char*n=rs(f);uint32_t nd,type;uint64_t d[16],off;

        if(!n||!r32(f,&nd)||nd>16){
            free(n);printf("TENSOR HEADER FAILED at %llu\n",(unsigned long long)i);
            fclose(f);return;
        }

        for(uint32_t j=0;j<nd;j++)if(!r64(f,&d[j])){
            free(n);fclose(f);return;
        }

        if(!r32(f,&type)||!r64(f,&off)){
            free(n);fclose(f);return;
        }

        printf("[%3llu] %-45s [",(unsigned long long)i,n);
        for(uint32_t j=0;j<nd;j++){
            if(j)printf(" x ");
            printf("%llu",(unsigned long long)d[j]);
        }
        printf("] %-10s offset=%llu\n",tt(type),(unsigned long long)off);
        free(n);
    }

    fclose(f);
}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"Usage: %s file.gguf\n",argv[0]);return 1;}
    for(int i=1;i<argc;i++)inspect(argv[i]);
    return 0;
}
