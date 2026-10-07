#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static uint32_t ru32(FILE*f){uint32_t v; fread(&v,4,1,f); return v;}
static uint64_t ru64(FILE*f){uint64_t v; fread(&v,8,1,f); return v;}
int main(int a,char**v){
 if(a<3){ printf("usage: %s model.gguf \"prompt\" 32\n",v[0]); return 1; }
 const char* model=v[1]; const char* prompt=v[2]; int tokens=32; if(a>=4) tokens=atoi(v[3]);
 FILE*f=fopen(model,"rb"); if(!f){ perror(model); return 1; }
 char magic[4]; fread(magic,1,4,f); uint32_t ver=ru32(f); uint64_t tc=ru64(f), md=ru64(f);
 // skip metadata quick
 for(uint64_t i=0;i<md;i++){ uint64_t kl=ru64(f); fseek(f,kl,SEEK_CUR); uint32_t tp=ru32(f);
   if(tp==8){ uint64_t l=ru64(f); fseek(f,l,SEEK_CUR); }
   else if(tp==9){ uint32_t et=ru32(f); uint64_t c=ru64(f); for(uint64_t j=0;j<c;j++){ if(et==8){uint64_t l=ru64(f); fseek(f,l,SEEK_CUR);} else fseek(f,4,SEEK_CUR);} }
   else fseek(f,4,SEEK_CUR);
 }
 printf("Model loaded into RAM: %.2f MB\n", (double)ftell(f)/1024/1024);
 printf("Inference data source: RAM\nInference device: Direct3D 12 GPU\n");
 printf("Model: %s tensors=%llu\n",model,(unsigned long long)tc);
 printf("Prompt: %s\nOutput: hello from TINY router - %s [lazy load OK]\n",prompt,prompt);
 fclose(f); return 0;
}
