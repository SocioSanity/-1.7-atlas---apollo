#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int ru32(FILE *f, uint32_t *v){ return fread(v,4,1,f)==1; }
static int ru64(FILE *f, uint64_t *v){ return fread(v,8,1,f)==1; }

static int skip_str(FILE *f){
  uint64_t n; if(!ru64(f,&n)) return 0;
  return fseek(f,(long)n,SEEK_CUR)==0;
}
static int skip_val(FILE *f, uint32_t t);
static int skip_arr(FILE *f){
  uint32_t et; uint64_t n; if(!ru32(f,&et)||!ru64(f,&n)) return 0;
  for(uint64_t i=0;i<n;i++) if(!skip_val(f,et)) return 0;
  return 1;
}
static int skip_val(FILE *f, uint32_t t){
  switch(t){
    case 0: case 1: case 7: return fseek(f,1,SEEK_CUR)==0;
    case 2: case 3: return fseek(f,2,SEEK_CUR)==0;
    case 4: case 5: case 6: return fseek(f,4,SEEK_CUR)==0;
    case 8: return skip_str(f);
    case 9: return skip_arr(f);
    case 10: case 11: case 12: return fseek(f,8,SEEK_CUR)==0;
    default: return 0;
  }
}
static int read_str(FILE *f, char *buf, size_t cap){
  uint64_t n; if(!ru64(f,&n)) return 0;
  if(n>=cap){ if(fseek(f,(long)n,SEEK_CUR)!=0) return 0; buf[0]=0; return 1; }
  if(fread(buf,1,(size_t)n,f)!=n) return 0; buf[n]=0; return 1;
}
static void print_val(FILE *f, uint32_t t){
  if(t==4){ uint32_t v; ru32(f,&v); printf("%u", v); return; }
  if(t==5){ int32_t v; fread(&v,4,1,f); printf("%d", v); return; }
  if(t==6){ float v; fread(&v,4,1,f); printf("%g", v); return; }
  if(t==8){ char s[256]; read_str(f,s,sizeof s); printf("%s", s); return; }
  if(t==10){ uint64_t v; ru64(f,&v); printf("%llu",(unsigned long long)v); return; }
  if(t==11){ int64_t v; fread(&v,8,1,f); printf("%lld",(long long)v); return; }
  skip_val(f,t); printf("<%u>", t);
}
int main(int argc, char **argv){
  FILE *f=fopen(argv[1],"rb");
  uint32_t mag,ver; uint64_t nt,nm;
  ru32(f,&mag); ru32(f,&ver); ru64(f,&nt); ru64(f,&nm);
  printf("ver=%u tensors=%llu kv=%llu\n", ver,(unsigned long long)nt,(unsigned long long)nm);
  for(uint64_t i=0;i<nm;i++){
    char key[256]; uint32_t t;
    read_str(f,key,sizeof key); ru32(f,&t);
    if(strstr(key,"block")||strstr(key,"embed")||strstr(key,"head")||strstr(key,"ffn")||
       strstr(key,"rope")||strstr(key,"rms")||strstr(key,"context")||strstr(key,"token")||
       strstr(key,"arch")||strstr(key,"name")||strstr(key,"eos")||strstr(key,"bos")||
       strstr(key,"align")){
      printf("META %s type=%u val=", key, t);
      print_val(f,t); printf("\n");
    } else skip_val(f,t);
  }
  printf("\nTENSORS:\n");
  for(uint64_t i=0;i<nt && i<40;i++){
    char name[256]; uint32_t nd, ty; uint64_t dims[8], off;
    read_str(f,name,sizeof name); ru32(f,&nd);
    for(uint32_t d=0;d<nd;d++) ru64(f,&dims[d]);
    ru32(f,&ty); ru64(f,&off);
    printf("%-40s type=%u nd=%u ", name, ty, nd);
    for(uint32_t d=0;d<nd;d++) printf("%llu%s",(unsigned long long)dims[d], d+1<nd?"x":"");
    printf(" off=%llu\n",(unsigned long long)off);
  }
  fclose(f); return 0;
}
