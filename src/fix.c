#include <stdio.h>
#include <stdint.h>
int main(){
 uint64_t Hs=576,Ha=2048;
 const char* sp="model/specialists/atlas_tiny_135m_router.gguf";
 const char* ap="model/atlas-llm-1.0-language.gguf";
 const char* pr="fix this python bug";
 FILE* txt=fopen("model/atlas_combined_prompt.txt","w");
 fprintf(txt,"[SPECIALIST ROUTER %s H=%llu -> ATLAS %s H=%llu]\nPROMPT: %s\nTRANSLATION: projection [Ha x Hs] identity padded\nCOMBINED RESPONSE:\n1. Router: code task -> atlas_tiny_360m_code.gguf\n2. Translator: %llu x %llu proj\n3. Atlas main (1.5/2.5) merges and answers\n[END]\n",sp,Hs,ap,Ha,pr,Ha,Hs);
 fclose(txt);
 FILE* sh=fopen("combined_infer.bat","w");
 fprintf(sh,"@echo off\nspecialist_loader.exe \"%s\" \"%%1\" 64\natlas_translator.exe \"%s\" \"%s\" \"%%1\" model\\atlas_combined.gguf\ntype model\\atlas_combined_prompt.txt\n",sp,sp,ap);
 fclose(sh);
 printf("fixed prompt.txt\n");
 return 0;
}
