#include <stdio.h>
#include <stdlib.h>
int main(int a,char**v){
 if(a<2){printf("usage: atlas_combined_infer.exe \"prompt\"\n"); return 1;}
 char cmd[2048];
 snprintf(cmd,2048,"atlas_translate_chain.bat \"%s\"",v[1]);
 system(cmd);
 printf("\n=== ATLAS MAIN (1.5/2.5 standin) ===\n");
 // call your real main if you have it built - for now use language model infer
 if(fopen("model/atlas-llm-1.0-language.gguf","rb")){
   // brain_runtime expects GGUF + prompt - use your existing infer
   // If you have atlas_1.5_infer.exe / atlas_2.5_infer.exe replace here
   printf("Feeding combined context to Atlas main...\n");
   snprintf(cmd,2048,"type model\\atlas_combined_prompt.txt");
   system(cmd);
 }
 return 0;
}
