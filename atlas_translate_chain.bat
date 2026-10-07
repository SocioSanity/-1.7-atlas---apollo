@echo off
echo [1/3] Specialist loader...
specialist_loader.exe "model\specialists\atlas_tiny_135m_router.gguf" "%~1" 64
echo [2/3] Translator...
atlas_translator.exe "model\specialists\atlas_tiny_135m_router.gguf" "model\atlas-llm-1.0-language.gguf" "%~1"
echo [3/3] Combined output for Atlas 2.5/1.5:
type model\atlas_combined_prompt.txt
if exist atlas_2.5_infer.exe atlas_2.5_infer.exe model\atlas-llm-1.0-language.gguf "%~1" 128
if exist atlas_1.5_infer.exe atlas_1.5_infer.exe model\atlas-llm-1.0-language.gguf "%~1" 128
