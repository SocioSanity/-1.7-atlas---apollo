@echo off
specialist_loader.exe "model/specialists/atlas_tiny_135m_router.gguf" "%1" 64
atlas_translator.exe "model/specialists/atlas_tiny_135m_router.gguf" "model/atlas-llm-1.0-language.gguf" "%1" model\atlas_combined.gguf
type model\atlas_combined_prompt.txt
