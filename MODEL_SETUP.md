# Local model setup

Atlas runs GGUF model files locally. The executable and GUI are written in C; model weights remain separate files with their own licenses.

## Built-in models

Place the project models in `model/` using these filenames:

- `atlas-llm-1.0-language.gguf` — default conversational coordinator
- `atlas-llm-1.0-compact.gguf` — compact coding model
- `atlas-llm-1.0.gguf` — optional 1.5B reference model, enabled by default

The specialist loader scans `model/specialists/` for supported Qwen2 and Qwen3 GGUF files, reads their dimensions and tensor types from model metadata, and loads one specialist at a time. Additional compatible files do not require recompilation. The inference backend supports these architectures and the quantization formats listed in README.md; arbitrary GGUF architectures are not runnable just because their files are discoverable.

## Larger specialists and speed

Specialists are invoked serially, so Atlas does not keep all specialist weights resident together. Larger specialists can answer harder subtasks, but loading and generating with them costs more time and memory. The bundled Qwen3 8B Q4_K_M fallback is CPU-compatible but may be slow on machines without enough GPU memory; routing should leave it as a late fallback. Compare answer quality and latency before adding other larger specialists, and remember that model size does not guarantee accuracy.

Model weights may have separate redistribution and commercial-use terms. Confirm each model's license before packaging or selling Atlas with the weights.
