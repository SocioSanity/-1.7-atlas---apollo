# Third-party notices

## Qwen2.5 model files

The repository's local GGUF files are converted or quantized Qwen2.5-family checkpoints, including language, coder, math, and instruct variants. Their upstream family is published by Alibaba Cloud under Apache License 2.0. The original upstream model terms and notices apply to those weights; a copy of Apache 2.0 is in `LICENSES/Apache-2.0.txt`.

Upstream model family: <https://huggingface.co/Qwen>

GGUF conversion and quantization can have separate contributors and terms. Before redistributing any model file, record the exact source repository, revision, converter, and quantization provenance, and confirm its redistribution terms. Model weights are excluded from source archives by `.gitignore`.

## Qwen3 8B local fallback

`model/specialists/qwen/qwen3-8b-q4_k_m.gguf` is a locally available Qwen3 8B Q4_K_M GGUF model. The local Ollama manifest labels the source checkpoint `qwen3:8b`; its `qwen3:8b-tools` alias points to the same weight blob and is not a second model. The upstream Qwen3 model is published by Alibaba Cloud under Apache License 2.0; see `LICENSES/Qwen3-Apache-2.0.txt`. This file records the locally observed source and quantization label; it is not a substitute for exact converter/revision provenance when redistributing the weights.

Upstream model family: <https://huggingface.co/Qwen/Qwen3-8B>

## Other components

The C application uses Windows APIs and Direct3D 12. Record any additional third-party source or library and its license here before redistributing a build that bundles it. This file does not replace notices required by a dependency or model author.
