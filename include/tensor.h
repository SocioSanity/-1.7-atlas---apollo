#ifndef VM_AI_TENSOR_H
#define VM_AI_TENSOR_H

#include <stdint.h>

#define VM_AI_QK_K 256

int vm_ai_q4k_matvec(
    const uint8_t *weights,
    uint64_t rows,
    uint64_t cols,
    const float *input,
    float *output
);

int vm_ai_q6k_matvec(
    const uint8_t *weights,
    uint64_t rows,
    uint64_t cols,
    const float *input,
    float *output
);

int vm_ai_f32_matvec(
    const uint8_t *weights,
    uint64_t rows,
    uint64_t cols,
    const float *input,
    float *output
);

#endif
