#ifndef BRAIN_GPU_MATVEC_H
#define BRAIN_GPU_MATVEC_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t offset, rows, cols;
    uint32_t type;
    const float *input;
    float *output;
} BrainGpuMatvec;

int brain_gpu_init(const uint8_t *model, size_t model_size);
int brain_gpu_matvec_batch(const BrainGpuMatvec *ops, size_t count);
int brain_gpu_matvec(uint64_t offset, uint32_t type, uint64_t rows,
                     uint64_t cols, const float *input, float *output);
void brain_gpu_shutdown(void);

#endif
