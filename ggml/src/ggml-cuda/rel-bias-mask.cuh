#include "common.cuh"

#define CUDA_REL_BIAS_MASK_BLOCK_SIZE 256

void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
