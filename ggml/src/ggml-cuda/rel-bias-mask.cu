#include "rel-bias-mask.cuh"
#include "common.cuh"

// fused: gather the relative bias table by distance and add the causal mask,
// output in the per-head attention mask layout (replaces the get_rows + reshape +
// permute + cont + add + cast chain without the intermediate n_kv x n_tokens x n_head buffer)
//
// dst[j, i, h, s] = table[h, rel_pos[j, i+s*n_i], i+s*n_i, 0] + mask[j, i, 0, s]
// (the token index is t = i + s*n_i, matching reshape_4d({n_head,n_kv,nt,1} -> {n_head,n_kv,ni,ns}))
//
// table   F32   {n_head, n_rel, nt, 1}
// rel_pos I32   {n_kv, nt, 1, 1}
// mask    F16|F32 {n_kv, nt/ns, 1, ns}
// dst             {n_kv, nt/ns, n_head, ns}  (type of mask)

template<typename T>
static __global__ void k_rel_bias_mask(
        const float * __restrict__ table,
        const int32_t * __restrict__ rel_pos,
        const T * __restrict__ mask,
        T * __restrict__ dst,
        const int64_t n_kv, const int64_t n_i, const int64_t n_head, const int64_t ns,
        const int64_t nt, const int64_t n_rel) {
    const int64_t idx = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t total = n_kv*n_i*n_head*ns;
    if (idx >= total) {
        return;
    }

    // ggml contiguous layout (dim 0 fastest):
    //   dst     {n_kv,n_i,n_head,ns}: off(j,i,h,s) = j + i*n_kv + h*(n_kv*n_i) + s*(n_kv*n_i*n_head)
    //   rel_pos {n_kv,nt,1,1}:        off(j,t)     = j + t*n_kv
    //   table   {n_head,n_rel,nt,1}:  off(h,d,t)   = h + d*n_head + t*(n_head*n_rel)
    //   mask    {n_kv,n_i,1,ns}:      off(j,i,s)   = j + i*n_kv + s*(n_kv*n_i)
    const int64_t j = idx % n_kv;
    const int64_t i = (idx / n_kv) % n_i;
    const int64_t h = (idx / (n_kv*n_i)) % n_head;
    const int64_t s = idx / (n_kv*n_i*n_head);

    // reshape_4d({n_head,n_kv,nt,1} -> {n_head,n_kv,ni,ns}) keeps nb[0] and re-derives the rest,
    // so the token index is t = i + s*n_i (not i*ns + s)
    const int64_t t = i + s*n_i;
    const int32_t d = rel_pos[j + t*n_kv];
    const float bias = table[h + d*n_head + t*(n_head*n_rel)];
    const float mval = (float) mask[j + i*n_kv + s*(n_kv*n_i)];
    dst[idx] = (T)(bias + mval);
}

void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table   = dst->src[0];
    const ggml_tensor * rel_pos = dst->src[1];
    const ggml_tensor * mask    = dst->src[2];

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(table->type == GGML_TYPE_F32);
    GGML_ASSERT(rel_pos->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == mask->type);

    const int64_t n_kv   = dst->ne[0];
    const int64_t n_i    = dst->ne[1];
    const int64_t n_head = dst->ne[2];
    const int64_t ns     = dst->ne[3];
    const int64_t nt     = rel_pos->ne[1];
    const int64_t n_rel  = table->ne[1];

    const int64_t total = n_kv*n_i*n_head*ns;
    const int block_size = CUDA_REL_BIAS_MASK_BLOCK_SIZE;
    const int n_blocks = (int) ((total + block_size - 1) / block_size);

    switch (dst->type) {
        case GGML_TYPE_F16:
            k_rel_bias_mask<half><<<n_blocks, block_size, 0, stream>>>(
                (const float *) table->data, (const int32_t *) rel_pos->data, (const half *) mask->data, (half *) dst->data,
                n_kv, n_i, n_head, ns, nt, n_rel);
            break;
        case GGML_TYPE_F32:
            k_rel_bias_mask<float><<<n_blocks, block_size, 0, stream>>>(
                (const float *) table->data, (const int32_t *) rel_pos->data, (const float *) mask->data, (float *) dst->data,
                n_kv, n_i, n_head, ns, nt, n_rel);
            break;
        default:
            GGML_ABORT("%s: unsupported type %s\n", __func__, ggml_type_name(dst->type));
    }
}
