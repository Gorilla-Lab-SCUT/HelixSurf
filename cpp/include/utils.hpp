#pragma once

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) \
    CHECK_CPU(x);          \
    CHECK_CONTIGUOUS(x)

#if defined(__CUDACC__)

#else

#endif
#define _SQR(x) ((x) * (x))
