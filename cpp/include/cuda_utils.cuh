// Copyright 2021 Alex Yu
#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "utils.hpp"

#define DEVICE_GUARD(_ten) const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define CUDA_GET_THREAD_ID(tid, Q)                             \
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (tid >= Q) return
#define CUDA_GET_THREAD_ID_U64(tid, Q)                        \
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (tid >= Q) return
#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)

#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CUDA_CHECK_THROW(x)                                                                 \
    do {                                                                                    \
        cudaError_t result = x;                                                             \
        if (result != cudaSuccess)                                                          \
            throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error: ") + \
                                     cudaGetErrorString(result));                           \
    } while (0)

#define CUDA_CHECK_ERRORS                 \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess)               \
    printf("Error in gnerf.%s : %s\n", __FUNCTION__, cudaGetErrorString(err))

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ inline double atomicAdd(double* address, double val) {
    unsigned long long int32_t* address_as_ull = (unsigned long long int32_t*)address;
    unsigned long long int32_t old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__ inline void atomicMax(float* result, float value) {
    unsigned* result_as_u = (unsigned*)result;
    unsigned old = *result_as_u, assumed;
    do {
        assumed = old;
        old = atomicCAS(result_as_u, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (old != assumed);
    return;
}

__device__ inline void atomicMax(double* result, double value) {
    unsigned long long int* result_as_ull = (unsigned long long int*)result;
    unsigned long long int old = *result_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(result_as_ull, assumed,
                        __double_as_longlong(fmaxf(value, __longlong_as_double(assumed))));
    } while (old != assumed);
    return;
}

__device__ __inline__ void transform_coord(float* __restrict__ point,
                                           const float* __restrict__ scaling,
                                           const float* __restrict__ offset) {
    point[0] = fmaf(point[0], scaling[0], offset[0]);  // a*b + c
    point[1] = fmaf(point[1], scaling[1], offset[1]);  // a*b + c
    point[2] = fmaf(point[2], scaling[2], offset[2]);  // a*b + c
}

template <typename T>
__host__ __device__ __inline__ T clamp(T val, T min, T max) {
    return fminf(fmaxf(val, min), max);
}

// Linear interp
// Subtract and fused multiply-add
// (1-w) a + w b
template <typename T>
__host__ __device__ __inline__ T lerp(T a, T b, T w) {
    return fmaf(w, b - a, a);
}

__device__ __inline__ static float _norm(const float* __restrict__ dir) {
    // return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    return norm3df(dir[0], dir[1], dir[2]);
}

__device__ __inline__ static float _rnorm(const float* __restrict__ dir) {
    // return 1.f / _norm(dir);
    return rnorm3df(dir[0], dir[1], dir[2]);
}

__host__ __device__ __inline__ static float _dot(const float* __restrict__ x,
                                                 const float* __restrict__ y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

__host__ __device__ __inline__ static void _cross(const float* __restrict__ a,
                                                  const float* __restrict__ b,
                                                  float* __restrict__ out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

#define int_div2_ceil(x) ((((x)-1) >> 1) + 1)
