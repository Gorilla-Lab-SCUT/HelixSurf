// Copyright 2022 Gorilla-Lab

#include <vector_types.h>
#include "cuda_utils.cuh"
#include "sliding_window.cuh"

__global__ void sliding_window_normal_kernel(const bool* __restrict__ planar_region_ptr,
                                             const float* __restrict__ input_normal_ptr,
                                             const int32_t height,
                                             const int32_t width,
                                             const int32_t window_size,
                                             const int32_t thresh,
                                             // output
                                             float* __restrict__ output_normal_ptr) {
    const int32_t pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_id >= height*width){
        return;
    }
    const int32_t row = pixel_id / width;
    const int32_t col = pixel_id % width;

    const int32_t half_size = window_size / 2;
    // judge the planar region
    if (!planar_region_ptr[row * width + col]) {
        output_normal_ptr[(row * width + col) * 3 + 0] = 0.f;
        output_normal_ptr[(row * width + col) * 3 + 1] = 0.f;
        output_normal_ptr[(row * width + col) * 3 + 2] = 0.f;
        return;
    }
    // sliding windows
    int32_t valid_count = 0;
    float3 n = make_float3(0.f, 0.f, 0.f);
    for (int32_t r = row - half_size; r < row + half_size + 1; ++r) {
        for (int32_t c = col - half_size; c < col + half_size + 1; ++c) {
            if (r < 0 || r >= height || c < 0 || c >= width) {
                continue;
            }
            if (!planar_region_ptr[r * width + c]) {
                continue;
            }
            ++valid_count;
            n.x += input_normal_ptr[(r * width + c) * 3 + 0];
            n.y += input_normal_ptr[(r * width + c) * 3 + 1];
            n.z += input_normal_ptr[(r * width + c) * 3 + 2];
        }
    }

    n.x /= valid_count;
    n.y /= valid_count;
    n.z /= valid_count;
    __syncthreads();

    // enough valid normal
    output_normal_ptr[(row * width + col) * 3 + 0] = n.x;
    output_normal_ptr[(row * width + col) * 3 + 1] = n.y;
    output_normal_ptr[(row * width + col) * 3 + 2] = n.z;
}

__global__ void sliding_window_normal_with_primitive_kernel(
        const int32_t* __restrict__ primitive_ids_map_ptr,
        const float* __restrict__ input_normal_ptr,
        const int32_t height,
        const int32_t width,
        const int32_t window_size,
        const int32_t thresh,
        // output
        float* __restrict__ output_normal_ptr,
        float* __restrict__ primitives_normals,
        int32_t* __restrict__ primitives_counts) {
    const int32_t pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_id >= height*width){
        return;
    }
    const int32_t row = pixel_id / width;
    const int32_t col = pixel_id % width;


    const int32_t half_size = window_size / 2;
    // judge the planar region
    const int32_t primitive_id = primitive_ids_map_ptr[row * width + col];
    if (primitive_id == 0) {
        output_normal_ptr[(row * width + col) * 3 + 0] = 0.f;
        output_normal_ptr[(row * width + col) * 3 + 1] = 0.f;
        output_normal_ptr[(row * width + col) * 3 + 2] = 0.f;
        return;
    }
    // sliding windows
    int32_t valid_count = 0;
    float3 n = make_float3(0.f, 0.f, 0.f);
    for (int32_t r = row - half_size; r < row + half_size + 1; ++r) {
        for (int32_t c = col - half_size; c < col + half_size + 1; ++c) {
            if (r < 0 || r >= height || c < 0 || c >= width) {
                continue;
            }
            if (primitive_ids_map_ptr[r * width + c] == 0) {
                continue;
            }
            ++valid_count;
            n.x += input_normal_ptr[(r * width + c) * 3 + 0];
            n.y += input_normal_ptr[(r * width + c) * 3 + 1];
            n.z += input_normal_ptr[(r * width + c) * 3 + 2];
        }
    }
    n.x /= valid_count;
    n.y /= valid_count;
    n.z /= valid_count;

    __syncthreads();
    atomicAdd(&primitives_counts[primitive_id - 1], 1);
    atomicAdd(&primitives_normals[(primitive_id - 1) * 3 + 0], n.x);
    atomicAdd(&primitives_normals[(primitive_id - 1) * 3 + 1], n.y);
    atomicAdd(&primitives_normals[(primitive_id - 1) * 3 + 2], n.z);

    // enough valid normal
    output_normal_ptr[(row * width + col) * 3 + 0] = n.x;
    output_normal_ptr[(row * width + col) * 3 + 1] = n.y;
    output_normal_ptr[(row * width + col) * 3 + 2] = n.z;
}

__global__ void count_primitive_kernel(const int32_t* __restrict__ primitive_ids_map_ptr,
                                       const float* __restrict__ input_normal_ptr,
                                       const int32_t height,
                                       const int32_t width,
                                       // output
                                       float* __restrict__ primitives_normals,
                                       int32_t* __restrict__ primitives_counts) {
    const int32_t pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_id >= height*width){
        return;
    }
    const int32_t row = pixel_id / width;
    const int32_t col = pixel_id % width;

    // judge the planar region
    const int32_t primitive_id = primitive_ids_map_ptr[row * width + col];
    if (primitive_id == 0) {
        return;
    }
    float3 n = make_float3(0.f, 0.f, 0.f);
    n.x = input_normal_ptr[(row * width + col) * 3 + 0];
    n.y = input_normal_ptr[(row * width + col) * 3 + 1];
    n.z = input_normal_ptr[(row * width + col) * 3 + 2];

    __syncthreads();
    atomicAdd(&primitives_counts[primitive_id - 1], 1);
    atomicAdd(&primitives_normals[(primitive_id - 1) * 3 + 0], n.x);
    atomicAdd(&primitives_normals[(primitive_id - 1) * 3 + 1], n.y);
    atomicAdd(&primitives_normals[(primitive_id - 1) * 3 + 2], n.z);
}
// End kernel

Tensor sliding_window::sliding_window_normal(const Tensor& planar_region,
                                             const Tensor& normal,
                                             const int32_t window_size,
                                             const int32_t thresh) {
    CHECK_CPU_INPUT(planar_region);
    CHECK_CPU_INPUT(normal);
    Tensor smooth_normal = torch::zeros_like(normal);

    const int32_t height = normal.size(0);
    const int32_t width = normal.size(1);

    // cuda pointer
    bool* cuda_planar_region_ptr = NULL;
    float* cuda_normal_ptr = NULL;
    float* cuda_smooth_normal_ptr = NULL;
    size_t normal_size = height * width * 3 * sizeof(float);
    cudaMalloc((void**)&cuda_planar_region_ptr, height * width * sizeof(bool));
    cudaMalloc((void**)&cuda_normal_ptr, normal_size);
    cudaMalloc((void**)&cuda_smooth_normal_ptr, normal_size);
    cudaMemcpy(cuda_planar_region_ptr, planar_region.data_ptr<bool>(),
               height * width * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_normal_ptr, normal.data_ptr<float>(), normal_size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    int32_t num_threads = 256;
    int32_t num_blocks = (height * width - 1) / num_threads + 1;
    if ((height<1024) && (width<1024)){
        num_threads = width;
        num_blocks = height;
    }
    sliding_window_normal_kernel<<<num_blocks, num_threads>>>(cuda_planar_region_ptr, cuda_normal_ptr, height,
                                                    width, window_size, thresh,
                                                    cuda_smooth_normal_ptr);
    cudaDeviceSynchronize();

    cudaMemcpy(smooth_normal.data_ptr<float>(), cuda_smooth_normal_ptr, normal_size,
               cudaMemcpyDeviceToHost);
    cudaFree(cuda_planar_region_ptr);
    cudaFree(cuda_normal_ptr);
    cudaFree(cuda_smooth_normal_ptr);
    // return normal;
    return smooth_normal;
}

Tensor sliding_window::sliding_window_normal_cu(const Tensor& planar_region,
                                                const Tensor& normal,
                                                const int32_t window_size,
                                                const int32_t thresh) {
    CHECK_INPUT(planar_region);
    CHECK_INPUT(normal);
    Tensor smooth_normal = torch::zeros_like(normal);

    const int32_t height = normal.size(0);
    const int32_t width = normal.size(1);

    // cuda pointer
    cudaDeviceSynchronize();

    int32_t num_threads = 256;
    int32_t num_blocks = (height * width - 1) / num_threads + 1;
    if ((height<1024) && (width<1024)){
        num_threads = width;
        num_blocks = height;
    }
    sliding_window_normal_kernel<<<num_blocks, num_threads>>>(
        planar_region.data_ptr<bool>(), 
        normal.data_ptr<float>(), 
        height, width, window_size, thresh,
        smooth_normal.data_ptr<float>());
    cudaDeviceSynchronize();

    // return normal;
    return smooth_normal;
}

Tensor sliding_window::sliding_window_normal_with_primitive(const Tensor& primitive_ids_map,
                                                            const Tensor& normal,
                                                            Tensor& primitives_normals,
                                                            Tensor& primitives_counts,
                                                            const int32_t window_size,
                                                            const int32_t thresh) {
    CHECK_CPU_INPUT(primitive_ids_map);
    CHECK_CPU_INPUT(normal);
    CHECK_INPUT(primitives_normals);
    CHECK_INPUT(primitives_counts);
    Tensor smooth_normal = torch::zeros_like(normal);

    const int32_t height = normal.size(0);
    const int32_t width = normal.size(1);

    // cuda pointer
    int32_t* cuda_primitive_ids_map_ptr = NULL;
    float* cuda_normal_ptr = NULL;
    float* cuda_smooth_normal_ptr = NULL;
    size_t normal_size = height * width * 3 * sizeof(float);
    cudaMalloc((void**)&cuda_primitive_ids_map_ptr, height * width * sizeof(int32_t));
    cudaMalloc((void**)&cuda_normal_ptr, normal_size);
    cudaMalloc((void**)&cuda_smooth_normal_ptr, normal_size);
    cudaMemcpy(cuda_primitive_ids_map_ptr, primitive_ids_map.data_ptr<int32_t>(),
               height * width * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_normal_ptr, normal.data_ptr<float>(), normal_size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    int32_t num_threads = 256;
    int32_t num_blocks = (height * width - 1) / num_threads + 1;
    if ((height<1024) && (width<1024)){
        num_threads = width;
        num_blocks = height;
    }
    sliding_window_normal_with_primitive_kernel<<<num_blocks, num_threads>>>(
            cuda_primitive_ids_map_ptr, cuda_normal_ptr, height, width, window_size, thresh,
            // output
            cuda_smooth_normal_ptr, primitives_normals.data_ptr<float>(),
            primitives_counts.data_ptr<int32_t>());
    cudaDeviceSynchronize();

    cudaMemcpy(smooth_normal.data_ptr<float>(), cuda_smooth_normal_ptr, normal_size,
               cudaMemcpyDeviceToHost);
    cudaFree(cuda_primitive_ids_map_ptr);
    cudaFree(cuda_normal_ptr);
    cudaFree(cuda_smooth_normal_ptr);

    // return normal;
    return smooth_normal;
}

void sliding_window::count_primitives(const Tensor& primitive_ids_map,
                                      const Tensor& normal,
                                      Tensor& primitives_normals,
                                      Tensor& primitives_counts) {
    CHECK_CPU_INPUT(primitive_ids_map);
    CHECK_CPU_INPUT(normal);
    CHECK_INPUT(primitives_normals);
    CHECK_INPUT(primitives_counts);

    const int32_t height = normal.size(0);
    const int32_t width = normal.size(1);

    // cuda pointer
    int32_t* cuda_primitive_ids_map_ptr = NULL;
    float* cuda_normal_ptr = NULL;
    size_t normal_size = height * width * 3 * sizeof(float);
    cudaMalloc((void**)&cuda_primitive_ids_map_ptr, height * width * sizeof(int32_t));
    cudaMalloc((void**)&cuda_normal_ptr, normal_size);
    cudaMemcpy(cuda_primitive_ids_map_ptr, primitive_ids_map.data_ptr<int32_t>(),
               height * width * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_normal_ptr, normal.data_ptr<float>(), normal_size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    int32_t num_threads = 256;
    int32_t num_blocks = (height * width - 1) / num_threads + 1;
    if ((height<1024) && (width<1024)){
        num_threads = width;
        num_blocks = height;
    }
    count_primitive_kernel<<<num_blocks, num_threads>>>(
            cuda_primitive_ids_map_ptr, cuda_normal_ptr, height, width,
            // output
            primitives_normals.data_ptr<float>(), primitives_counts.data_ptr<int32_t>());
    cudaDeviceSynchronize();

    cudaFree(cuda_primitive_ids_map_ptr);
    cudaFree(cuda_normal_ptr);
}

void sliding_window::count_primitives_cu(const Tensor& primitive_ids_map,
                                        const Tensor& normal,
                                        Tensor& primitives_normals,
                                        Tensor& primitives_counts) {
    CHECK_INPUT(primitive_ids_map);
    CHECK_INPUT(normal);
    CHECK_INPUT(primitives_normals);
    CHECK_INPUT(primitives_counts);

    const int32_t height = normal.size(0);
    const int32_t width = normal.size(1);

    cudaDeviceSynchronize();
    int32_t num_threads = 256;
    int32_t num_blocks = (height * width - 1) / num_threads + 1;
    if ((height<1024) && (width<1024)){
        num_threads = width;
        num_blocks = height;
    }
    count_primitive_kernel<<<num_blocks, num_threads>>>(
            primitive_ids_map.data_ptr<int32_t>(),
            normal.data_ptr<float>(), 
            height, width,
            // output
            primitives_normals.data_ptr<float>(), 
            primitives_counts.data_ptr<int32_t>());
    cudaDeviceSynchronize();
}
