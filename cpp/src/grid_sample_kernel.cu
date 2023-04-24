// Copyright 2022 Gorilla-Lab

#include "cuda_utils.cuh"
#include "grid_sample_kernel.cuh"

#define MAX_PATCH_SIZE 32

// kernel
__global__ void grid_sample_kernel(
        const float* __restrict__ images_ptr,         // [n_img, H, W, 3]
        const float* __restrict__ grid_ptr,           // [n_patch, n_ref, ps * ps, 2]
        const int32_t* __restrict__ img_ids_src_ptr,  // [n_patch, n_ref]
        const int32_t H,
        const int32_t W,
        const int32_t n_ref,
        const int32_t n_patch,
        const int32_t ps,
        // output: store the not empty points for each ray
        float* __restrict__ warped_patch_ptr  // [n_ref, n_patch, ps * ps, 3]
) {
    const int32_t ref_idx = blockIdx.x, patch_idx = blockIdx.y;
    const int32_t row = threadIdx.x, col = threadIdx.y;
    const int32_t ps2 = ps * ps;

    // get the img_id and uv for query
    const int32_t img_id = img_ids_src_ptr[patch_idx * n_ref + ref_idx];
    const float u =
            grid_ptr[patch_idx * n_ref * ps2 * 2 + ref_idx * ps2 * 2 + row * ps * 2 + col * 2 + 0];
    const float v =
            grid_ptr[patch_idx * n_ref * ps2 * 2 + ref_idx * ps2 * 2 + row * ps * 2 + col * 2 + 1];

    if (u < 0 || u > W - 1 || v < 0 || v > H - 1) {
        return;
    }
    /* illustration for bilinear interpolation
      0 ----- 1
      |       |
      |       |
      2 ----- 3
     */
    const int32_t u0 = static_cast<int32_t>(u), v0 = static_cast<int32_t>(v);
    const float rel_u = u - static_cast<float>(u0);  // W
    const float rel_v = v - static_cast<float>(v0);  // H
    // assert(rel_u >= 0 && rel_u <= 1);
    // assert(rel_v >= 0 && rel_v <= 1);
    // assert(u0 >= 0 && u0 <= W - 1);
    // assert(v0 >= 0 && v0 <= H - 1);

    const float* img_ptr = images_ptr + img_id * H * W * 3;
    float* patch_ptr = warped_patch_ptr + ref_idx * n_patch * ps2 * 3 + patch_idx * ps2 * 3;

    for (int32_t c_idx = 0; c_idx < 3; ++c_idx) {
        const float c0 = img_ptr[v0 * W * 3 + u0 * 3 + c_idx];
        const float c1 = img_ptr[v0 * W * 3 + (u0 + 1) * 3 + c_idx];
        const float c2 = img_ptr[(v0 + 1) * W * 3 + u0 * 3 + c_idx];
        const float c3 = img_ptr[(v0 + 1) * W * 3 + (u0 + 1) * 3 + c_idx];
        // bilinear interpolation
        const float ret = lerp(lerp(c0, c1, rel_u), lerp(c2, c3, rel_u), rel_v);
        patch_ptr[(row * ps + col) * 3 + c_idx] = ret;
    }
}

__global__ void sample_patch_kernel(const int32_t* __restrict__ sample_coords_ptr,  // [n_ref, 3]
                                    const float* __restrict__ images_ptr,  // [n_img, H, W, 3]
                                    const int32_t H,
                                    const int32_t W,
                                    const int32_t ps,
                                    // output: store the not empty points for each ray
                                    float* __restrict__ patch_ptr  // [n_ref, ps * ps, 3]
) {
    const int32_t ref_idx = blockIdx.x, c_idx = blockIdx.y;
    const int32_t row = threadIdx.x, col = threadIdx.y;
    const int32_t ps2 = ps * ps;

    const int32_t img_id = sample_coords_ptr[ref_idx * 3],
                  h_idx = sample_coords_ptr[ref_idx * 3 + 1] + row - ps / 2,
                  w_idx = sample_coords_ptr[ref_idx * 3 + 2] + col - ps / 2;

    if (h_idx < 0 || h_idx > H - 1 || w_idx < 0 || w_idx > W - 1) {
        assert(false);
        return;
    }

    const float* img_ptr = images_ptr + img_id * H * W * 3;  // [H, W, 3]

    const float ret = img_ptr[(h_idx * W + w_idx) * 3 + c_idx];

    patch_ptr[ref_idx * ps2 * 3 + (row * ps + col) * 3 + c_idx] = ret;
}

// wrapper

Tensor sampling::grid_sample(const Tensor& images,       // [n_img, H, W, 3]
                             const Tensor& grid,         // [n_patch, n_ref, ps * ps, 2]
                             const Tensor& img_ids_src,  // [n_patch, n_ref]
                             const int32_t patch_size) {
    // check
    CHECK_INPUT(images);
    CHECK_INPUT(grid);
    CHECK_INPUT(img_ids_src);

    // NOTE: avoid exceed the maximum of threads
    assert(patch_size < MAX_PATCH_SIZE);

    const int32_t H = images.size(1), W = images.size(2);
    const int32_t n_patch = grid.size(0), n_ref = grid.size(1);

    Tensor warped_patch =
            torch::zeros({n_ref, n_patch, patch_size * patch_size, 3},
                         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    dim3 block_dim(n_ref, n_patch), thread_dim(patch_size, patch_size);

    grid_sample_kernel<<<block_dim, thread_dim>>>(images.data_ptr<float>(), grid.data_ptr<float>(),
                                                  img_ids_src.data_ptr<int32_t>(), H, W, n_ref,
                                                  n_patch, patch_size,
                                                  // output
                                                  warped_patch.data_ptr<float>());

    return warped_patch;
}

Tensor sampling::sample_patch(const Tensor& sample_coords,  // [n_ref, 3]
                              const Tensor& images,         // [n_img, H, W, 3]
                              const int32_t patch_size) {
    // check
    CHECK_INPUT(sample_coords);
    CHECK_INPUT(images);

    // NOTE: avoid exceed the maximum of threads
    assert(patch_size < MAX_PATCH_SIZE);

    const int32_t H = images.size(1), W = images.size(2);
    const int32_t n_ref = sample_coords.size(0);

    Tensor patch = torch::zeros({n_ref, patch_size * patch_size, 3},
                                torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    dim3 block_dim(n_ref, 3), thread_dim(patch_size, patch_size);

    sample_patch_kernel<<<block_dim, thread_dim>>>(sample_coords.data_ptr<int32_t>(),
                                                   images.data_ptr<float>(), H, W, patch_size,
                                                   // output
                                                   patch.data_ptr<float>());

    return patch;
}
