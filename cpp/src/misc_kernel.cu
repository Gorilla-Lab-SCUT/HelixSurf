// Copyright 2021 Alex Yu

#include "cuda_utils.cuh"
#include "misc_kernel.cuh"

const int32_t MISC_CUDA_THREADS = 256;

// kernel
__global__ void accel_dist_set_kernel(
        const torch::PackedTensorAccessor32<int8_t, 3, torch::RestrictPtrTraits> grid,
        bool *__restrict__ tmp) {
    int32_t sz_x = grid.size(0);
    int32_t sz_y = grid.size(1);
    int32_t sz_z = grid.size(2);
    CUDA_GET_THREAD_ID(tid, sz_x * sz_y * sz_z);

    int32_t z = tid % grid.size(2);
    const int32_t xy = tid / grid.size(2);
    int32_t y = xy % grid.size(1);
    int32_t x = xy / grid.size(1);

    bool *tmp_base = tmp;

    if (grid[x][y][z] >= 0) {
        while (sz_x > 1 && sz_y > 1 && sz_z > 1) {
            // Propagate occupied cell throughout the temp tree parent nodes
            x >>= 1;
            y >>= 1;
            z >>= 1;
            sz_x = int_div2_ceil(sz_x);
            sz_y = int_div2_ceil(sz_y);
            sz_z = int_div2_ceil(sz_z);

            const int32_t idx = x * sz_y * sz_z + y * sz_z + z;
            tmp_base[idx] = true;
            tmp_base += sz_x * sz_y * sz_z;
        }
    }
}

__global__ void accel_dist_prop_kernel(
        torch::PackedTensorAccessor32<int8_t, 3, torch::RestrictPtrTraits> grid,
        const bool *__restrict__ tmp) {
    int32_t sz_x = grid.size(0);
    int32_t sz_y = grid.size(1);
    int32_t sz_z = grid.size(2);
    CUDA_GET_THREAD_ID(tid, sz_x * sz_y * sz_z);

    int32_t z = tid % grid.size(2);
    const int32_t xy = tid / grid.size(2);
    int32_t y = xy % grid.size(1);
    int32_t x = xy / grid.size(1);
    const bool *tmp_base = tmp;
    int8_t *__restrict__ val = &grid[x][y][z];

    if (*val < 0) {
        int8_t result = -1;
        while (sz_x > 1 && sz_y > 1 && sz_z > 1) {
            // Find the lowest set parent cell if it exists
            x >>= 1;
            y >>= 1;
            z >>= 1;
            sz_x = int_div2_ceil(sz_x);
            sz_y = int_div2_ceil(sz_y);
            sz_z = int_div2_ceil(sz_z);

            const int32_t idx = x * sz_y * sz_z + y * sz_z + z;
            if (tmp_base[idx]) {
                break;
            }
            result -= 1;
            tmp_base += sz_x * sz_y * sz_z;
        }
        *val = result;
    }
}

// wrapper
void misc::accel_dist_prop(Tensor grid) {
    // Grid here is links array from the sparse grid
    DEVICE_GUARD(grid);
    CHECK_INPUT(grid);
    TORCH_CHECK(!grid.is_floating_point());
    TORCH_CHECK(grid.ndimension() == 3);

    int64_t sz_x = grid.size(0);
    int64_t sz_y = grid.size(1);
    int64_t sz_z = grid.size(2);

    int32_t Q = grid.size(0) * grid.size(1) * grid.size(2);

    const int32_t blocks = CUDA_N_BLOCKS_NEEDED(Q, MISC_CUDA_THREADS);

    int64_t req_size = 0;
    while (sz_x > 1 && sz_y > 1 && sz_z > 1) {
        sz_x = int_div2_ceil(sz_x);
        sz_y = int_div2_ceil(sz_y);
        sz_z = int_div2_ceil(sz_z);
        req_size += sz_x * sz_y * sz_z;
    }

    auto tmp_options = torch::TensorOptions()
                               .dtype(torch::kBool)
                               .layout(torch::kStrided)
                               .device(grid.device())
                               .requires_grad(false);
    Tensor tmp = torch::zeros({req_size}, tmp_options);
    accel_dist_set_kernel<<<blocks, MISC_CUDA_THREADS>>>(
            grid.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>(), tmp.data_ptr<bool>());

    accel_dist_prop_kernel<<<blocks, MISC_CUDA_THREADS>>>(
            grid.packed_accessor32<int8_t, 3, torch::RestrictPtrTraits>(), tmp.data_ptr<bool>());

    // cuda(Free(tmp));
    CUDA_CHECK_ERRORS;
}
