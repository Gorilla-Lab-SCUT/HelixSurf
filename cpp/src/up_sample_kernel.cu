// Copyright 2022 Gorilla-Lab

#include "cuda_utils.cuh"
#include "data_spec_packed.cuh"
#include "up_sample_kernel.cuh"

// kernels
__global__ void count_diff_kernel(const int32_t* __restrict__ offsets_ptr,  // [n_rays + 1]
                                  const int32_t n_rays,
                                  // output
                                  int32_t* __restrict__ valids_count_ptr  // [n_rays + 1]
) {
    CUDA_GET_THREAD_ID(ray_id, n_rays);

    const int32_t start = offsets_ptr[ray_id], end = offsets_ptr[ray_id + 1];

    if (start == end) {
        return;
    }
    valids_count_ptr[ray_id + 1] = end - start - 1;
}

__global__ void prev_next_diff_kernel(const int32_t* __restrict__ offsets_ptr,
                                      const int32_t* __restrict__ diff_offsets_ptr,
                                      const float* __restrict__ sdf_ptr,
                                      const float* __restrict__ uniform_steps_ptr,
                                      const int32_t n_rays,
                                      // output
                                      float* __restrict__ mid_sdf_ptr,
                                      float* __restrict__ diff_sdf_ptr,
                                      float* __restrict__ dist_ptr,
                                      float* __restrict__ cos_val_ptr) {
    CUDA_GET_THREAD_ID(ray_id, n_rays);

    const int32_t start = offsets_ptr[ray_id], end = offsets_ptr[ray_id + 1];
    const int32_t diff_start = diff_offsets_ptr[ray_id];
    // const int32_t diff_end = diff_offsets_ptr[ray_id + 1];

    float prev_cos_val = 0.f;
    for (int32_t idx = start, diff_idx = diff_start; idx < end - 1; ++idx, ++diff_idx) {
        const float prev_sdf = sdf_ptr[idx], next_sdf = sdf_ptr[idx + 1];
        const float prev_step = uniform_steps_ptr[idx], next_step = uniform_steps_ptr[idx + 1];
        const float mid_sdf = (prev_sdf + next_sdf) * 0.5, diff_sdf = next_sdf - prev_sdf,
                    dist = next_step - prev_step;
        mid_sdf_ptr[diff_idx] = mid_sdf;
        diff_sdf_ptr[diff_idx] = diff_sdf;
        dist_ptr[diff_idx] = dist;
        const float _cos_val = diff_sdf / (dist + 1e-5);
        const float cos_val = clamp(fminf(prev_cos_val, _cos_val), -1e3f, 0.0f);
        prev_cos_val = _cos_val;
        cos_val_ptr[diff_idx] = cos_val;
    }
}

__global__ void count_importance_kernel(
        PackedRaysSpec rays,
        RenderOptions opt,
        const bool* __restrict__ valid_grid_ptr,
        const float* __restrict__ uniform_steps_ptr,      // [uniform_size]
        const float* __restrict__ weights_ptr,            // [uniform_size]
        const int32_t* __restrict__ uniform_offsets_ptr,  // [n_rays + 1]
        const int32_t* __restrict__ diff_offsets_ptr,     // [n_rays + 1]
        const float* __restrict__ u_ptr,                  // [n_importance]
        const int32_t n_importance,
        const int32_t n_rays,
        // output
        int32_t* __restrict__ valids_count_ptr,  // [n_rays + 1]
        int32_t* __restrict__ importance_count_ptr) {
    CUDA_GET_THREAD_ID(ray_id, n_rays);

    const int32_t uniform_start = uniform_offsets_ptr[ray_id],
                  uniform_end = uniform_offsets_ptr[ray_id + 1];
    const int32_t diff_start = diff_offsets_ptr[ray_id], diff_end = diff_offsets_ptr[ray_id + 1];

    if (uniform_start == uniform_end) {
        assert(diff_end == diff_start);
        return;
    }
    assert(uniform_end - uniform_start == diff_end - diff_start + 1);

    // get the ray
    SingleRaySpec ray;
    ray.set(rays.origins[ray_id].data(), rays.dirs[ray_id].data());

    float weigth_accum = 0.f;
    for (int32_t diff_idx = diff_start; diff_idx < diff_end; ++diff_idx) {
        weigth_accum += weights_ptr[diff_idx] + 1e-5;
    }

    // get cdf and perform searchsorted
    float prev_step = 0, curr_step = 0, prev_cdf = 0, curr_cdf = 0;
    int32_t up_sample_idx = 0, valid_count = 0;
    const float size = (2 * opt.bound) / opt.grid_res;  // the size of grid
    const int32_t offx = opt.grid_res * opt.grid_res, offy = opt.grid_res;

    for (int32_t uniform_idx = uniform_start, diff_idx = diff_start; diff_idx < diff_end;
         ++uniform_idx, ++diff_idx) {
        prev_step = uniform_steps_ptr[uniform_idx];
        curr_step = uniform_steps_ptr[uniform_idx + 1];
        const float dist = curr_step - prev_step;
        curr_cdf += (weights_ptr[diff_idx] + 1e-5) / weigth_accum;
        float denom = curr_cdf - prev_cdf;
        denom = denom < 1e-5 ? 1.f : denom;
        while (curr_cdf > u_ptr[up_sample_idx] && up_sample_idx < n_importance) {
            // interpolate the step
            const float bias = u_ptr[up_sample_idx] - prev_cdf;
            const float ratio = bias / denom;
            assert(ratio >= 0 && ratio <= 1);
            const float step = fmaf(ratio, dist, prev_step);
#pragma unroll 3
            for (int32_t j = 0; j < 3; ++j) {
                ray.pos[j] = fmaf(step, ray.dir[j], ray.origin[j]);
                ray.pos[j] = clamp(ray.pos[j], -opt.bound, opt.bound);
                ray.l[j] = min(int32_t((ray.pos[j] + opt.bound) / size), opt.grid_res - 1);
            }
            const bool occupancy = valid_grid_ptr[ray.l[0] * offx + ray.l[1] * offy + ray.l[2]];
            if (occupancy) {
                ++valid_count;
            }
            // update index
            ++up_sample_idx;
        }
        if (up_sample_idx == n_importance) {
            break;
        }
        prev_cdf = curr_cdf;
    }
    assert(valid_count <= n_importance);
    valids_count_ptr[ray_id + 1] = uniform_end - uniform_start + valid_count;
    importance_count_ptr[ray_id + 1] = valid_count;
}

__global__ void importance_sampling_kernel(
        PackedRaysSpec rays,
        RenderOptions opt,
        const bool* __restrict__ valid_grid_ptr,
        const float* __restrict__ uniform_steps_ptr,         // [uniform_size]
        const float* __restrict__ weights_ptr,               // [uniform_size]
        const int32_t* __restrict__ uniform_offsets_ptr,     // [n_rays + 1]
        const int32_t* __restrict__ diff_offsets_ptr,        // [n_rays + 1]
        const int32_t* __restrict__ importance_offsets_ptr,  // [n_rays + 1]
        const float* __restrict__ u_ptr,                     // [n_importance]
        const int32_t n_importance,
        const int32_t n_rays,
        // output
        float* __restrict__ importance_steps_ptr  // [importance_size]
) {
    CUDA_GET_THREAD_ID(ray_id, n_rays);

    const int32_t uniform_start = uniform_offsets_ptr[ray_id],
                  uniform_end = uniform_offsets_ptr[ray_id + 1];
    const int32_t diff_start = diff_offsets_ptr[ray_id], diff_end = diff_offsets_ptr[ray_id + 1];
    const int32_t import_start = importance_offsets_ptr[ray_id];

    if (uniform_start == uniform_end) {
        return;
    }

    // get the ray
    SingleRaySpec ray;
    ray.set(rays.origins[ray_id].data(), rays.dirs[ray_id].data());

    float weigth_accum = 0.f;
    for (int32_t diff_idx = diff_start; diff_idx < diff_end; ++diff_idx) {
        weigth_accum += weights_ptr[diff_idx] + 1e-5;
    }

    // get cdf and perform searchsorted
    float prev_step = 0, curr_step = 0, prev_cdf = 0, curr_cdf = 0;
    int32_t up_sample_idx = 0, valid_count = 0;
    const float size = (2 * opt.bound) / opt.grid_res;  // the size of grid
    const int32_t offx = opt.grid_res * opt.grid_res, offy = opt.grid_res;

    for (int32_t uniform_idx = uniform_start, diff_idx = diff_start; diff_idx < diff_end;
         ++uniform_idx, ++diff_idx) {
        prev_step = uniform_steps_ptr[uniform_idx];
        curr_step = uniform_steps_ptr[uniform_idx + 1];
        const float dist = curr_step - prev_step;
        curr_cdf += (weights_ptr[diff_idx] + 1e-5) / weigth_accum;
        float denom = curr_cdf - prev_cdf;
        denom = denom < 1e-5 ? 1.f : denom;
        while (curr_cdf > u_ptr[up_sample_idx] && up_sample_idx < n_importance) {
            // put the importance samples
            const float bias = u_ptr[up_sample_idx] - prev_cdf;
            const float ratio = bias / denom;
            assert(ratio >= 0 && ratio <= 1);
            const float step = fmaf(ratio, dist, prev_step);
#pragma unroll 3
            for (int32_t j = 0; j < 3; ++j) {
                ray.pos[j] = fmaf(step, ray.dir[j], ray.origin[j]);
                ray.pos[j] = clamp(ray.pos[j], -opt.bound, opt.bound);
                ray.l[j] = min(int32_t((ray.pos[j] + opt.bound) / size), opt.grid_res - 1);
            }
            const bool occupancy = valid_grid_ptr[ray.l[0] * offx + ray.l[1] * offy + ray.l[2]];
            if (occupancy) {
                importance_steps_ptr[import_start + valid_count] = step;
                ++valid_count;
            }
            // update index
            ++up_sample_idx;
        }
        if (up_sample_idx == n_importance) {
            break;
        }
        prev_cdf = curr_cdf;
    }
    assert(valid_count <= n_importance);
    // assert(up_sample_idx == n_importance);
}

__global__ void sort_sample_points_kernel(
        PackedRaysSpec rays,
        const float* __restrict__ uniform_steps_ptr,        // [uniform_size]
        const float* __restrict__ importance_steps_ptr,     // [importance_size]
        const int32_t* __restrict__ uniform_offsets_ptr,    // [n_rays + 1]
        const int32_t* __restrict__ importance_offsets,     // [n_rays + 1]
        const int32_t* __restrict__ up_sample_offsets_ptr,  // [n_rays + 1]
        const int32_t n_rays,
        // output
        float* __restrict__ up_sample_steps,     // [up_sample_size]
        float* __restrict__ up_sample_pos_dirs,  // [up_sample_size, 6]
        float* __restrict__ up_sample_deltas     // [up_sample_size]
) {
    CUDA_GET_THREAD_ID(ray_id, n_rays);

    const int32_t uniform_start = uniform_offsets_ptr[ray_id],
                  uniform_end = uniform_offsets_ptr[ray_id + 1];
    const int32_t import_start = importance_offsets[ray_id],
                  import_end = importance_offsets[ray_id + 1];
    const int32_t up_sample_start = up_sample_offsets_ptr[ray_id],
                  up_sample_end = up_sample_offsets_ptr[ray_id + 1];

    if (uniform_start == uniform_end) {
        assert(import_start == import_end);
        return;
    }

    // sort steps
    int32_t uniform_idx = uniform_start, import_idx = import_start, up_sample_idx = up_sample_start;
    while (uniform_idx < uniform_end || import_idx < import_end) {
        float step;
        if (uniform_idx == uniform_end) {
            step = importance_steps_ptr[import_idx];
            ++import_idx;
        } else if (import_idx == import_end) {
            step = uniform_steps_ptr[uniform_idx];
            ++uniform_idx;
        } else {
            const float uniform_step = uniform_steps_ptr[uniform_idx];
            const float import_step = importance_steps_ptr[import_idx];
            if (uniform_step < import_step) {
                step = uniform_step;
                ++uniform_idx;
            } else {
                step = import_step;
                ++import_idx;
            }
        }
        up_sample_steps[up_sample_idx] = step;
        ++up_sample_idx;
    }
    assert(up_sample_idx == up_sample_end);

    // get the ray
    SingleRaySpec ray;
    ray.set(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    // get the pos_dirs and deltas
    for (int32_t idx = up_sample_start; idx < up_sample_end; ++idx) {
        const float step = up_sample_steps[idx];
#pragma unroll 3
        for (int32_t j = 0; j < 3; ++j) {
            up_sample_pos_dirs[idx * 6 + j] = fmaf(step, ray.dir[j], ray.origin[j]);
            up_sample_pos_dirs[idx * 6 + j + 3] = ray.dir[j];
        }
        if (idx == up_sample_end - 1) {
            up_sample_deltas[idx] = step - up_sample_steps[idx - 1];
        } else {
            up_sample_deltas[idx] = up_sample_steps[idx + 1] - step;
        }
    }
}

// wrappers
vector<Tensor> upsample::prev_next_diff(const Tensor& sdf,
                                        const Tensor& steps,
                                        const Tensor& offsets) {
    CHECK_INPUT(sdf);
    CHECK_INPUT(steps);
    CHECK_INPUT(offsets);

    Tensor valids_count = torch::zeros_like(offsets);

    const int32_t n_rays = offsets.size(0) - 1, threads = 1024;
    const int32_t blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);
    count_diff_kernel<<<blocks, threads>>>(offsets.data_ptr<int32_t>(), n_rays,
                                           // output
                                           valids_count.data_ptr<int32_t>());
    Tensor diff_offsets = torch::cumsum(valids_count, 0, torch::kInt);  // [n_rays + 1]
    const int32_t compact_size = diff_offsets.index({-1}).item<int32_t>();
    Tensor mid_sdf = torch::zeros({compact_size},  // [compact_size]
                                  torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor diff_sdf =
            torch::zeros({compact_size},  // [compact_size]
                         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor dist = torch::zeros({compact_size},  // [compact_size]
                               torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor cos_val = torch::zeros({compact_size},  // [compact_size]
                                  torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    prev_next_diff_kernel<<<blocks, threads>>>(
            offsets.data_ptr<int32_t>(), diff_offsets.data_ptr<int32_t>(), sdf.data_ptr<float>(),
            steps.data_ptr<float>(), n_rays,
            // output
            mid_sdf.data_ptr<float>(), diff_sdf.data_ptr<float>(), dist.data_ptr<float>(),
            cos_val.data_ptr<float>());

    vector<Tensor> results(5);
    results[0] = mid_sdf;
    results[1] = diff_sdf;
    results[2] = dist;
    results[3] = cos_val;
    results[4] = diff_offsets;

    return results;
}

vector<Tensor> upsample::up_sample(const Tensor& valid_grid,
                                   const Tensor& uniform_steps,  // [num_full]
                                   const Tensor& weights,        // [num_diff]
                                   const Tensor& uniform_offsets,
                                   const Tensor& diff_offsets,
                                   const int32_t n_importance,
                                   RaysSpec& rays,
                                   RenderOptions& opt) {
    // check
    CHECK_INPUT(uniform_steps);
    CHECK_INPUT(weights);
    CHECK_INPUT(uniform_offsets);
    CHECK_INPUT(diff_offsets);

    const int32_t n_rays = rays.origins.size(0), threads = 1024;
    const int32_t sample_blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // get the valid count
    Tensor valids_count = torch::zeros(
            {n_rays + 1}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    Tensor importance_count = torch::zeros(
            {n_rays + 1}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    Tensor u = torch::linspace(0.f + 0.5 / n_importance, 1.f - 0.5 / n_importance, n_importance,
                               torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    count_importance_kernel<<<sample_blocks, threads>>>(
            rays, opt, valid_grid.data_ptr<bool>(), uniform_steps.data_ptr<float>(),
            weights.data_ptr<float>(), uniform_offsets.data_ptr<int32_t>(),
            diff_offsets.data_ptr<int32_t>(), u.data_ptr<float>(), n_importance, n_rays,
            // output
            valids_count.data_ptr<int32_t>(), importance_count.data_ptr<int32_t>());

    // allocate memory
    Tensor importance_offsets = torch::cumsum(importance_count, 0, torch::kInt);  // [n_rays + 1]
    const int32_t importance_size = importance_offsets.index({-1}).item<int32_t>();
    Tensor importance_steps =
            torch::zeros({importance_size},  // [importance_size]
                         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    // importance sampling
    importance_sampling_kernel<<<sample_blocks, threads>>>(
            rays, opt, valid_grid.data_ptr<bool>(), uniform_steps.data_ptr<float>(),
            weights.data_ptr<float>(), uniform_offsets.data_ptr<int32_t>(),
            diff_offsets.data_ptr<int32_t>(), importance_offsets.data_ptr<int32_t>(),
            u.data_ptr<float>(), n_importance, n_rays,
            // Output
            importance_steps.data_ptr<float>());  // [importance_size]

    // allocate memory
    Tensor up_sample_offsets = torch::cumsum(valids_count, 0, torch::kInt);  // [n_rays + 1]
    const int32_t up_sample_size = up_sample_offsets.index({-1}).item<int32_t>();
    Tensor up_sample_steps =
            torch::zeros({up_sample_size},  // [up_sample_size]
                         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor up_sample_pos_dirs =
            torch::zeros({up_sample_size, 6},  // [up_sample_size]
                         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor up_sample_deltas = torch::zeros_like(up_sample_steps);

    // sort the sample points and get the pos_dirs and deltas according to the
    // steps
    sort_sample_points_kernel<<<sample_blocks, threads>>>(
            rays, uniform_steps.data_ptr<float>(), importance_steps.data_ptr<float>(),
            uniform_offsets.data_ptr<int32_t>(), importance_offsets.data_ptr<int32_t>(),
            up_sample_offsets.data_ptr<int32_t>(), n_rays,
            // Output
            up_sample_steps.data_ptr<float>(), up_sample_pos_dirs.data_ptr<float>(),
            up_sample_deltas.data_ptr<float>());

    vector<Tensor> results(5);
    results[0] = up_sample_offsets;
    results[1] = up_sample_steps;
    results[2] = up_sample_pos_dirs;
    results[3] = up_sample_deltas;
    results[4] = importance_steps;
    return results;
}