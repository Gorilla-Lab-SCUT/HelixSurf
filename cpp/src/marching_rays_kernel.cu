// Copyright 2022 Gorilla-Lab

#include "cuda_utils.cuh"
#include "data_spec_packed.cuh"
#include "marching_rays_kernel.cuh"
#include "render_utils.cuh"

const int32_t TRACE_RAY_CUDA_THREADS = 128;
const int32_t MAX_SAMPLES = 1024;

__device__ __inline__ void find_bounds(SingleRaySpec& __restrict__ ray,
                                       RenderOptions& __restrict__ opt) {
    // calculate world step
    ray.world_step = opt.step_size / (float)opt.grid_sample_res;

    ray.tmin = opt.near_clip / ray.world_step;
    ray.tmax = 2e3f;
#pragma unroll 3
    for (int32_t i = 0; i < 3; ++i) {
        if (ray.dir[i] != 0.f) {
            const float invdir = 1.f / ray.dir[i];
            const float t1 = (-opt.bound - ray.origin[i]) * invdir;
            const float t2 = (opt.bound - ray.origin[i]) * invdir;
            ray.tmin = max(ray.tmin, min(t1, t2));
            ray.tmax = min(ray.tmax, max(t1, t2));
        }
    }
}

// kernel
__global__ void count_rays_kernel(const bool* __restrict__ valid_grid_ptr,
                                  const int8_t* __restrict__ accel_grid_ptr,
                                  PackedRaysSpec rays,
                                  RenderOptions opt,
                                  // store the not empty points for each ray
                                  int32_t* __restrict__ valids_count_ptr) {
    CUDA_GET_THREAD_ID(ray_id, rays.origins.size(0));

    // get the ray
    SingleRaySpec ray;
    ray.set(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    find_bounds(ray, opt);
    // printf("tmin: %f, tmax: %f\n", ray.tmin, ray.tmax);

    if (ray.tmin > ray.tmax) {
        return;
    }

    float t = ray.tmin;
    const float size = (2 * opt.bound) / opt.grid_res;  // the size of grid
    const int32_t offx = opt.grid_res * opt.grid_res, offy = opt.grid_res;

    while (t <= ray.tmax && valids_count_ptr[ray_id + 1] < MAX_SAMPLES) {
        // get the lower indice(link) and the relative position
#pragma unroll 3
        for (int32_t j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j],
                              ray.origin[j]);  // now the `ray.pos` means the real position of
                                               // the sample in sparse grid coordinate
            ray.pos[j] = clamp(ray.pos[j], -opt.bound, opt.bound);
            // find the corresponding occupancy grid
            ray.l[j] = min(int32_t((ray.pos[j] + opt.bound) / size), opt.grid_res - 1);
            ray.pos[j] = fminf((ray.pos[j] + opt.bound) / size - ray.l[j], float(opt.grid_res - 1));
        }

        const float skip = compute_skip_dist(ray, accel_grid_ptr, opt.grid_res * opt.grid_res,
                                             opt.grid_res, 0) *
                           size;

        if (skip >= ray.world_step) {
            // For consistency, we skip the by step size
            t += ceilf(skip / ray.world_step) * ray.world_step;
            continue;
        }

        // query the occupancy(density)
        const bool occupancy = valid_grid_ptr[ray.l[0] * offx + ray.l[1] * offy + ray.l[2]];

        if (occupancy) {
            ++valids_count_ptr[ray_id + 1];
        }

        // TODO: variable step size
        t += ray.world_step;
    }
}

__global__ void sample_rays_kernel(const bool* __restrict__ valid_grid_ptr,
                                   const int8_t* __restrict__ accel_grid_ptr,
                                   PackedRaysSpec rays,
                                   RenderOptions opt,
                                   const int32_t* __restrict__ offsets_ptr,  // [n_rays + 1]
                                   // output
                                   float* __restrict__ pos_dirs_ptr,  // [compact_size, 6]
                                   float* __restrict__ deltas_ptr,    // [compact_size]
                                   float* __restrict__ steps_ptr      // [compact_size]
) {
    CUDA_GET_THREAD_ID(ray_id, rays.origins.size(0));

    SingleRaySpec ray;
    ray.set(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    find_bounds(ray, opt);

    int32_t start = offsets_ptr[ray_id], end = offsets_ptr[ray_id + 1];

    if (start == end || ray.tmin > ray.tmax) {
        return;
    }

    float t = ray.tmin;
    const float size = (2 * opt.bound) / opt.grid_res;  // the size of grid
    int32_t idx = start;

    while (t <= ray.tmax && idx < end) {
        // get the lower indice(link) and the relative position
#pragma unroll 3
        for (int32_t j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j],
                              ray.origin[j]);  // now the `ray.pos` means the real position of
                                               // the sample in sparse grid coordinate
            ray.pos[j] = clamp(ray.pos[j], -opt.bound, opt.bound);
            // find the corresponding occupancy grid
            ray.l[j] = min(int32_t((ray.pos[j] + opt.bound) / size), opt.grid_res - 1);
            ray.pos[j] = (ray.pos[j] + opt.bound) / size - ray.l[j];
        }

        const float skip = compute_skip_dist(ray, accel_grid_ptr, opt.grid_res * opt.grid_res,
                                             opt.grid_res, 0) *
                           size;

        if (skip >= ray.world_step) {
            // For consistency, we skip the by step size
            t += ceilf(skip / ray.world_step) * ray.world_step;
            continue;
        }

        // query the occupancy(density)
        const int32_t offx = opt.grid_res * opt.grid_res, offy = opt.grid_res;
        const bool occupancy = valid_grid_ptr[ray.l[0] * offx + ray.l[1] * offy + ray.l[2]];

        if (occupancy) {
#pragma unroll 3
            for (int32_t j = 0; j < 3; ++j) {
                pos_dirs_ptr[idx * 6 + j] = fmaf(t, ray.dir[j], ray.origin[j]);
                pos_dirs_ptr[idx * 6 + j + 3] = ray.dir[j];
            }
            deltas_ptr[idx] = ray.world_step;
            steps_ptr[idx] = t;
            ++idx;
        }

        t += ray.world_step;
    }
}

// wrapper
vector<Tensor> marching::marching_rays(const Tensor& valid_grid,
                                       const Tensor& accel_grid,
                                       RaysSpec& rays,
                                       RenderOptions& opt) {
    // check
    CHECK_INPUT(valid_grid);
    CHECK_INPUT(accel_grid);

    /*** begin forward ***/
    const int32_t n_rays = rays.origins.size(0);
    Tensor valids_count =
            torch::zeros({n_rays + 1},
                         torch::TensorOptions()
                                 .dtype(torch::kInt)
                                 .device(torch::kCUDA));  // store the count of valid points for
                                                          // each ray [n_rays + 1, 1]
    {
        /* sampling */
        /* get the number of not empty points for each ray(according to the
         * sigma and threshold) */
        const int32_t sample_blocks = CUDA_N_BLOCKS_NEEDED(n_rays, TRACE_RAY_CUDA_THREADS);
        count_rays_kernel<<<sample_blocks, TRACE_RAY_CUDA_THREADS>>>(
                valid_grid.data_ptr<bool>(), accel_grid.data_ptr<int8_t>(), rays, opt,
                // Output
                valids_count.data_ptr<int32_t>());  // [n_rays + 1, 1]
    }
    Tensor offsets = torch::cumsum(valids_count, 0, torch::kInt);  // [n_rays + 1]

    // allocate memory
    const int32_t compact_size = offsets.index({-1}).item<int32_t>();
    Tensor pos_dirs =
            torch::zeros({compact_size, 6},  // [compact_size, 6]
                         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor deltas = torch::zeros({compact_size},  // [compact_size]
                                 torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor steps = torch::zeros({compact_size},  // [compact_size]
                                torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    {
        /* sampling */
        // printf("generate input\n");
        /* get the not empty points for each ray(according to the sigma and
         * threshold) as samples */
        const int32_t sample_blocks = CUDA_N_BLOCKS_NEEDED(n_rays, TRACE_RAY_CUDA_THREADS);
        sample_rays_kernel<<<sample_blocks, TRACE_RAY_CUDA_THREADS>>>(
                valid_grid.data_ptr<bool>(), accel_grid.data_ptr<int8_t>(), rays, opt,
                offsets.data_ptr<int32_t>(),  // [n_rays + 1]
                // Output
                pos_dirs.data_ptr<float>(),  // [compact_size, 6]
                deltas.data_ptr<float>(),    // [compact_size]
                steps.data_ptr<float>());    // [compact_size]
    }

    vector<Tensor> results(4);
    results[0] = offsets;
    results[1] = pos_dirs;
    results[2] = deltas;
    results[3] = steps;
    return results;
}