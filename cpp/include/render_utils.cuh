// Copyright 2021 Alex Yu
#pragma once

#include "data_spec_packed.cuh"

// Compute the amount to skip for negative link values
__device__ __inline__ float compute_skip_dist(SingleRaySpec& __restrict__ ray,
                                              const int8_t* __restrict__ links,
                                              int32_t offx,
                                              int32_t offy,
                                              int32_t pos_offset = 0) {
    const int32_t link_val =
            static_cast<int32_t>(links[offx * (ray.l[0] + pos_offset) +
                                       offy * (ray.l[1] + pos_offset) + (ray.l[2] + pos_offset)]);
    if (link_val >= -1) return 0.f;  // Not worth

    const uint32_t dist = -link_val;
    const uint32_t cell_ul_shift = (dist - 1);
    const uint32_t cell_side_len = (1 << cell_ul_shift) - 1.f;

    // AABB intersection
    // Consider caching the invdir for the ray
    float tmin = 0.f;
    float tmax = 1e9f;
#pragma unroll 3
    for (int32_t i = 0; i < 3; ++i) {
        int32_t ul = (((ray.l[i] + pos_offset) >> cell_ul_shift) << cell_ul_shift);
        ul -= ray.l[i] + pos_offset;

        const float invdir = 1.0 / ray.dir[i];
        const float t1 = (ul - ray.pos[i] + pos_offset) * invdir;
        const float t2 = (ul - ray.pos[i] + pos_offset + cell_side_len) * invdir;
        if (ray.dir[i] != 0.f) {
            tmin = fmaxf(tmin, fminf(t1, t2));
            tmax = fminf(tmax, fmaxf(t1, t2));
        }
    }

    if (tmin > 0.f) {
        // Somehow the origin is not in the cube
        // Should not happen for distance transform

        // If using geometric distances:
        // will happen near the lowest vertex of a cell,
        // since l is always the lowest neighbor
        return 0.f;
    }
    return tmax;
}