// Copyright 2021 Alex Yu
#pragma once
#include <torch/extension.h>

#include "cuda_utils.cuh"
#include "data_spec.hpp"

struct PackedCameraSpec {
    PackedCameraSpec(CameraSpec& cam)
        : c2w(cam.c2w.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
          fx(cam.fx),
          fy(cam.fy),
          cx(cam.cx),
          cy(cam.cy),
          width(cam.width),
          height(cam.height),
          ndc_coeffx(cam.ndc_coeffx),
          ndc_coeffy(cam.ndc_coeffy) {}
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> c2w;
    float fx;
    float fy;
    float cx;
    float cy;
    int32_t width;
    int32_t height;

    float ndc_coeffx;
    float ndc_coeffy;
};

struct PackedRaysSpec {
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> origins;
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> dirs;
    PackedRaysSpec(RaysSpec& spec)
        : origins(spec.origins.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
          dirs(spec.dirs.packed_accessor32<float, 2, torch::RestrictPtrTraits>()) {}
};

struct SingleRaySpec {
    SingleRaySpec() = default;
    __device__ SingleRaySpec(const float* __restrict__ origin, const float* __restrict__ dir)
        : origin{origin[0], origin[1], origin[2]}, dir{dir[0], dir[1], dir[2]} {}
    __device__ void set(const float* __restrict__ origin, const float* __restrict__ dir) {
#pragma unroll 3
        for (int32_t i = 0; i < 3; ++i) {
            this->origin[i] = origin[i];
            this->dir[i] = dir[i];
        }
    }

    float origin[3];
    float dir[3];
    float tmin, tmax, world_step;

    float pos[3];
    int32_t l[3];
};
