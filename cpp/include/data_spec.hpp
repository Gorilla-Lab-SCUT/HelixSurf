// Copyright 2021 Alex Yu
#pragma once
#include <torch/extension.h>

#include "utils.hpp"

using torch::Tensor;

enum LossType {
    L1 = 0,
    SmoothL1 = 1,
    L2 = 2,
};

struct CameraSpec {
    Tensor c2w;
    float fx;
    float fy;
    float cx;
    float cy;
    int32_t width;
    int32_t height;

    float ndc_coeffx;
    float ndc_coeffy;

    inline void check() {
        CHECK_INPUT(c2w);
        TORCH_CHECK(c2w.is_floating_point());
        TORCH_CHECK(c2w.ndimension() == 2);
        TORCH_CHECK(c2w.size(1) == 4);
    }
};

struct RaysSpec {
    Tensor origins;
    Tensor dirs;
    inline void check() {
        CHECK_INPUT(origins);
        CHECK_INPUT(dirs);
        TORCH_CHECK(origins.is_floating_point());
        TORCH_CHECK(dirs.is_floating_point());
    }
};

struct RenderOptions {
    float background_brightness;
    // float step_epsilon;
    float step_size;
    float sigma_thresh;
    float stop_thresh;

    float near_clip;
    bool use_spheric_clip;

    bool last_sample_opaque;

    // mlp parameters
    float bound;
    // float mean_density;
    float density_thresh;

    // grid paremeters
    int32_t grid_res;
    int32_t grid_sample_res;
};
