// Copyright 2022 Gorilla-Lab
#pragma once
#include <torch/extension.h>
#include <vector>

using std::vector;
using torch::Tensor;

namespace sampling {
Tensor grid_sample(const Tensor&, const Tensor&, const Tensor&, const int32_t);
Tensor sample_patch(const Tensor&, const Tensor&, const int32_t);
}  // namespace sampling
