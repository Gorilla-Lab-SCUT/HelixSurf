// Copyright 2022 Gorilla-Lab
#pragma once
#include <torch/extension.h>
#include <vector>

using std::vector;
using torch::Tensor;

namespace marching {
vector<Tensor> marching_rays(const Tensor&, const Tensor&, RaysSpec&, RenderOptions&);
}  // namespace marching