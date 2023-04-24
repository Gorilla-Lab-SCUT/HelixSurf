// Copyright 2022 Gorilla-Lab
#pragma once
#include <torch/extension.h>
#include <vector>

using std::vector;
using torch::Tensor;

namespace upsample {
vector<Tensor> prev_next_diff(const Tensor&, const Tensor&, const Tensor&);

vector<Tensor> up_sample(const Tensor&,
                         const Tensor&,
                         const Tensor&,
                         const Tensor&,
                         const Tensor&,
                         const int32_t,
                         RaysSpec&,
                         RenderOptions&);
}  // namespace upsample