// Copyright 2021 Alex Yu
#pragma once
#include <torch/extension.h>

using torch::Tensor;

namespace scatter {
Tensor offsets_to_index(const Tensor&);
Tensor scatter_sum2d_forward(const Tensor&, const Tensor&);
Tensor scatter_sum2d_backward(const Tensor&, const Tensor&);
Tensor scatter_sum_broadcast(const Tensor&, const Tensor&);
Tensor scatter_cumsum_forward(const Tensor&, const Tensor&);
Tensor scatter_cumsum_backward(const Tensor&, const Tensor&);
Tensor scatter_cumprod_forward(const Tensor&, const Tensor&, const bool);
Tensor scatter_cumprod_backward(
        const Tensor&, const Tensor&, const Tensor&, const Tensor&, const bool);
Tensor scatter_max(const Tensor&, const Tensor&);
Tensor scatter_var(const Tensor&, const Tensor&, const Tensor&);
}  // namespace scatter
