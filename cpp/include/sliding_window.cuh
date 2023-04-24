// Copyright 2022 Gorilla-Lab
#pragma once
#include <torch/extension.h>

using torch::Tensor;

namespace sliding_window {
Tensor sliding_window_normal(const Tensor&, const Tensor&, const int32_t, const int32_t);
Tensor sliding_window_normal_cu(const Tensor&, const Tensor&, const int32_t, const int32_t);
Tensor sliding_window_normal_with_primitive(
        const Tensor&, const Tensor&, Tensor&, Tensor&, const int32_t, const int32_t);
void count_primitives(const Tensor&, const Tensor&, Tensor&, Tensor&);
void count_primitives_cu(const Tensor&, const Tensor&, Tensor&, Tensor&);
}  // namespace sliding_window
