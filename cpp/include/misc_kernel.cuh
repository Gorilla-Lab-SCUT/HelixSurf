// Copyright 2021 Alex Yu
#pragma once
#include <torch/extension.h>

using torch::Tensor;

namespace misc {
void accel_dist_prop(Tensor);
}  // namespace misc
