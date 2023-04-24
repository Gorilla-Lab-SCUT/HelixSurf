#pragma once
#include <torch/extension.h>

using torch::Tensor;

namespace spherical_harmonics {
Tensor spherical_harmonic_forward(const Tensor&, const uint8_t);
Tensor spherical_harmonic_backward(const Tensor&, const Tensor&, const uint8_t);
}