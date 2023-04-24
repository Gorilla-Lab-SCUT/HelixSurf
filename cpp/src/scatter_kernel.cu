// Copyright 2022 Gorilla-Lab

#include "cuda_utils.cuh"
#include "scatter_kernel.cuh"

const int32_t NUM_THREADS = 1024;

// kernel
__global__ void offsets_to_index_kernel(const int32_t num_groups,
                                        const int32_t* __restrict__ offsets_ptr,
                                        // output: store the not empty points for each ray
                                        int32_t* __restrict__ index_ptr) {
    CUDA_GET_THREAD_ID(group_id, num_groups);

    int32_t start = offsets_ptr[group_id], end = offsets_ptr[group_id + 1];

    if (start == end) {
        return;
    }

    for (int32_t i = start; i < end; ++i) {
        index_ptr[i] = group_id;
    }
}

__global__ void scatter_sum2d_forward_kernel(const int32_t num_feat,
                                             const int32_t num_groups,
                                             const float* __restrict__ inputs_ptr,
                                             const int32_t* __restrict__ offsets_ptr,
                                             // output: store the not empty points for each ray
                                             float* __restrict__ outputs_ptr) {
    CUDA_GET_THREAD_ID(group_id, num_groups);

    int32_t start = offsets_ptr[group_id], end = offsets_ptr[group_id + 1];

    if (start == end) {
        return;
    }

    for (int32_t feat_idx = 0; feat_idx < num_feat; ++feat_idx) {
        for (int32_t i = start; i < end; ++i) {
            outputs_ptr[group_id * num_feat + feat_idx] += inputs_ptr[i * num_feat + feat_idx];
        }
    }
}

__global__ void scatter_sum2d_backward_kernel(const int32_t num_feat,
                                              const int32_t num_groups,
                                              const float* __restrict__ outputs_grad_ptr,
                                              const int32_t* __restrict__ offsets_ptr,
                                              // output: store the not empty points for each ray
                                              float* __restrict__ inputs_grad_ptr) {
    CUDA_GET_THREAD_ID(group_id, num_groups);

    int32_t start = offsets_ptr[group_id], end = offsets_ptr[group_id + 1];

    if (start == end) {
        return;
    }

    for (int32_t feat_idx = 0; feat_idx < num_feat; ++feat_idx) {
        for (int32_t i = start; i < end; ++i) {
            inputs_grad_ptr[i * num_feat + feat_idx] =
                    outputs_grad_ptr[group_id * num_feat + feat_idx];
        }
    }
}

__global__ void scatter_sum_broadcast_kernel(const int32_t num_groups,
                                             const float* __restrict__ inputs_ptr,
                                             const int32_t* __restrict__ offsets_ptr,
                                             // output: store the not empty points for each ray
                                             float* __restrict__ outputs_ptr) {
    CUDA_GET_THREAD_ID(group_id, num_groups);

    int32_t start = offsets_ptr[group_id], end = offsets_ptr[group_id + 1];

    if (start == end) {
        return;
    }

    float accum = 0.f;
    for (int32_t i = start; i < end; ++i) {
        accum += inputs_ptr[i];
    }
    for (int32_t i = start; i < end; ++i) {
        outputs_ptr[i] = accum;
    }
}

__global__ void scatter_cumsum_forward_kernel(const int32_t num_groups,
                                              const float* __restrict__ inputs_ptr,
                                              const int32_t* __restrict__ offsets_ptr,
                                              // output: store the not empty points for each ray
                                              float* __restrict__ outputs_ptr) {
    CUDA_GET_THREAD_ID(group_id, num_groups);

    int32_t start = offsets_ptr[group_id], end = offsets_ptr[group_id + 1];

    if (start == end) {
        return;
    }

    float accum = 0;
    for (int32_t i = start; i < end; ++i) {
        accum += inputs_ptr[i];
        outputs_ptr[i] = accum;
    }
}

__global__ void scatter_cumsum_backward_kernel(const int32_t num_groups,
                                               const float* __restrict__ outputs_grad_ptr,
                                               const int32_t* __restrict__ offsets_ptr,
                                               // output: store the not empty points for each ray
                                               float* __restrict__ inputs_grad_ptr) {
    CUDA_GET_THREAD_ID(group_id, num_groups);

    int32_t start = offsets_ptr[group_id], end = offsets_ptr[group_id + 1];

    if (start == end) {
        return;
    }

    float inv_accum_grad = 0;
    for (int32_t i = end - 1; i >= start; --i) {
        inv_accum_grad += outputs_grad_ptr[i];
        inputs_grad_ptr[i] = inv_accum_grad;
    }
}

__global__ void scatter_cumprod_forward_kernel(const bool one_start,
                                               const int32_t num_groups,
                                               const float* __restrict__ inputs_ptr,
                                               const int32_t* __restrict__ offsets_ptr,
                                               // output: store the not empty points for each ray
                                               float* __restrict__ outputs_ptr) {
    CUDA_GET_THREAD_ID(group_id, num_groups);

    int32_t start = offsets_ptr[group_id], end = offsets_ptr[group_id + 1];

    if (start == end) {
        return;
    }

    float cumul = 1;
    for (int32_t i = start; i < end; ++i) {
        if (one_start) {
            outputs_ptr[i] = cumul;
            cumul *= inputs_ptr[i];
        } else {
            cumul *= inputs_ptr[i];
            outputs_ptr[i] = cumul;
        }
    }
}

__global__ void scatter_cumprod_backward_kernel(const bool one_start,
                                                const int32_t num_groups,
                                                const float* __restrict__ outputs_grad_ptr,
                                                const float* __restrict__ inputs_ptr,
                                                const float* __restrict__ outputs_ptr,
                                                const int32_t* __restrict__ offsets_ptr,
                                                // output: store the not empty points for each ray
                                                float* __restrict__ inputs_grad_ptr) {
    CUDA_GET_THREAD_ID(group_id, num_groups);

    int32_t start = offsets_ptr[group_id], end = offsets_ptr[group_id + 1];

    if (start == end) {
        return;
    }

    float accum_grad = 0;
    for (int32_t i = end - 1; i >= start; --i) {
        if (one_start) {
            inputs_grad_ptr[i] = accum_grad / inputs_ptr[i];
            accum_grad += outputs_ptr[i] * outputs_grad_ptr[i];
        } else {
            accum_grad += outputs_ptr[i] * outputs_grad_ptr[i];
            inputs_grad_ptr[i] = accum_grad / inputs_ptr[i];
        }
    }
}

__global__ void scatter_max_broadcast_kernel(const int32_t num_groups,
                                             const float* __restrict__ inputs_ptr,
                                             const int32_t* __restrict__ offsets_ptr,
                                             // output: store the not empty points for each ray
                                             bool* __restrict__ outputs_ptr) {
    CUDA_GET_THREAD_ID(group_id, num_groups);

    int32_t start = offsets_ptr[group_id], end = offsets_ptr[group_id + 1];

    if (start == end) {
        return;
    }

    float max_weights = 0.f;
    int max_idx = 0;
    for (int32_t i = start; i < end; ++i) {
        if (inputs_ptr[i] > max_weights) {
            max_weights = inputs_ptr[i];
            max_idx = i;
        }
    }
    outputs_ptr[max_idx] = true;
    // for (int32_t i = start; i < end; ++i) {
    //     outputs_ptr[i] = (i == max_idx) ? 1 : 0;
    // }
}

__global__ void scatter_var_kernel(const int32_t num_groups,
                                   const float* __restrict__ inputs_ptr,
                                   const float* __restrict__ weights_ptr,
                                   const int32_t* __restrict__ offsets_ptr,
                                   // output: store the not empty points for each ray
                                   float* __restrict__ outputs_ptr) {
    CUDA_GET_THREAD_ID(group_id, num_groups);

    int32_t start = offsets_ptr[group_id], end = offsets_ptr[group_id + 1];

    if (start == end) {
        return;
    }

    float avg = 0.f;
    for (int32_t i = start; i < end; ++i) {
        avg += inputs_ptr[i] * weights_ptr[i];
    }

    float var = 0.f;
    for (int32_t i = start; i < end; ++i) {
        var += weights_ptr[i] * powf(inputs_ptr[i] - avg, 2);
    }
    outputs_ptr[group_id] = var;
}

// wrapper

Tensor scatter::offsets_to_index(const Tensor& offsets) {
    // check
    CHECK_INPUT(offsets);

    const int32_t num_inputs = offsets.index({-1}).item<int32_t>();
    const int32_t num_groups = offsets.size(0) - 1;
    const int32_t num_blocks = CUDA_N_BLOCKS_NEEDED(num_groups, NUM_THREADS);
    Tensor index = torch::zeros({num_inputs},
                                torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));

    offsets_to_index_kernel<<<num_blocks, NUM_THREADS>>>(
            num_groups, offsets.data_ptr<int32_t>(),
            // Output
            index.data_ptr<int32_t>());  // [num_inputs]

    return index;
}

Tensor scatter::scatter_cumsum_forward(const Tensor& inputs, const Tensor& offsets) {
    // check
    CHECK_INPUT(inputs);
    CHECK_INPUT(offsets);

    const int32_t num_groups = offsets.size(0) - 1;
    const int32_t num_blocks = CUDA_N_BLOCKS_NEEDED(num_groups, NUM_THREADS);

    Tensor outputs = torch::zeros_like(inputs);
    scatter_cumsum_forward_kernel<<<num_blocks, NUM_THREADS>>>(
            num_groups, inputs.data_ptr<float>(), offsets.data_ptr<int32_t>(),
            // Output
            outputs.data_ptr<float>());  // [num_inputs]

    return outputs;
}

Tensor scatter::scatter_cumsum_backward(const Tensor& outputs_grad, const Tensor& offsets) {
    // check
    CHECK_INPUT(outputs_grad);
    CHECK_INPUT(offsets);

    const int32_t num_groups = offsets.size(0) - 1;
    const int32_t num_blocks = CUDA_N_BLOCKS_NEEDED(num_groups, NUM_THREADS);

    Tensor inputs_grad = torch::zeros_like(outputs_grad);
    scatter_cumsum_backward_kernel<<<num_blocks, NUM_THREADS>>>(
            num_groups, outputs_grad.data_ptr<float>(), offsets.data_ptr<int32_t>(),
            // Output
            inputs_grad.data_ptr<float>());  // [num_inputs]

    return inputs_grad;
}

Tensor scatter::scatter_sum2d_forward(const Tensor& inputs, const Tensor& offsets) {
    // check
    CHECK_INPUT(inputs);
    CHECK_INPUT(offsets);

    const int32_t num_feat = inputs.size(1);
    const int32_t num_groups = offsets.size(0) - 1;
    const int32_t num_blocks = CUDA_N_BLOCKS_NEEDED(num_groups, NUM_THREADS);

    Tensor outputs = torch::zeros({num_groups, num_feat},
                                  torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    scatter_sum2d_forward_kernel<<<num_blocks, NUM_THREADS>>>(
            num_feat, num_groups, inputs.data_ptr<float>(), offsets.data_ptr<int32_t>(),
            // Output
            outputs.data_ptr<float>());  // [num_groups, num_feat]

    return outputs;
}

Tensor scatter::scatter_sum2d_backward(const Tensor& outputs_grad, const Tensor& offsets) {
    // check
    CHECK_INPUT(outputs_grad);
    CHECK_INPUT(offsets);

    const int32_t num_inputs = offsets.index({-1}).item<int32_t>();
    const int32_t num_feat = outputs_grad.size(1);
    const int32_t num_groups = offsets.size(0) - 1;
    const int32_t num_blocks = CUDA_N_BLOCKS_NEEDED(num_groups, NUM_THREADS);

    Tensor inputs_grad =
            torch::zeros({num_inputs, num_feat},
                         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    scatter_sum2d_backward_kernel<<<num_blocks, NUM_THREADS>>>(
            num_feat, num_groups, outputs_grad.data_ptr<float>(), offsets.data_ptr<int32_t>(),
            // Output
            inputs_grad.data_ptr<float>());  // [num_inputs]

    return inputs_grad;
}

Tensor scatter::scatter_sum_broadcast(const Tensor& inputs, const Tensor& offsets) {
    // check
    CHECK_INPUT(inputs);
    CHECK_INPUT(offsets);

    const int32_t num_groups = offsets.size(0) - 1;
    const int32_t num_blocks = CUDA_N_BLOCKS_NEEDED(num_groups, NUM_THREADS);

    Tensor outputs = torch::zeros_like(inputs);

    scatter_sum_broadcast_kernel<<<num_blocks, NUM_THREADS>>>(num_groups, inputs.data_ptr<float>(),
                                                              offsets.data_ptr<int32_t>(),
                                                              // Output
                                                              outputs.data_ptr<float>());

    return outputs;
}

Tensor scatter::scatter_cumprod_forward(const Tensor& inputs,
                                        const Tensor& offsets,
                                        const bool one_start) {
    // check
    CHECK_INPUT(inputs);
    CHECK_INPUT(offsets);

    const int32_t num_groups = offsets.size(0) - 1;
    const int32_t num_blocks = CUDA_N_BLOCKS_NEEDED(num_groups, NUM_THREADS);

    Tensor outputs = torch::zeros_like(inputs);
    scatter_cumprod_forward_kernel<<<num_blocks, NUM_THREADS>>>(
            one_start, num_groups, inputs.data_ptr<float>(), offsets.data_ptr<int32_t>(),
            // Output
            outputs.data_ptr<float>());  // [num_inputs]

    return outputs;
}

Tensor scatter::scatter_cumprod_backward(const Tensor& outputs_grad,
                                         const Tensor& inputs,
                                         const Tensor& outputs,
                                         const Tensor& offsets,
                                         const bool one_start) {
    // check
    CHECK_INPUT(outputs_grad);
    CHECK_INPUT(inputs);
    CHECK_INPUT(outputs);
    CHECK_INPUT(offsets);

    const int32_t num_groups = offsets.size(0) - 1;
    const int32_t num_blocks = CUDA_N_BLOCKS_NEEDED(num_groups, NUM_THREADS);

    Tensor inputs_grad = torch::zeros_like(outputs_grad);
    scatter_cumprod_backward_kernel<<<num_blocks, NUM_THREADS>>>(
            one_start, num_groups, outputs_grad.data_ptr<float>(), inputs.data_ptr<float>(),
            outputs.data_ptr<float>(), offsets.data_ptr<int32_t>(),
            // Output
            inputs_grad.data_ptr<float>());  // [num_inputs]

    return inputs_grad;
}

Tensor scatter::scatter_max(const Tensor& inputs, const Tensor& offsets) {
    // check
    CHECK_INPUT(inputs);
    CHECK_INPUT(offsets);

    const int32_t num_groups = offsets.size(0) - 1;
    const int32_t num_length = inputs.size(0);
    const int32_t num_blocks = CUDA_N_BLOCKS_NEEDED(num_groups, NUM_THREADS);

    Tensor outputs = torch::zeros({num_length},
                                  torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    scatter_max_broadcast_kernel<<<num_blocks, NUM_THREADS>>>(num_groups, inputs.data_ptr<float>(),
                                                              offsets.data_ptr<int32_t>(),
                                                              // Output
                                                              outputs.data_ptr<bool>());

    return outputs;
}

Tensor scatter::scatter_var(const Tensor& inputs, const Tensor& weights, const Tensor& offsets) {
    // check
    CHECK_INPUT(inputs);
    CHECK_INPUT(weights);
    CHECK_INPUT(offsets);

    const int32_t num_groups = offsets.size(0) - 1;
    const int32_t num_blocks = CUDA_N_BLOCKS_NEEDED(num_groups, NUM_THREADS);

    Tensor outputs = torch::zeros({num_groups},
                                  torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    scatter_var_kernel<<<num_blocks, NUM_THREADS>>>(num_groups, inputs.data_ptr<float>(),
                                                    weights.data_ptr<float>(),
                                                    offsets.data_ptr<int32_t>(),
                                                    // Output
                                                    outputs.data_ptr<float>());

    return outputs;
}
