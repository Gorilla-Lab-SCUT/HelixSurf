// Copyright 2022 Gorilla-Lab
#include <torch/extension.h>

#include "data_spec.hpp"
#include "grid_sample_kernel.cuh"
#include "marching_rays_kernel.cuh"
#include "misc_kernel.cuh"
#include "scatter_kernel.cuh"
#include "sliding_window.cuh"
#include "up_sample_kernel.cuh"
#include "spherical_harmonics.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // marchingrs
    m.def("grid_sample", &sampling::grid_sample);
    m.def("sample_patch", &sampling::sample_patch);

    // marchingrs
    m.def("marching_rays", &marching::marching_rays);

    // spherical harmonics encoding
    m.def("spherical_harmonic_forward", &spherical_harmonics::spherical_harmonic_forward);
    m.def("spherical_harmonic_backward", &spherical_harmonics::spherical_harmonic_backward);

    // misc
    m.def("accel_dist_prop", &misc::accel_dist_prop);

    // sliding window
    m.def("sliding_window_normal", &sliding_window::sliding_window_normal);
    m.def("sliding_window_normal_cu", &sliding_window::sliding_window_normal_cu);
    m.def("sliding_window_normal_with_primitive",
          &sliding_window::sliding_window_normal_with_primitive);
    m.def("count_primitives", &sliding_window::count_primitives);
    m.def("count_primitives_cu", &sliding_window::count_primitives_cu);

    // scatter
    m.def("offsets_to_index", &scatter::offsets_to_index);
    m.def("scatter_sum2d_forward", &scatter::scatter_sum2d_forward);
    m.def("scatter_sum2d_backward", &scatter::scatter_sum2d_backward);
    m.def("scatter_sum_broadcast", &scatter::scatter_sum_broadcast);
    m.def("scatter_cumsum_forward", &scatter::scatter_cumsum_forward);
    m.def("scatter_cumsum_backward", &scatter::scatter_cumsum_backward);
    m.def("scatter_cumprod_forward", &scatter::scatter_cumprod_forward);
    m.def("scatter_cumprod_backward", &scatter::scatter_cumprod_backward);
    m.def("scatter_max", &scatter::scatter_max);
    m.def("scatter_var", &scatter::scatter_var);

    // upsample
    m.def("prev_next_diff", &upsample::prev_next_diff);
    m.def("up_sample", &upsample::up_sample);

    // data structure
    py::enum_<LossType>(m, "loss_type")
            .value("l1", LossType::L1)
            .value("smooth_l1", LossType::SmoothL1)
            .value("l2", LossType::L2)
            .export_values();
    py::class_<CameraSpec>(m, "CameraSpec")
            .def(py::init<>())
            .def_readwrite("c2w", &CameraSpec::c2w)
            .def_readwrite("fx", &CameraSpec::fx)
            .def_readwrite("fy", &CameraSpec::fy)
            .def_readwrite("cx", &CameraSpec::cx)
            .def_readwrite("cy", &CameraSpec::cy)
            .def_readwrite("width", &CameraSpec::width)
            .def_readwrite("height", &CameraSpec::height)
            .def_readwrite("ndc_coeffx", &CameraSpec::ndc_coeffx)
            .def_readwrite("ndc_coeffy", &CameraSpec::ndc_coeffy);

    py::class_<RaysSpec>(m, "RaysSpec")
            .def(py::init<>())
            .def_readwrite("origins", &RaysSpec::origins)
            .def_readwrite("dirs", &RaysSpec::dirs);

    py::class_<RenderOptions>(m, "RenderOptions")
            .def(py::init<>())
            .def_readwrite("background_brightness", &RenderOptions::background_brightness)
            .def_readwrite("step_size", &RenderOptions::step_size)
            .def_readwrite("sigma_thresh", &RenderOptions::sigma_thresh)
            .def_readwrite("stop_thresh", &RenderOptions::stop_thresh)
            .def_readwrite("near_clip", &RenderOptions::near_clip)
            .def_readwrite("use_spheric_clip", &RenderOptions::use_spheric_clip)
            .def_readwrite("last_sample_opaque", &RenderOptions::last_sample_opaque)
            .def_readwrite("bound", &RenderOptions::bound)
            .def_readwrite("density_thresh", &RenderOptions::density_thresh)
            .def_readwrite("grid_res", &RenderOptions::grid_res)
            .def_readwrite("grid_sample_res", &RenderOptions::grid_sample_res);
}
