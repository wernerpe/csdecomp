load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("//:tools/lint.bzl", "add_lint_tests")

add_lint_tests()

cuda_library(
    name = "cuda_utilities",
    hdrs = ["cuda_utilities.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_distance_field",
    srcs = ["cuda_distance_field.cu"],
    hdrs = ["cuda_distance_field.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_utilities",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_forward_kinematics",
    srcs = ["cuda_forward_kinematics.cu"],
    hdrs = ["cuda_forward_kinematics.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_utilities",
        "@//csdecomp/src/cpp:minimal_kinematic_tree",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_collision_checker",
    srcs = ["cuda_collision_checker.cu"],
    hdrs = [
        "cuda_collision_checker.h",
        "cuda_geometry_utilities.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_forward_kinematics",
        ":cuda_utilities",
        "@//csdecomp/src/cpp:minimal_plant",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_hit_and_run_sampling",
    srcs = ["cuda_hit_and_run_sampling.cu"],
    hdrs = ["cuda_hit_and_run_sampling.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_utilities",
        "@//csdecomp/src/cpp:hpolyhedron",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_set_builder_utils",
    srcs = ["cuda_set_builder_utils.cu"],
    hdrs = ["cuda_set_builder_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_utilities",
        "@//csdecomp/src/cpp:hpolyhedron",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_visibility_graph",
    srcs = ["cuda_visibility_graph.cu"],
    hdrs = ["cuda_visibility_graph.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_collision_checker",
        ":cuda_utilities",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_edge_inflation_zero_order",
    srcs = ["cuda_edge_inflation_zero_order.cu"],
    hdrs = ["cuda_edge_inflation_zero_order.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_collision_checker",
        ":cuda_hit_and_run_sampling",
        ":cuda_set_builder_utils",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_polytope_builder",
    srcs = ["cuda_polytope_builder.cu"],
    hdrs = ["cuda_polytope_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_edge_inflation_zero_order",
        "@eigen",
    ],
)

cuda_library(
    name = "cuda_edit_regions",
    srcs = ["cuda_edit_regions.cu"],
    hdrs = ["cuda_edit_regions.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_set_builder_utils",
        ":cuda_polytope_builder",
        ":cuda_collision_checker",
        "@eigen",
    ],
)

cuda_library(
    name = "drm_cuda_utils",
    srcs = ["drm_cuda_utils.cu"],
    hdrs = ["drm_cuda_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@local_cuda//:cuda_runtime",
    ],
)
