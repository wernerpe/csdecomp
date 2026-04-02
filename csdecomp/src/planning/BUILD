load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("//:tools/lint.bzl", "add_lint_tests")

add_lint_tests()

cc_library(
    name = "roadmap",
    srcs = ["roadmap.cpp"],
    hdrs = ["roadmap.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "@cereal",
        "@eigen",
    ],
)

cc_library(
    name = "bezier_curve",
    srcs = ["bezier_curve.cpp"],
    hdrs = ["bezier_curve.h"],
    copts = [
        "-Icsdecomp/src",
        "-fopenmp",
        "-O2",
    ],
    linkopts = [
        "-fopenmp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/geometry:hpolyhedron",
        "@eigen",
    ],
)

cuda_library(
    name = "cuda_set_builder_utils",
    srcs = ["cuda_set_builder_utils.cu"],
    hdrs = ["cuda_set_builder_utils.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/utils:cuda_utilities",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_visibility_graph",
    srcs = ["cuda_visibility_graph.cu"],
    hdrs = ["cuda_visibility_graph.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/plant:cuda_collision_checker",
        "//csdecomp/src/utils:cuda_utilities",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_edge_inflation_zero_order",
    srcs = ["cuda_edge_inflation_zero_order.cu"],
    hdrs = ["cuda_edge_inflation_zero_order.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_set_builder_utils",
        "//csdecomp/src/geometry:cuda_hit_and_run_sampling",
        "//csdecomp/src/plant:cuda_collision_checker",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_polytope_builder",
    srcs = ["cuda_polytope_builder.cu"],
    hdrs = ["cuda_polytope_builder.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_edge_inflation_zero_order",
        "@eigen",
    ],
)

cuda_library(
    name = "cuda_plane_update",
    srcs = ["cuda_plane_update.cu"],
    hdrs = ["cuda_plane_update.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_edge_inflation_zero_order",
        ":cuda_set_builder_utils",
        "//csdecomp/src/plant:cuda_collision_checker",
        "//csdecomp/src/utils:cuda_utilities",
        "@eigen",
        "@fmt",
    ],
)

cuda_library(
    name = "cuda_edit_regions",
    srcs = ["cuda_edit_regions.cu"],
    hdrs = ["cuda_edit_regions.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_polytope_builder",
        ":cuda_set_builder_utils",
        "//csdecomp/src/plant:cuda_collision_checker",
        "@eigen",
    ],
)

cuda_library(
    name = "drm_cuda_utils",
    srcs = ["drm_cuda_utils.cu"],
    hdrs = ["drm_cuda_utils.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "roadmap_builder",
    srcs = ["roadmap_builder.cpp"],
    hdrs = ["roadmap_builder.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":drm_cuda_utils",
        ":roadmap",
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/plant:collision_checker",
        "//csdecomp/src/plant:cuda_collision_checker",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/src/utils:cuda_utilities",
        "@eigen",
    ],
)

cc_library(
    name = "drm_planner",
    srcs = ["drm_planner.cpp"],
    hdrs = ["drm_planner.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":drm_cuda_utils",
        ":roadmap",
        ":roadmap_builder",
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/plant:collision_checker",
        "//csdecomp/src/plant:cuda_collision_checker",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/src/utils:cuda_utilities",
        "@eigen",
    ],
)
