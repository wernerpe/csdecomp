load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("//:tools/lint.bzl", "add_lint_tests")

add_lint_tests()

cc_library(
    name = "hpolyhedron",
    srcs = ["hpolyhedron.cpp"],
    hdrs = ["hpolyhedron.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@fmt",
        "@glpk",
    ],
)

cc_library(
    name = "linesegment_aabb_checker",
    srcs = ["linesegment_aabb_checker.cpp"],
    hdrs = ["linesegment_aabb_checker.h"],
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
        "@eigen",
    ],
)

cc_library(
    name = "distance_aabb_linesegment",
    srcs = ["distance_aabb_linesegment.cpp"],
    hdrs = ["distance_aabb_linesegment.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
    ],
)

cuda_library(
    name = "cuda_geometry_utilities",
    hdrs = ["cuda_geometry_utilities.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
    ],
)

cuda_library(
    name = "cuda_distance_field",
    srcs = ["cuda_distance_field.cu"],
    hdrs = ["cuda_distance_field.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/utils:cuda_utilities",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_hit_and_run_sampling",
    srcs = ["cuda_hit_and_run_sampling.cu"],
    hdrs = ["cuda_hit_and_run_sampling.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":hpolyhedron",
        "//csdecomp/src/utils:cuda_utilities",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)
