load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("//:tools/lint.bzl", "add_lint_tests")

add_lint_tests()

cc_library(
    name = "collision_geometry",
    srcs = [],
    hdrs = ["collision_geometry.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = ["@eigen"],
)

cc_library(
    name = "minimal_kinematic_tree",
    srcs = ["minimal_kinematic_tree.cpp"],
    hdrs = ["minimal_kinematic_tree.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@fmt",
    ],
)

cc_library(
    name = "kinematic_tree",
    srcs = ["kinematic_tree.cpp"],
    hdrs = ["kinematic_tree.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":minimal_kinematic_tree",
        "@eigen",
        "@fmt",
    ],
)

cc_library(
    name = "minimal_plant",
    srcs = [],
    hdrs = ["minimal_plant.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":collision_geometry",
        ":kinematic_tree",
        ":minimal_kinematic_tree",
        "@eigen",
    ],
)

cc_library(
    name = "collision_checker",
    srcs = ["collision_checker.cpp"],
    hdrs = ["collision_checker.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":collision_geometry",
        ":kinematic_tree",
        ":minimal_plant",
        "@fmt",
    ],
)

cc_library(
    name = "urdf_parser",
    srcs = ["urdf_parser.cpp"],
    hdrs = ["urdf_parser.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":collision_geometry",
        ":kinematic_tree",
        ":minimal_plant",
        "@eigen",
        "@tinyxml2",
        "@yaml-cpp//:yaml-cpp",
    ],
)

cuda_library(
    name = "cuda_forward_kinematics",
    srcs = ["cuda_forward_kinematics.cu"],
    hdrs = ["cuda_forward_kinematics.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":minimal_kinematic_tree",
        "//csdecomp/src/utils:cuda_utilities",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)

cuda_library(
    name = "cuda_collision_checker",
    srcs = ["cuda_collision_checker.cu"],
    hdrs = [
        "cuda_collision_checker.h",
    ],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_forward_kinematics",
        ":minimal_plant",
        "//csdecomp/src/geometry:cuda_geometry_utilities",
        "//csdecomp/src/utils:cuda_utilities",
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)
