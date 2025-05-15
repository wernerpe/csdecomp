load("//:tools/lint.bzl", "add_lint_tests")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

add_lint_tests()

#defines hpolyhedron and minimalhpolyhedron
cc_library(
    name = "hpolyhedron",
    srcs = ["hpolyhedron.cpp"],
    hdrs = ["hpolyhedron.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@fmt",
        "@glpk",
    ],
)

cc_library(
    name = "linesegment_aabb_checker",
    srcs = ["linesegment_aabb_checker.h"],
    hdrs = ["linesegment_aabb_checker.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
    ],
)

#defines shape primitives and scene geometries struct
cc_library(
    name = "collision_geometry",
    srcs = [],
    hdrs = ["collision_geometry.h"],
    visibility = ["//visibility:public"],
    deps = ["@eigen"],
)

#defines minimal functionality for evaluating FK
cc_library(
    name = "minimal_kinematic_tree",
    srcs = ["minimal_kinematic_tree.cpp"],
    hdrs = ["minimal_kinematic_tree.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@fmt",
    ],
)

#defines cpp class that is easy to debug for FK
cc_library(
    name = "kinematic_tree",
    srcs = ["kinematic_tree.cpp"],
    hdrs = ["kinematic_tree.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":minimal_kinematic_tree",
        "@eigen",
        "@fmt",
    ],
)

#defines shape primitives and scene geometries struct
cc_library(
    name = "minimal_plant",
    srcs = [],
    hdrs = ["minimal_plant.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collision_geometry",
        ":kinematic_tree",
        ":minimal_kinematic_tree",
        "@eigen",
    ],
)

#defines cpp function that is easy to debug for collision checking
cc_library(
    name = "collision_checker",
    srcs = ["collision_checker.cpp"],
    hdrs = ["collision_checker.h"],
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

#defines roadmap.
cc_library(
    name = "roadmap",
    srcs = ["roadmap.cpp"],
    hdrs = ["roadmap.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@cereal",
        "@eigen",
    ],
)

#defines roadmap builder.
cuda_library(
    name = "roadmap_builder",
    srcs = ["roadmap_builder.cpp"],
    hdrs = ["roadmap_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collision_checker",
        ":hpolyhedron",
        ":roadmap",
        ":urdf_parser",
        "@//csdecomp/src/cuda:cuda_collision_checker",
        "@//csdecomp/src/cuda:cuda_utilities",
        "@//csdecomp/src/cuda:drm_cuda_utils",
        "@eigen",
    ],
)

#defines roadmap.
cc_library(
    name = "drm_planner",
    srcs = ["drm_planner.cpp"],
    hdrs = ["drm_planner.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collision_checker",
        ":hpolyhedron",
        ":roadmap",
        ":roadmap_builder",
        ":urdf_parser",
        "@//csdecomp/src/cuda:cuda_collision_checker",
        "@//csdecomp/src/cuda:cuda_utilities",
        "@//csdecomp/src/cuda:drm_cuda_utils",
        "@eigen",
    ],
    # copts = ["-fopenmp"],
    # linkopts = ["-fopenmp"]
)
