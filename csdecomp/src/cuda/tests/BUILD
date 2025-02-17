load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("//:tools/cc_with_python.bzl", "cc_test_with_python")
load("//:tools/lint.bzl", "add_lint_tests")

add_lint_tests()

cc_test(
    name = "cuda_forward_kinematics_test",
    srcs = ["cuda_forward_kinematics_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    data = [
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_forward_kinematics",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_collision_checker_test",
    srcs = ["cuda_collision_checker_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/cpp:collision_checker",
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_collision_checker",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_python(
    name = "edge_inflation_zero_order_test",
    srcs = ["edge_inflation_zero_order_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    linkopts = [],
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_edge_inflation_zero_order",
        "//csdecomp/src/cuda:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_python(
    name = "polytope_builder_test",
    srcs = ["polytope_builder_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    linkopts = [],
    deps = [
        "//csdecomp/src/cpp:collision_checker",
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_collision_checker",
        "//csdecomp/src/cuda:cuda_hit_and_run_sampling",
        "//csdecomp/src/cuda:cuda_polytope_builder",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_python(
    name = "cuda_visibility_graph_test",
    srcs = ["cuda_visibility_graph_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    linkopts = [],
    deps = [
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_collision_checker",
        "//csdecomp/src/cuda:cuda_hit_and_run_sampling",
        "//csdecomp/src/cuda:cuda_visibility_graph",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_python(
    name = "cuda_hit_and_run_sampling_test",
    srcs = ["cuda_hit_and_run_sampling_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    data = [],
    linkopts = [],
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cuda:cuda_hit_and_run_sampling",
        "//csdecomp/src/cuda:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_distance_field_test",
    srcs = ["cuda_distance_field_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    data = [
        "//csdecomp/tests/test_assets:voxel_grids",
    ],
    deps = [
        "//csdecomp/src/cuda:cuda_distance_field",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test_with_python(
    name = "cuda_set_builder_utils_test",
    srcs = ["cuda_set_builder_utils_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    data = [],
    linkopts = [],
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cuda:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)
