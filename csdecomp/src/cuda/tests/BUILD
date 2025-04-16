load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("@pip//:requirements.bzl", "requirement")
load("//:tools/lint.bzl", "add_lint_tests")
load("@//:tools/my_python_version.bzl", "get_my_system_python_copts")
load("//:tools/cc_with_system_python.bzl", "cc_test_with_system_python")

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

cc_test(
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
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_edge_inflation_zero_order",
        "//csdecomp/src/cuda:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

# This builds the same test but links it with the system python,
# such that the binary can be directly executed for interactive debugging.
cc_test_with_system_python(
    name = "edge_inflation_zero_order_test_interactive",
    srcs = ["edge_inflation_zero_order_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ] + get_my_system_python_copts(),
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    linkopts = [],
    tags = ["manual"],  #only mannualy build for interactive debugging
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_edge_inflation_zero_order",
        "//csdecomp/src/cuda:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
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

cc_test_with_system_python(
    name = "polytope_builder_test_interactive",
    srcs = ["polytope_builder_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ] + get_my_system_python_copts(),
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    linkopts = [],
    tags = ["manual"],
    deps = [
        "//csdecomp/src/cpp:collision_checker",
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_collision_checker",
        "//csdecomp/src/cuda:cuda_hit_and_run_sampling",
        "//csdecomp/src/cuda:cuda_polytope_builder",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
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
    deps = [
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_collision_checker",
        "//csdecomp/src/cuda:cuda_hit_and_run_sampling",
        "//csdecomp/src/cuda:cuda_visibility_graph",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "cuda_visibility_graph_test_interactive",
    srcs = ["polytope_builder_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ] + get_my_system_python_copts(),
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    linkopts = [],
    tags = ["manual"],  #only mannualy build for interactive debugging
    deps = [
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_collision_checker",
        "//csdecomp/src/cuda:cuda_hit_and_run_sampling",
        "//csdecomp/src/cuda:cuda_visibility_graph",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_hit_and_run_sampling_test",
    srcs = ["cuda_hit_and_run_sampling_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cuda:cuda_hit_and_run_sampling",
        "//csdecomp/src/cuda:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "cuda_hit_and_run_sampling_test_interactive",
    srcs = ["cuda_hit_and_run_sampling_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ] + get_my_system_python_copts(),
    data = [],
    linkopts = [],
    tags = ["manual"],  #only mannualy build for interactive debugging
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cuda:cuda_hit_and_run_sampling",
        "//csdecomp/src/cuda:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils_interactive",
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

cc_test(
    name = "cuda_set_builder_utils_test",
    srcs = ["cuda_set_builder_utils_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    data = [],
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cuda:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "cuda_set_builder_utils_test_interactive",
    srcs = ["cuda_set_builder_utils_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ] + get_my_system_python_copts(),
    data = [],
    linkopts = [],
    tags = ["manual"],  #only mannualy build for interactive debugging
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cuda:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_edit_regions_test",
    srcs = ["cuda_edit_regions_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    data = [],
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_edit_regions",
        "//csdecomp/src/cuda:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "cuda_edit_regions_test_interactive",
    srcs = ["cuda_edit_regions_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ] + get_my_system_python_copts(),
    data = [],
    linkopts = [],
    tags = ["manual"],  #only mannualy build for interactive debugging
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_edit_regions",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)
