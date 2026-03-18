load("@rules_cc//cc:defs.bzl", "cc_test")
load("//:tools/lint.bzl", "add_lint_tests")
load("@//:tools/my_python_version.bzl", "get_my_system_python_copts")
load("//:tools/cc_with_system_python.bzl", "cc_test_with_system_python")

add_lint_tests()

cc_test(
    name = "bezier_curve_test",
    srcs = ["bezier_curve_test.cpp"],
    copts = ["-Icsdecomp/src"],
    deps = [
        "//csdecomp/src/planning:bezier_curve",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "build_roadmap_test",
    srcs = ["build_roadmap_test.cpp"],
    copts = ["-Icsdecomp/src"],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/geometry:cuda_hit_and_run_sampling",
        "//csdecomp/src/planning:drm_planner",
        "//csdecomp/src/planning:roadmap_builder",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "build_roadmap_test_interactive",
    srcs = ["build_roadmap_test.cpp"],
    copts = ["-Icsdecomp/src"] + get_my_system_python_copts(),
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    linkopts = [],
    tags = ["manual"],
    deps = [
        "//csdecomp/src/geometry:cuda_hit_and_run_sampling",
        "//csdecomp/src/planning:drm_planner",
        "//csdecomp/src/planning:roadmap_builder",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "edge_inflation_zero_order_test",
    srcs = ["edge_inflation_zero_order_test.cpp"],
    copts = ["-Icsdecomp/src"],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/planning:cuda_edge_inflation_zero_order",
        "//csdecomp/src/planning:cuda_set_builder_utils",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "edge_inflation_zero_order_test_interactive",
    srcs = ["edge_inflation_zero_order_test.cpp"],
    copts = ["-Icsdecomp/src"] + get_my_system_python_copts(),
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    linkopts = [],
    tags = ["manual"],
    deps = [
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/planning:cuda_edge_inflation_zero_order",
        "//csdecomp/src/planning:cuda_set_builder_utils",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "polytope_builder_test",
    srcs = ["polytope_builder_test.cpp"],
    copts = ["-Icsdecomp/src"],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/geometry:cuda_hit_and_run_sampling",
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/planning:cuda_polytope_builder",
        "//csdecomp/src/plant:collision_checker",
        "//csdecomp/src/plant:cuda_collision_checker",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "polytope_builder_test_interactive",
    srcs = ["polytope_builder_test.cpp"],
    copts = ["-Icsdecomp/src"] + get_my_system_python_copts(),
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    linkopts = [],
    tags = ["manual"],
    deps = [
        "//csdecomp/src/geometry:cuda_hit_and_run_sampling",
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/planning:cuda_polytope_builder",
        "//csdecomp/src/plant:collision_checker",
        "//csdecomp/src/plant:cuda_collision_checker",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_visibility_graph_test",
    srcs = ["cuda_visibility_graph_test.cpp"],
    copts = ["-Icsdecomp/src"],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/geometry:cuda_hit_and_run_sampling",
        "//csdecomp/src/planning:cuda_visibility_graph",
        "//csdecomp/src/plant:cuda_collision_checker",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "cuda_visibility_graph_test_interactive",
    srcs = ["polytope_builder_test.cpp"],
    copts = ["-Icsdecomp/src"] + get_my_system_python_copts(),
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    linkopts = [],
    tags = ["manual"],
    deps = [
        "//csdecomp/src/geometry:cuda_hit_and_run_sampling",
        "//csdecomp/src/planning:cuda_visibility_graph",
        "//csdecomp/src/plant:cuda_collision_checker",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_set_builder_utils_test",
    srcs = ["cuda_set_builder_utils_test.cpp"],
    copts = ["-Icsdecomp/src"],
    deps = [
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/planning:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "cuda_set_builder_utils_test_interactive",
    srcs = ["cuda_set_builder_utils_test.cpp"],
    copts = ["-Icsdecomp/src"] + get_my_system_python_copts(),
    data = [],
    linkopts = [],
    tags = ["manual"],
    deps = [
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/planning:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_edit_regions_test",
    srcs = ["cuda_edit_regions_test.cpp"],
    copts = ["-Icsdecomp/src"],
    deps = [
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/planning:cuda_edit_regions",
        "//csdecomp/src/planning:cuda_set_builder_utils",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "cuda_edit_regions_test_interactive",
    srcs = ["cuda_edit_regions_test.cpp"],
    copts = ["-Icsdecomp/src"] + get_my_system_python_copts(),
    data = [],
    linkopts = [],
    tags = ["manual"],
    deps = [
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/planning:cuda_edit_regions",
        "//csdecomp/src/plant:urdf_parser",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)
