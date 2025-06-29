load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("//:tools/lint.bzl", "add_lint_tests")
load("@//:tools/my_python_version.bzl", "get_my_system_python_copts")
load("//:tools/cc_with_system_python.bzl", "cc_test_with_system_python")

add_lint_tests()

cc_test(
    name = "urdf_parser_test",
    srcs = ["urdf_parser_test.cpp"],
    copts = ["-Icsdecomp/src/cpp"],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/cpp:urdf_parser",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kinematic_tree_test",
    srcs = ["kinematic_tree_test.cpp"],
    copts = ["-Icsdecomp/src/cpp"],
    data = [
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/cpp:urdf_parser",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "minimal_kinematic_tree_test",
    srcs = ["minimal_kinematic_tree_test.cpp"],
    copts = ["-Icsdecomp/src/cpp"],
    data = [
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/cpp:urdf_parser",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "collision_checker_test",
    srcs = ["collision_checker_test.cpp"],
    copts = ["-Icsdecomp/src/cpp"],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/cpp:collision_checker",
        "//csdecomp/src/cpp:urdf_parser",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "hpolyhedron_test",
    srcs = ["hpolyhedron_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
    ],
    data = [],
    linkopts = [],
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "build_roadmap_test",
    srcs = ["build_roadmap_test.cpp"],
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
        "//csdecomp/src/cpp:drm_planner",
        "//csdecomp/src/cpp:roadmap_builder",
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_hit_and_run_sampling",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "linesegment_aabb_checker_test",
    srcs = ["linesegment_aabb_checker_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
    ],
    data = [],
    linkopts = [],
    deps = [
        "//csdecomp/src/cpp:linesegment_aabb_checker",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "distance_aabb_linesegment_test",
    srcs = ["distance_aabb_linesegment_test.cpp"],
    copts = [
        "-Icsdecomp/src/cpp",
    ],
    data = [],
    linkopts = [],
    deps = [
        "//csdecomp/src/cpp:distance_aabb_linesegment",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "build_roadmap_test_interactive",
    srcs = ["build_roadmap_test.cpp"],
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
        "//csdecomp/src/cpp:drm_planner",
        "//csdecomp/src/cpp:roadmap_builder",
        "//csdecomp/src/cpp:urdf_parser",
        "//csdecomp/src/cuda:cuda_hit_and_run_sampling",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)
