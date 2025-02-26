load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("//:tools/lint.bzl", "add_lint_tests")

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
