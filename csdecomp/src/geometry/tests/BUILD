load("@rules_cc//cc:defs.bzl", "cc_test")
load("//:tools/lint.bzl", "add_lint_tests")
load("@//:tools/my_python_version.bzl", "get_my_system_python_copts")
load("//:tools/cc_with_system_python.bzl", "cc_test_with_system_python")

add_lint_tests()

cc_test(
    name = "hpolyhedron_test",
    srcs = ["hpolyhedron_test.cpp"],
    copts = ["-Icsdecomp/src"],
    deps = [
        "//csdecomp/src/geometry:hpolyhedron",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "linesegment_aabb_checker_test",
    srcs = ["linesegment_aabb_checker_test.cpp"],
    copts = ["-Icsdecomp/src"],
    deps = [
        "//csdecomp/src/geometry:linesegment_aabb_checker",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "distance_aabb_linesegment_test",
    srcs = ["distance_aabb_linesegment_test.cpp"],
    copts = ["-Icsdecomp/src"],
    deps = [
        "//csdecomp/src/geometry:distance_aabb_linesegment",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_distance_field_test",
    srcs = ["cuda_distance_field_test.cpp"],
    copts = ["-Icsdecomp/src"],
    data = [
        "//csdecomp/tests/test_assets:voxel_grids",
    ],
    deps = [
        "//csdecomp/src/geometry:cuda_distance_field",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_test(
    name = "cuda_hit_and_run_sampling_test",
    srcs = ["cuda_hit_and_run_sampling_test.cpp"],
    copts = ["-Icsdecomp/src"],
    deps = [
        "//csdecomp/src/geometry:cuda_hit_and_run_sampling",
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/planning:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_with_system_python(
    name = "cuda_hit_and_run_sampling_test_interactive",
    srcs = ["cuda_hit_and_run_sampling_test.cpp"],
    copts = ["-Icsdecomp/src"] + get_my_system_python_copts(),
    data = [],
    linkopts = [],
    tags = ["manual"],
    deps = [
        "//csdecomp/src/geometry:cuda_hit_and_run_sampling",
        "//csdecomp/src/geometry:hpolyhedron",
        "//csdecomp/src/planning:cuda_set_builder_utils",
        "//csdecomp/tests:plotting_utils_interactive",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "omp_test",
    srcs = ["omp_test.cpp"],
    copts = [
        "-fopenmp",
        "-O2",
    ],
    linkopts = [
        "-fopenmp",
    ],
)
