load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_test")
load("//:tools/lint.bzl", "add_lint_tests")
load("//:tools/cc_with_system_python.bzl", "cc_binary_with_system_python")
load("//:tools/embedded_py.bzl", "cc_py_library")
load("@//:tools/my_python_version.bzl", "get_my_system_python_copts")

add_lint_tests()

cc_library(
    name = "plotting_utils",
    srcs = ["plotting_utils.h"],
    hdrs = ["plotting_utils.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/third_party:matplotlib_cpp",
        "@eigen",
    ],
)

cc_library(
    name = "plotting_utils_interactive",
    srcs = ["plotting_utils.h"],
    hdrs = ["plotting_utils.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
        "//csdecomp/third_party:matplotlib_cpp_interactive",
        "@eigen",
    ],
)

py_test(
    name = "pycsdecomp_test",
    srcs = ["pycsdecomp_test.py"],
    deps = [
        "//csdecomp/src/pybind/pycsdecomp:pycsdecomp_lib_internal",
    ],
)

cc_binary(
    name = "test_plot",
    srcs = ["test_matplotlib.cpp"],
    deps = [":plotting_utils"],
)

cc_binary_with_system_python(
    name = "test_plot_interactive",
    srcs = ["test_matplotlib.cpp"],
    copts = get_my_system_python_copts(),
    deps = [":plotting_utils_interactive"],
)
