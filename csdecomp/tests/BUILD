load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("//:tools/lint.bzl", "add_lint_tests")
# load("//:tools/cc_with_python.bzl", "cc_library_with_python")
load("//:tools/embedded_py.bzl", "cc_py_library")
add_lint_tests()

cc_py_library(
    name = "plotting_utils",
    srcs = ["plotting_utils.h"],
    hdrs = ["plotting_utils.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@matplotlib_cpp",
        "@rules_python//python/cc:current_py_cc_headers",
        "@rules_python//python/cc:current_py_cc_libs"
    ],
)

py_test(
    name = "pycsdecomp_test",
    srcs = ["pycsdecomp_test.py"],
    deps = [
        "//csdecomp/src/pybind/pycsdecomp:pycsdecomp_lib_internal",
    ],
)
