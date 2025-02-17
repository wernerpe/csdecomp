load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("//:tools/lint.bzl", "add_lint_tests")
load("//:tools/cc_with_python.bzl", "cc_library_with_python")

add_lint_tests()

cc_library_with_python(
    name = "plotting_utils",
    srcs = ["plotting_utils.h"],
    hdrs = ["plotting_utils.h"],
    copts = ["-Icsdecomp/tests"],
    includes = ["."],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@matplotlib_cpp",
    ],
)

py_test(
    name = "pycsdecomp_test",
    srcs = ["pycsdecomp_test.py"],
    deps = [
        "//csdecomp/src/pybind/pycsdecomp:pycsdecomp_lib_internal",
    ],
)
