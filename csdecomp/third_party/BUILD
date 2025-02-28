load("@//:tools/embedded_py.bzl", "cc_py_library")
load("@pip//:requirements.bzl", "requirement")
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_py_library(
    name = "matplotlib_cpp",
    hdrs = ["matplotlibcpp.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@rules_python//python/cc:current_py_cc_headers",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
    py_deps=[
        requirement("matplotlib"),
        requirement("pygobject"),
        requirement("numpy")]
)

cc_library(
    name = "matplotlib_cpp_interactive",
    srcs = ["matplotlibcpp.h"],
    hdrs = ["matplotlibcpp.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)