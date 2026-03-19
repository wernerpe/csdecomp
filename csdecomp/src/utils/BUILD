load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("//:tools/lint.bzl", "add_lint_tests")

add_lint_tests()

cuda_library(
    name = "cuda_utilities",
    hdrs = ["cuda_utilities.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@local_cuda//:cuda_runtime",
    ],
)
