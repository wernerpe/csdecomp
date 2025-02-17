load("@//:tools/my_python_version.bzl", "get_my_python_version")

cc_library(
    name = "matplotlib_cpp",
    hdrs = ["matplotlibcpp.h"],
    includes = [".", "/usr/include/python{}".format(get_my_python_version())],
    visibility = ["//visibility:public"],
    copts = ["-I/usr/include/python{}".format(get_my_python_version())]
)