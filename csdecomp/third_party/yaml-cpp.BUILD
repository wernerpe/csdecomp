cc_library(
    name = "yaml-cpp",
    srcs = glob([
        "src/*.cpp",
        "src/*.h",
    ]),
    hdrs = glob([
        "include/yaml-cpp/**/*.h",
        "include/yaml-cpp/*.h",
    ]),
    includes = ["include"],
    copts = ["-I$(GENDIR)/external/yaml-cpp/include"],
    visibility = ["//visibility:public"],
)