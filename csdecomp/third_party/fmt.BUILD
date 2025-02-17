cc_library(
    name = "fmt",
    srcs = glob(["src/*.cc"], exclude = ["src/fmt.cc"]),
    hdrs = glob(["include/fmt/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    copts = ["-DFMT_HEADER_ONLY=1"],
)