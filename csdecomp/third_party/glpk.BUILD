
cc_library(
    name = "glpk",
    srcs = glob([
        "src/*.cpp",
        "src/**/*.c",
        "src/*.h",
        "src/**/*.h",
    ]),
    hdrs = glob([
        "src/*.h",
        "src/**/*.h",
    ]),
    includes = ["src",
                "src/amd",
                "src/api",
                "src/bflib",
                "src/colamd",
                "src/draft",
                "src/env",
                "src/intopt",
                "src/minisat",
                "src/misc",
                "src/mpl",
                "src/npp",
                "src/proxy",
                "src/simplex",
                "src/zlib",
                ], 
    copts = ["-I$(GENDIR)/external/glpk/src"],
    visibility = ["//visibility:public"],
)