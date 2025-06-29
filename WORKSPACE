# WORKSPACE
# Load required Bazel rules
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Eigen (linear algebra library)
http_archive(
    name = "eigen",
    build_file = "@//csdecomp/third_party:eigen.BUILD",
    sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
    strip_prefix = "eigen-3.4.0",
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"],
)

http_archive(
    name = "tinyxml2",
    build_file = "@//csdecomp/third_party:tinyxml2.BUILD",
    sha256 = "cc2f1417c308b1f6acc54f88eb70771a0bf65f76282ce5c40e54cfe52952702c",
    strip_prefix = "tinyxml2-9.0.0",
    urls = ["https://github.com/leethomason/tinyxml2/archive/9.0.0.tar.gz"],
)

http_archive(
    name = "yaml-cpp",
    build_file = "@//csdecomp/third_party:yaml-cpp.BUILD",
    integrity = "sha256-Q+ap/LFGrYcVFfDQhzlH5dSXocnGDFjLECqXtHIIt8M=",
    strip_prefix = "yaml-cpp-yaml-cpp-0.7.0",
    urls = ["https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.7.0.tar.gz"],
)

http_archive(
    name = "cereal",
    build_file = "@//csdecomp/third_party:cereal.BUILD",
    integrity = "sha256-FqetmzG6WIDaxV1itdbyQ8PryNRqNRQUnla15+qB+F8=",
    strip_prefix = "cereal-1.3.2",
    urls = ["https://github.com/USCiLab/cereal/archive/v1.3.2.tar.gz"],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "353571c2440176ded91c2de6d6cd88ddd41401d14692ec1f99e35d013feda55a",
    strip_prefix = "googletest-release-1.11.0",
    urls = ["https://github.com/google/googletest/archive/release-1.11.0.zip"],
)

http_archive(
    name = "fmt",
    build_file = "@//csdecomp/third_party:fmt.BUILD",
    integrity = "sha256-Z0dELBiQZLhXM2AH3X+jqvWFEqoaCyuna/EYLu+wECU=",
    strip_prefix = "fmt-8.0.1",
    urls = ["https://github.com/fmtlib/fmt/archive/8.0.1.zip"],
)

http_archive(
    name = "glpk",
    build_file = "@//csdecomp/third_party:glpk.BUILD",
    integrity = "sha256-ShAT7rtQ9yj8YBvdgzsLKHAzPDs+WoFu66kh2VvsbxU=",
    strip_prefix = "glpk-5.0",
    urls = [
        "https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz",
    ],
)

http_archive(
    name = "com_google_protobuf",
    integrity = "sha256-O9eCiqWvSxO5nBkeix6ITr+prTcbDOJkYF00fxNdJWg=",
    strip_prefix = "protobuf-3.19.4",
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/v3.19.4.tar.gz",
    ],
)

http_archive(
    name = "io_bazel_rules_go",
    integrity = "sha256-bcLaerTPXXv8fJSXdrG3xzPwXlbtxLzZAiuySdLiqZY=",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.39.1/rules_go-v0.39.1.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.39.1/rules_go-v0.39.1.zip",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

go_rules_dependencies()

go_register_toolchains(version = "1.19.3")

protobuf_deps()

http_archive(
    name = "com_github_bazelbuild_buildtools",
    integrity = "sha256-rjTDRFFOCMI+kNoOLWy3APzSjoDALiPk1XFd3ctC97M=",
    strip_prefix = "buildtools-4.2.2",
    urls = [
        "https://github.com/bazelbuild/buildtools/archive/refs/tags/4.2.2.tar.gz",
    ],
)