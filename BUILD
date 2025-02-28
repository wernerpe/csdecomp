# BUILD
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "clang_format_config",
    srcs = [".clang-format"],
    visibility = ["//visibility:public"],
)

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

bool_flag(
    name = "compile_interactive_tests",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)