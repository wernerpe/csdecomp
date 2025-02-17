load("//:tools/buildifier_test.bzl", "buildifier_test")

load("//:tools/clang_format_test.bzl", "clang_format_test")

def add_lint_tests():
    buildifier_test(
    name = "buildifier_test",
    srcs = native.glob(
        [
            "BUILD",
            "WORKSPACE",
            "MODULE.bazel",
            "*BUILD",
            "*bzl"
        ],
        allow_empty = True,
        exclude = [
            "bazel-*/**",  # Exclude bazel-* symlink directories
        ],
      ),
    )
    clang_format_test(
      name = "cpp_lint_test",
      srcs = native.glob(
        ["*.cpp", 
         "*.h", 
         "*.cu"
        ],
        allow_empty = True,
      ),
      clang_format_file = "//:clang_format_config"
    )