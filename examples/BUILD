load("@rules_python//python:defs.bzl", "py_test")
load("@pip//:requirements.bzl", "requirement")
load("//:tools/lint.bzl", "add_lint_tests")

add_lint_tests()

py_test(
    name = "minimal_test",
    srcs = ["minimal_test.py"],
    deps = [
        "//csdecomp/src/pybind/csdecomp:csdecomp_lib_internal",
    ],
)

py_test(
    name = "test_eizo",
    srcs = ["test_eizo.py"],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/pybind/csdecomp:csdecomp_lib_internal",
    ],
)

py_test(
    name = "test_drake_bridge",
    srcs = ["test_drake_bridge.py"],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/pybind/csdecomp:csdecomp_lib_internal",
        requirement("drake"),
        requirement("scipy"),
    ],
)
