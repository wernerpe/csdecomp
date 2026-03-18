load("@rules_cc//cc:defs.bzl", "cc_test")
load("//:tools/lint.bzl", "add_lint_tests")

add_lint_tests()

cc_test(
    name = "urdf_parser_test",
    srcs = ["urdf_parser_test.cpp"],
    copts = ["-Icsdecomp/src"],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/plant:urdf_parser",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kinematic_tree_test",
    srcs = ["kinematic_tree_test.cpp"],
    copts = ["-Icsdecomp/src"],
    data = [
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/plant:urdf_parser",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "minimal_kinematic_tree_test",
    srcs = ["minimal_kinematic_tree_test.cpp"],
    copts = ["-Icsdecomp/src"],
    data = [
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/plant:urdf_parser",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "collision_checker_test",
    srcs = ["collision_checker_test.cpp"],
    copts = ["-Icsdecomp/src"],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/plant:collision_checker",
        "//csdecomp/src/plant:urdf_parser",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_forward_kinematics_test",
    srcs = ["cuda_forward_kinematics_test.cpp"],
    copts = ["-Icsdecomp/src"],
    data = [
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/plant:cuda_forward_kinematics",
        "//csdecomp/src/plant:urdf_parser",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_collision_checker_test",
    srcs = ["cuda_collision_checker_test.cpp"],
    copts = ["-Icsdecomp/src"],
    data = [
        "//csdecomp/tests/test_assets:directives",
        "//csdecomp/tests/test_assets:urdf_files",
    ],
    deps = [
        "//csdecomp/src/plant:collision_checker",
        "//csdecomp/src/plant:cuda_collision_checker",
        "//csdecomp/src/plant:urdf_parser",
        "@com_google_googletest//:gtest_main",
    ],
)
