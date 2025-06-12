load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("//:tools/lint.bzl", "add_lint_tests")
load("//:tools/my_python_version.bzl", "get_my_python_tag")
load("@pip//:requirements.bzl", "requirement")

add_lint_tests()

cc_library(
    name = "voxel_wrapper",
    srcs = ["voxel_wrapper.h"],
    hdrs = ["voxel_wrapper.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/cpp:collision_geometry",
    ],
)

cc_library(
    name = "plant_bindings",
    srcs = ["plant_bindings.h"],
    hdrs = ["plant_bindings.h"],
    copts = ["-Icsdecomp/src/cpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":voxel_wrapper",
        "//csdecomp/src/cpp:urdf_parser",
    ],
)

cc_library(
    name = "collision_checker_bindings",
    srcs = ["collision_checker_bindings.h"],
    hdrs = ["collision_checker_bindings.h"],
    copts = ["-Icsdecomp/src/cpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":voxel_wrapper",
        "//csdecomp/src/cpp:collision_checker",
    ],
)

cc_library(
    name = "cpp_utils_bindings",
    srcs = ["cpp_utils_bindings.h"],
    hdrs = ["cpp_utils_bindings.h"],
    copts = ["-Icsdecomp/src/cpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":voxel_wrapper",
        "//csdecomp/src/cpp:distance_aabb_linesegment",
        "//csdecomp/src/cpp:linesegment_aabb_checker",
    ],
)

cc_library(
    name = "hpolyhedron_bindings",
    srcs = ["hpolyhedron_bindings.h"],
    hdrs = ["hpolyhedron_bindings.h"],
    copts = ["-Icsdecomp/src/cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/cpp:hpolyhedron",
    ],
)

cc_library(
    name = "cuda_bindings",
    srcs = ["cuda_bindings.h"],
    hdrs = ["cuda_bindings.h"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/cuda:cuda_edit_regions",
        "//csdecomp/src/cuda:cuda_polytope_builder",
        "//csdecomp/src/cuda:cuda_visibility_graph",
    ],
)

cc_library(
    name = "drm_bindings",
    srcs = ["drm_bindings.h"],
    hdrs = ["drm_bindings.h"],
    copts = [
        "-Icsdecomp/src/cpp",
        "-Icsdecomp/src/cuda",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/cpp:drm_planner",
    ],
)

pybind_extension(
    name = "pycsdecomp_bindings",
    srcs = ["pycsdecomp_bindings.cpp"],
    copts = [
        "-Icsdecomp/src/cuda",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":collision_checker_bindings",
        ":cpp_utils_bindings",
        ":cuda_bindings",
        ":drm_bindings",
        ":hpolyhedron_bindings",
        ":plant_bindings",
    ],
)

py_binary(
    name = "generate_stubs",
    srcs = ["generate_stubs.py"],
    main = "generate_stubs.py",
    deps = [
        requirement("mypy"),
    ],
)

genrule(
    name = "copy_init_and_generate_stubs",
    srcs = [
        "__init__.py",
        "py.typed",
        ":pycsdecomp_bindings",
        ":generate_stubs",
        "drake_csd_bridge.py",
    ],
    outs = [
        "pycsdecomp/__init__.py",
        "pycsdecomp/py.typed",
        "pycsdecomp/drake_csd_bridge.py",
        "pycsdecomp/pycsdecomp_bindings.so",
        "pycsdecomp/pycsdecomp_bindings.pyi",
    ],
    cmd = """
        # Create directory
        mkdir -p $(RULEDIR)/pycsdecomp

        # Copy required files
        cp $(location __init__.py) $(location py.typed) \
           $(location drake_csd_bridge.py) \
           $(location :pycsdecomp_bindings) $(RULEDIR)/pycsdecomp/

        # Run stub generation with Bazel's Python
        # Extract just the first path from the locations expansion
        STUBS_SCRIPT="$$(echo $(locations :generate_stubs) | cut -d' ' -f1)"
        $$STUBS_SCRIPT $(RULEDIR)/pycsdecomp
    """,
)

py_library(
    name = "pycsdecomp_lib",
    data = [":copy_init_and_generate_stubs"],
    imports = [".."],
    visibility = ["//visibility:public"],
)

py_library(
    name = "pycsdecomp_lib_internal",
    srcs = [
        "__init__.py",
        "drake_csd_bridge.py",
    ],
    data = [
        "py.typed",
        ":pycsdecomp_bindings",
    ],
    imports = [".."],
    visibility = ["//visibility:public"],
)

py_package(
    name = "pycsdecomp_package",
    deps = [":pycsdecomp_lib"],
)

py_wheel(
    name = "pycsdecomp_wheel",
    distribution = "pycsdecomp",
    platform = select({
        "@platforms//os:linux": "manylinux2014_x86_64",
    }),
    python_tag = get_my_python_tag(),
    requires = [
        "setuptools",
        "numpy>=2.0.0",
    ],
    strip_path_prefixes = [
        "csdecomp/src/pybind/pycsdecomp",  # Adjust this path to match your project structure
    ],
    version = "0.1",
    deps = [
        ":pycsdecomp_package",
    ],
)
