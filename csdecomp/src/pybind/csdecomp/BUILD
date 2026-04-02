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
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/plant:collision_geometry",
    ],
)

cc_library(
    name = "plant_bindings",
    srcs = ["plant_bindings.h"],
    hdrs = ["plant_bindings.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":voxel_wrapper",
        "//csdecomp/src/plant:urdf_parser",
    ],
)

cc_library(
    name = "collision_checker_bindings",
    srcs = ["collision_checker_bindings.h"],
    hdrs = ["collision_checker_bindings.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":voxel_wrapper",
        "//csdecomp/src/plant:collision_checker",
    ],
)

cc_library(
    name = "cpp_utils_bindings",
    srcs = ["cpp_utils_bindings.h"],
    hdrs = ["cpp_utils_bindings.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        ":voxel_wrapper",
        "//csdecomp/src/geometry:distance_aabb_linesegment",
        "//csdecomp/src/geometry:linesegment_aabb_checker",
        "//csdecomp/src/planning:bezier_curve",
    ],
)

cc_library(
    name = "hpolyhedron_bindings",
    srcs = ["hpolyhedron_bindings.h"],
    hdrs = ["hpolyhedron_bindings.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/geometry:hpolyhedron",
    ],
)

cc_library(
    name = "cuda_bindings",
    srcs = ["cuda_bindings.h"],
    hdrs = ["cuda_bindings.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/geometry:cuda_hit_and_run_sampling",
        "//csdecomp/src/planning:cuda_edit_regions",
        "//csdecomp/src/planning:cuda_plane_update",
        "//csdecomp/src/planning:cuda_polytope_builder",
        "//csdecomp/src/planning:cuda_visibility_graph",
        "//csdecomp/src/plant:cuda_collision_checker",
    ],
)

cc_library(
    name = "drm_bindings",
    srcs = ["drm_bindings.h"],
    hdrs = ["drm_bindings.h"],
    copts = ["-Icsdecomp/src"],
    visibility = ["//visibility:public"],
    deps = [
        "//csdecomp/src/planning:drm_planner",
    ],
)

pybind_extension(
    name = "pycsdecomp_bindings",
    srcs = ["pycsdecomp_bindings.cpp"],
    copts = ["-Icsdecomp/src"],
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
        requirement("numpy"),
    ],
)

genrule(
    name = "copy_init_and_generate_stubs",
    srcs = [
        "__init__.py",
        "py.typed",
        ":pycsdecomp_bindings",
        "drake_csd_bridge.py",
    ],
    outs = [
        "csdecomp/__init__.py",
        "csdecomp/py.typed",
        "csdecomp/drake_csd_bridge.py",
        "csdecomp/pycsdecomp_bindings.so",
        "csdecomp/pycsdecomp_bindings.pyi",
    ],
    cmd = """
        # Create directory
        mkdir -p $(RULEDIR)/csdecomp

        # Copy required files
        cp $(location __init__.py) $(location py.typed) \
           $(location drake_csd_bridge.py) \
           $(location :pycsdecomp_bindings) $(RULEDIR)/csdecomp/

        # Run stub generation using the tool (Bazel handles runfiles for tools)
        $(execpath :generate_stubs) $(RULEDIR)/csdecomp
    """,
    tools = [":generate_stubs"],
)

py_library(
    name = "csdecomp_lib",
    data = [":copy_init_and_generate_stubs"],
    imports = [".."],
    visibility = ["//visibility:public"],
)

py_library(
    name = "csdecomp_lib_internal",
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
    name = "csdecomp_package",
    deps = [":csdecomp_lib"],
)

py_wheel(
    name = "csdecomp_wheel",
    author = "Peter Werner",
    author_email = "wernerpe@mit.edu",
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    description_content_type = "text/markdown",
    description_file = "PYPI_README.md",
    distribution = "csdecomp",
    homepage = "https://github.com/wernerpe/csdecomp",
    license = "MIT",
    platform = select({
        "@platforms//os:linux": "manylinux_2_28_x86_64",
    }),
    python_tag = get_my_python_tag(),
    requires = [
        "numpy>=2.0.0",
    ],
    strip_path_prefixes = [
        "csdecomp/src/pybind/csdecomp",
    ],
    summary = "GPU-accelerated configuration space decomposition for robot motion planning",
    version = "0.1.2",
    deps = [
        ":csdecomp_package",
    ],
)
