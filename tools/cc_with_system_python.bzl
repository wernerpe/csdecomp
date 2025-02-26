load("//:tools/my_python_version.bzl", "get_my_python_version")
load("@rules_cc//cc:defs.bzl", "cc_test")

def cc_test_with_system_python(name, srcs, deps, copts, data, linkopts, **kwargs):
    python_version = get_my_python_version()
    python_include = "-I/usr/include/python{}".format(python_version)
    python_link = "-lpython{}".format(python_version)
    
    native.cc_test(
        name = name,
        srcs = srcs,
        copts = copts + [
            python_include,
        ],
        data = data,
        linkopts = linkopts + [python_link],
        deps = deps,
        **kwargs
    )

def cc_library_with_system_python(name, srcs, hdrs, deps, copts, linkopts, **kwargs):
    python_version = get_my_python_version()
    python_include = "-I/usr/include/python{}".format(python_version)
    python_link = "-lpython{}".format(python_version)
    
    native.cc_library(
        name = name,
        srcs = srcs,
        copts = copts + [
            python_include,
        ],
        linkopts = linkopts + [python_link],
        deps = deps,
        **kwargs
    )

def cc_binary_with_system_python(name, srcs, deps = [], copts = [], data = [], linkopts = [], **kwargs):
    python_version = get_my_python_version()
    python_include = "-I/usr/include/python{}".format(python_version)
    python_link = "-lpython{}".format(python_version)
    
    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = copts + [
            python_include,
        ],
        data = data,
        linkopts = linkopts + [python_link],
        deps = deps,
        **kwargs
    )