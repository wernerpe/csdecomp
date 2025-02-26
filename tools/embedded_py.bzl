def _cc_py_runtime_impl(ctx):
    toolchain = ctx.toolchains["@bazel_tools//tools/python:toolchain_type"]
    py3_runtime = toolchain.py3_runtime
    imports = []
    for dep in ctx.attr.deps:
        imports.append(dep[PyInfo].imports)
    python_path = ""
    for path in depset(transitive = imports).to_list():
        print("Printing python path: " + str(path))
        python_path += "external/" + path + ":"

    py3_runfiles = ctx.runfiles(files = py3_runtime.files.to_list())
    runfiles = [py3_runfiles]
    for dep in ctx.attr.deps:
        dep_runfiles = ctx.runfiles(files = dep[PyInfo].transitive_sources.to_list())
        runfiles.append(dep_runfiles)
        runfiles.append(dep[DefaultInfo].default_runfiles)

    runfiles = ctx.runfiles().merge_all(runfiles)

    print("Printing interpreter path: " + str(py3_runtime.interpreter.path))
    print("Printing interpreter home: " + str(py3_runtime.interpreter.dirname.rstrip("bin")))

    return [
        DefaultInfo(runfiles = runfiles),
        platform_common.TemplateVariableInfo({
            "PYTHON3": str(py3_runtime.interpreter.path),
            "PYTHONPATH": python_path,
            "PYTHONHOME": str(py3_runtime.interpreter.dirname.rstrip("bin")),
        }),
    ]

_cc_py_runtime = rule(
    implementation = _cc_py_runtime_impl,
    attrs = {
        "deps": attr.label_list(providers = [PyInfo]),
    },
    toolchains = [
        str(Label("@bazel_tools//tools/python:toolchain_type")),
    ],
)

def cc_py_test(name, py_deps = [], **kwargs):
    py_runtime_target = name + "_py_runtime"
    _cc_py_runtime(
        name = py_runtime_target,
        deps = py_deps,
    )

    kwargs.update({
        "data": kwargs.get("data", []) + [":" + py_runtime_target],
        "env": {"__PYVENV_LAUNCHER__": "$(PYTHON3)", "PYTHONPATH": "$(PYTHONPATH)", "PYTHONHOME": "$(PYTHONHOME)"},
        "toolchains": kwargs.get("toolchains", []) + [":" + py_runtime_target],
    })

    native.cc_test(
        name = name,
        **kwargs
    )

def cc_py_binary(name, py_deps = [], **kwargs):
    py_runtime_target = name + "_py_runtime"
    _cc_py_runtime(
        name = py_runtime_target,
        deps = py_deps,
    )

    kwargs.update({
        "data": kwargs.get("data", []) + [":" + py_runtime_target],
        "env": {"__PYVENV_LAUNCHER__": "$(PYTHON3)", "PYTHONPATH": "$(PYTHONPATH)", "PYTHONHOME": "$(PYTHONHOME)"},
        "toolchains": kwargs.get("toolchains", []) + [":" + py_runtime_target],
    })

    native.cc_binary(
        name = name,
        **kwargs
    )

def cc_py_library(name, py_deps = [], **kwargs):
    py_runtime_target = name + "_py_runtime"
    _cc_py_runtime(
        name = py_runtime_target,
        deps = py_deps,
    )

    kwargs.update({
        "data": kwargs.get("data", []) + [":" + py_runtime_target],
        "defines": [
            "FOO=\\\"BAR\\\"",
            "CPP_PYVENV_LAUNCHER=\\\"$(PYTHON3)\\\"",
            "CPP_PYTHON_PATH=\\\"$(PYTHONPATH)\\\"",
            "CPP_PYTHON_HOME=\\\"$(PYTHONHOME)\\\"",
            "WITHOUT_NUMPY",
        ],
        "toolchains": kwargs.get("toolchains", []) + [":" + py_runtime_target],
    })

    native.cc_library(
        name = name,
        **kwargs
    )