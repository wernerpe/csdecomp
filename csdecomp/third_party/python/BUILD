load("@python//3.10:defs.bzl", compile_pip_requirements_3_10 = "compile_pip_requirements")
load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

compile_pip_requirements_3_10(
  name = "requirements_3_10",
  requirements_in=":requirements_3_10.in",
  requirements_txt=":requirements_3_10.txt",
  extra_args=["--allow-unsafe"],
  timeout="eternal",
)

py_binary(
  name = "hello_world",
  srcs = ["hello_world.py"],
  deps = [
    requirement("numpy"),
    requirement("matplotlib")
  ]
)