# load("@python//:defs.bzl", "compile_pip_requirements")
load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@pip//:requirements.bzl", "requirement")

compile_pip_requirements(
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
    requirement("matplotlib"),
    requirement("PyGObject")
  ]
)