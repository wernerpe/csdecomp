
# Adding pip dependencies

We use a hermetic python instance so that we have the same python version running everywhere. In
order to bring in pip packages, add your desired package and version to "requirements.in". Then run
`bazel run //third_party/python:requirements.update` to update `requirements.txt`.

# Using a Pip Dependency

To use a pip dependency for your targets, do the following in your build file.
```
load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "mylib",
  src = ["mylib.py"],
  deps = [
    requirement("numpy"),
  ]
)


```

credit: Erick Fuentes