# CSDecomp: Configuration Space Decomposition

CSDecomp is a Python package that implements GPU-accelerated algorithms for computing convex decompositions of robot configuration spaces. The package provides implementations of Dynamic Roadmaps (DRMs) and the Edge Inflation Zero-Order (EI-ZO) algorithm in cuda/cpp as described in our paper "Superfast Configuration-Space Convex Set Computation on GPUs for Online Motion Planning".

## Installation
Currently, the installation requires building from source using bazel. I have only tested the software using python 3.10 and building on Ubuntu22.04.

1. Install the cuda toolchain. I used cuda 12.6 and the 560 driver:

    `https://developer.nvidia.com/cuda-toolkit-archive`

    (Note: You will need to make sure that your display driver is compatible with the cuda version installed. Otherwise the code might compile but all of the kernel launches will do nothing.)

2. Install bazel and clang-fromat if needed:

    `sudo apt install bazelisk && sudo apt-get install clang-format`


3. Build and test the code to ensure that everything was installed correctly:
    
    From the root of this repository run `bazel test ...`

    (If you are getting cudaMalloc errors, make sure there aren't any big applications running in the background.)

Notes: 

The build outputs a pip installable wheel at `bazel-bin/csdecomp/src/pybind/pycsdecomp` and a directory named `pycsdecomp` that can be appended to the python path and used directly like so:

```
import sys
sys.path.append(f'{PATH_TO_REPO}/bazel-bin/csdecomp/src/pybind/pycsdecomp')
import pycsdecomp as csd
```
The wheel and this folder contain a pystubs file for type hints. Make sure to point to those directories to simplfy code usage. 


If you want to try to change python version, it will need to be changed here:
1. https://github.com/wernerpe/cuciv0/blob/feature/drake_compat_layer/tools/my_python_version.bzl#L2-L5
2. https://github.com/wernerpe/cuciv0/blob/feature/drake_compat_layer/MODULE.bazel#L15

In general, the unit tests demonstrate how the code should be used. The python bindings closely follow the C++ syntax.

# Running the Python Examples

1. Install poetry `pip install poetry` 

2. Set automatic naming `poetry config virtualenvs.create true; poetry config virtualenvs.in-project true`

3. `cd examples && poetry install`

4. Run the exapmles! 
E.g. `poetry shell && python minimal_test.py`

    For the notebooks make sure to select the kernel corresponding to the venv created by poetry. If you are using vscode, you may need to open the examples folder speparately, e.g. `cd examples && code .`, for it to detect and list the kernel automatically.

# Citation

If you find this code useful, please consider citing our paper:

```
@article{werner2024superfast,
  title={Superfast Configuration-Space Convex Set Computation on GPUs for Online Motion Planning},
  author={Werner, Peter and Cheng, Richard and Stewart, Tom and Tedrake, Russ and Rus, Daniela},
  journal={TBD},
  year={2025}
}
```

# Miscellaneous
We developped the code using vscode. For convenience, I added the ".vscode" folder that shows how to launch the cpp unit tests interactively for debugging.

Random build command lookuptable

`bazel build //...`

`bazel build //csdecomp/pybind:pycsdecompbindings`

`bazel test //csdecomp/tests:urdf_parser_test`

Using a bashrc approach for adding csdecomp to the python path

`export PYTHONPATH=$(bazel info bazel-bin)/csdecomp/src/pybind:$PYTHONPATH`

Random profilig commands:

`nsys profile bazel-bin/csdecomp/tests/cuda_collision_checker_test`

`sudo $(which ncu) --set full --target-processes all -o tmp/nsightcompute_report bazel-bin/csdecomp/tests/cuda_collision_checker_test`

Drake slow to launch and returning LCM test failed:

```sudo ifconfig lo multicast & sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev lo ```


Running list of TODOs
* Improve forward kinematics efficiency by removing fixed joints from the evaluation and switching to 3x4 representation of the transforms
* Handle other python versions gracefully
* Handle Cylinder and Capsule collision geometries