# CSDecomp: Configuration Space Decomposition Toolbox

CSDecomp is a Python package that implements a simple GPU-accelerated collision checker and GPU-accelerated algorithms for computing approximate convex decompositions of robot configuration spaces. The package provides implementations of Dynamic Roadmaps (DRMs) and the Edge Inflation Zero-Order (EI-ZO) algorithm in cuda/cpp as described in our paper ["Superfast Configuration-Space Convex Set Computation on GPUs for Online Motion Planning"](https://arxiv.org/pdf/2504.10783).

Contributions are welcome!
## Installation

### From PyPI (recommended)

```bash
pip install csdecomp
```

Requires an NVIDIA GPU with a compatible driver. The wheel bundles the CUDA runtime.

### From source

1. Install the CUDA toolchain (12.x recommended): https://developer.nvidia.com/cuda-toolkit-archive

    (Your display driver must be compatible with the installed CUDA version.)

2. Install prereqs:

    Install bazel via bazelisk:
    [bazelisk instructions](https://github.com/bazelbuild/bazelisk/blob/master/README.md)

    Other prereqs:
    `sudo bash setup.sh`

3. Build and test:

    `bazel test //...`

    (If you are getting cudaMalloc errors, make sure there aren't any big applications running in the background.)

### Usage

```python
import csdecomp as csd
```

The build also outputs a pip-installable wheel at `bazel-bin/csdecomp/src/pybind/csdecomp/`.

To change Python version, edit `tools/my_python_version.bzl` and `MODULE.bazel`.

The unit tests demonstrate how the code should be used. The Python bindings closely follow the C++ syntax.
There is experimental documentation that can be built with doxygen: `cd csdecomp/docs/ && doxygen Doxyfile`.

# Running the Python Examples

1. Install poetry `pip install poetry` 

2. Set automatic naming `poetry config virtualenvs.create true; poetry config virtualenvs.in-project true`

3. `cd examples && poetry install`

4. Run the exapmles! 
E.g. `poetry shell && python minimal_test.py`

    For the notebooks make sure to select the kernel corresponding to the venv created by poetry. If you are using vscode, you may need to open the examples folder speparately, e.g. `cd examples && code .`, for it to detect and list the kernel automatically.

# Developing

For interactive debugging of C++ code with plotting, the targets need to depend on the system python. Make sure your system python has matplotlib installed along with the dev headers:

    `test -f /usr/include/python3.10/Python.h && echo "exists" || echo "not found"`

The `cc_test_with_system_python` targets (tagged `manual`) in the test BUILD files demonstrate this setup.



# Citation

If you find this code useful, please consider citing our paper:

```
@article{werner2024superfast,
  title={Superfast Configuration-Space Convex Set Computation on GPUs for Online Motion Planning},
  author={Werner, Peter and Cheng, Richard and Stewart, Tom and Tedrake, Russ and Rus, Daniela},
  journal={arXiv preprint arXiv:2504.10783},
  year={2025}
}
```

# Miscellaneous

Random build command lookuptable

`bazel build //...`

`bazel build //csdecomp/src/pybind/csdecomp:csdecomp_wheel`

`bazel test //csdecomp/tests:csdecomp_test`

Using a bashrc approach for adding csdecomp to the python path

`export PYTHONPATH=$(bazel info bazel-bin)/csdecomp/src/pybind/csdecomp:$PYTHONPATH`

Random profilig commands:

`nsys profile bazel-bin/csdecomp/tests/cuda_collision_checker_test`

`sudo $(which ncu) --set full --target-processes all -o tmp/nsightcompute_report bazel-bin/csdecomp/tests/cuda_collision_checker_test`

Drake slow to launch and returning LCM test failed:

```sudo ifconfig lo multicast & sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev lo ```


Running list of TODOs
* Improve forward kinematics efficiency by removing fixed joints from the evaluation and switching to 3x4 representation of the transforms
* Handle other python versions gracefully
* Handle cylinder and capsule collision geometries
* Finish Python documentation using Sphinx