# Running the Python Examples

1. Install poetry `pip install poetry` 

2. `cd examples && poetry install`

3. Run the exapmles! 
E.g. activate the environment via the command given with `poetry env activate` then run
     `python minimal_test.py`

    For the notebooks make sure to select the kernel corresponding to the venv created by poetry. If you are using vscode, you may need to open the examples folder speparately, e.g. `cd examples && code .`, for it to detect and list the kernel automatically.

# Example Index
### mimimal_test.py
Runs a simple test to test the installation.
### build_DRM.py
Example of how to use `csdecomp` to build a dynamic roadmap. You need to run this script if you want to use `kinova_mintime_planning.py`
### drake_csd_bridge.ipynb
Demonstrates how to construct a plant if you are given a drake plant. Currently, this only works if the drake plant only has revolute and prismatic joints, and the collision geometries are either boxes or spheres.
### kinova_mintime_planning.py
Gives an example of how to use SCSPlanning to optimizie smooth, minimum-time trajectories through a sequence of regions generated with EI-ZO, as described in the paper. Make sure to run `build_DRM.py` first, since the DRM is required to find an initial path.

### more examples
Check out the CPP unit tests for more examples, they should hopefully be clear. Otherwise feel free to reach out wernerpe __ at __ mit __ dot __ edu