"""Test script for EI-ZO edge inflation (extracted from kinova.ipynb).

Requires a GPU.
  Standalone: cd examples && uv run python test_eizo.py
  Bazel:      bazel test //examples:test_eizo
"""

import os
from csdecomp import (
    URDFParser,
    EizoOptions,
    CudaEdgeInflator,
    UniformSampleInHPolyhedraCuda,
    Voxels,
    CheckCollisionFreeVoxelsCuda,
    HPolyhedron as csdHPoly,
)
import numpy as np

# Resolve test asset paths for both Bazel and standalone execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("csdecomp/tests/test_assets"):
    # Running under Bazel (cwd is workspace root)
    ASSET_PATH = "csdecomp/tests/test_assets"
else:
    # Running standalone from examples/
    ASSET_PATH = os.path.join(SCRIPT_DIR, "..", "csdecomp", "tests", "test_assets")

# Build plant using CSD's own parser
csdecomp_parser = URDFParser()
csdecomp_parser.register_package("test_assets", ASSET_PATH)
csdecomp_parser.parse_directives(
    os.path.join(ASSET_PATH, "directives", "kinova_sens_on_table.yml")
)
kt = csdecomp_parser.get_kinematic_tree()
mp = csdecomp_parser.get_minimal_plant()
insp = csdecomp_parser.get_scene_inspector()

# Set up voxels
vox_loc = np.array(
    [[0.3, 0, 0.5], [0.3, 0.4, 0.5], [0.3, 0.4, 0.6], [0.3, -0.4, 0.5]]
).T
vox_radius = 0.02
csdecomp_vox = Voxels(vox_loc)

# Set up domain and EI-ZO options
csdecomp_domain = csdHPoly()
csdecomp_domain.MakeBox(kt.get_position_lower_limits(), kt.get_position_upper_limits())

options = EizoOptions()
options.num_particles = 10000
options.bisection_steps = 9
options.configuration_margin = 0.1
options.delta = 0.01
options.epsilon = 0.01
options.tau = 0.5
options.mixing_steps = 100
options.max_hyperplanes_per_iteration = 10

edge_inflator = CudaEdgeInflator(
    mp, insp.robot_geometry_ids, options, csdecomp_domain
)

# Run EI-ZO
l_start = np.array([-0.376, 0.3714, 0.0, 0.0, 0.0, 0.0, 0.0])
l_end = np.array([-0.376, 1.2024, 0.0, 0.0, 0.0, 0.0, 0.0])

# Burn-in run
region = edge_inflator.inflateEdge(l_start, l_end, csdecomp_vox, vox_radius, False)
# Timed run
region = edge_inflator.inflateEdge(l_start, l_end, csdecomp_vox, vox_radius, True)

# Verify the region contains the line segment endpoints
A = region.A()
b = region.b()
assert np.all(A @ l_start <= b + 1e-6), "Start point not in region"
assert np.all(A @ l_end <= b + 1e-6), "End point not in region"

# Sample points in the region and verify they are collision-free
samples = UniformSampleInHPolyhedraCuda(
    [region], region.ChebyshevCenter(), 10000, 100
)[0]
col_free = CheckCollisionFreeVoxelsCuda(
    samples, csdecomp_vox, vox_radius, mp, insp.robot_geometry_ids
)
collision_rate = 1.0 - np.mean(col_free)
print(f"Region has {A.shape[0]} hyperplanes")
print(f"Collision rate in region samples: {collision_rate:.4f}")
assert collision_rate < 0.02, f"Too many collisions in region: {collision_rate:.2%}"
print("Success!")
