"""Test script for the Drake-CSD bridge (extracted from using_drake_csd_bridge.ipynb).

Requires a GPU.
  Standalone: cd examples && uv run python test_drake_bridge.py
  Bazel:      bazel test //examples:test_drake_bridge
"""

import os
from pydrake.all import (
    RobotDiagramBuilder,
    LoadModelDirectives,
    ProcessModelDirectives,
    SceneGraphCollisionChecker,
    SceneGraphInspector,
    QueryObject,
)

from csdecomp import (
    convert_drake_plant_to_csd_plant,
    HPolyhedron,
    UniformSampleInHPolyhedraCuda,
    CheckCollisionFreeCuda,
)
import time

# Resolve test asset paths for both Bazel and standalone execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("csdecomp/tests/test_assets"):
    # Running under Bazel (cwd is workspace root)
    ASSET_PATH = "csdecomp/tests/test_assets"
else:
    # Running standalone from examples/
    ASSET_PATH = os.path.join(SCRIPT_DIR, "..", "csdecomp", "tests", "test_assets")

directives_path = os.path.join(ASSET_PATH, "directives", "kinova_sens_on_table.yml")

# Build drake plant
rbuilder = RobotDiagramBuilder()
plant = rbuilder.plant()
parser = rbuilder.parser()
scene_graph = rbuilder.scene_graph()
parser.package_map().Add("test_assets", ASSET_PATH)
parser.package_map().Add(
    "kortex_description", os.path.join(ASSET_PATH, "kinova", "kortex_description")
)
parser.package_map().Add(
    "robotiq_arg85_description", os.path.join(ASSET_PATH, "robotiq")
)

directives = LoadModelDirectives(directives_path)
models = ProcessModelDirectives(directives, plant, parser)
plant.Finalize()
diagram = rbuilder.Build()
diagram_context = diagram.CreateDefaultContext()
kin_idx = plant.GetModelInstanceByName("kinova")
rob_idx = plant.GetModelInstanceByName("robotiq")
checker = SceneGraphCollisionChecker(
    model=diagram, robot_model_instances=[kin_idx, rob_idx], edge_step_size=0.1
)
scene_graph_context = scene_graph.GetMyContextFromRoot(diagram_context)
plant_context = plant.GetMyContextFromRoot(diagram_context)
query: QueryObject = scene_graph.get_query_output_port().Eval(scene_graph_context)
inspector: SceneGraphInspector = query.inspector()

# Convert to CSD plant
csd_plant = convert_drake_plant_to_csd_plant(plant, plant_context, inspector)

# Sample and compare collision checking
n_samples = 100000
domain = HPolyhedron()
domain.MakeBox(csd_plant.getPositionLowerLimits(), csd_plant.getPositionUpperLimits())
samples = UniformSampleInHPolyhedraCuda(
    polyhedra=[domain],
    starting_points=domain.ChebyshevCenter(),
    num_samples_per_hpolyhedron=n_samples,
    mixing_steps=100,
)[0]

t0 = time.time()
res_csd = CheckCollisionFreeCuda(samples, csd_plant.getMinimalPlant())
t_csd = time.time() - t0

t0 = time.time()
res_drake = checker.CheckConfigsCollisionFree(samples.T, parallelize=True)
t_drake = time.time() - t0

number_differing_results = 0
for rcsd, rdr in zip(res_csd, res_drake):
    if rcsd != rdr:
        number_differing_results += 1

print(f"Checked {n_samples} configurations")
print(f"CSD time:   {t_csd:.3f}s")
print(f"Drake time: {t_drake:.3f}s")
print(f"Differing results: {number_differing_results}")

# Allow small differences due to drake's padding
assert number_differing_results < n_samples * 0.01, (
    f"Too many differing results: {number_differing_results}/{n_samples}"
)
print("Success!")
