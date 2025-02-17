from utils import CSD_EXAMPLES_ROOT, get_fake_voxel_map, densify_waypoints
from pydrake.all import (
    StartMeshcat,
    RobotDiagramBuilder,
    LoadModelDirectives,
    ProcessModelDirectives,
    VisualizationConfig,
    ApplyVisualizationConfig,
    Rgba,
    SceneGraphCollisionChecker,
    QueryObject,
    SceneGraphInspector,
    RigidTransform
)
from cspace_utils.plotting import plot_points, plot_triad
import pycsdecomp as csd
import numpy as np
import time

directives_path = CSD_EXAMPLES_ROOT+\
    '/../csdecomp/tests/test_assets/directives/kinova_sens_on_table.yml'
meshcat = StartMeshcat()
rbuilder = RobotDiagramBuilder()
plant= rbuilder.plant()
builder = rbuilder.builder()
parser = rbuilder.parser()
parser.package_map().Add("test_assets", "../csdecomp/tests/test_assets")
parser.package_map().Add("kortex_description", 
                         "../csdecomp/tests/test_assets/kinova/kortex_description")
parser.package_map().Add("robotiq_arg85_description", 
                         "../csdecomp/tests/test_assets/robotiq")

directives = LoadModelDirectives(directives_path)
models = ProcessModelDirectives(directives, plant, parser)
scene_graph = rbuilder.scene_graph()
config = VisualizationConfig()
config.enable_alpha_sliders = False
config.publish_contacts = False
config.publish_inertia = False
config.default_proximity_color = Rgba(0.8,0,0,0.5)
plant.Finalize()
ApplyVisualizationConfig(config, builder, meshcat=meshcat)
diagram = rbuilder.Build()
diagram_context = diagram.CreateDefaultContext()
diagram.ForcedPublish(diagram_context)
kin_idx = plant.GetModelInstanceByName("kinova")
rob_idx = plant.GetModelInstanceByName("robotiq")
checker = SceneGraphCollisionChecker(model = diagram,
                           robot_model_instances = [kin_idx, rob_idx],
                           edge_step_size = 0.1)
scene_graph_context = scene_graph.GetMyContextFromRoot(diagram_context)
plant_context = plant.GetMyContextFromRoot(diagram_context)
query : QueryObject = scene_graph.get_query_output_port()\
    .Eval(scene_graph_context)
inspector : SceneGraphInspector = query.inspector()
meshcat.SetProperty('/drake/proximity', 'visible', True)

csd_plant = csd.convert_drake_plant_to_csd_plant(plant, inspector)

pl_opts = csd.DrmPlannerOptions()
pl_opts.max_iterations_steering_to_node = 10
pl_opts.max_nodes_to_expand = int(1e4)
pl_opts.try_shortcutting = True
pl_opts.online_edge_step_size = 0.01
pl_opts.max_number_planning_attempts = 20
pl_opts.voxel_padding = 0.01
drm_planner = csd.DrmPlanner(csd_plant, pl_opts)
try:
    drm_planner.LoadRoadmap(CSD_EXAMPLES_ROOT + '/tmp/kinova_rm.map')
except:
    raise ValueError(
    " Could not load roadmap.Make sure you have run the 'build_DRM.py example")

locs, cols, radius = get_fake_voxel_map()
plot_points(meshcat, locs, 'fake voxels', size=radius, color = cols)

csd_vox = csd.Voxels(locs.T)
drm_planner.BuildCollisionSet(csd_vox)

start_config = np.array([-0.79714745,  1.8618815 , -2.0683706 ,  0.36096397, -2.404603  ,
        1.9386684 , -1.823328  ])
plant.SetPositions(plant_context, start_config)
diagram.ForcedPublish(diagram_context)

goal_pose = RigidTransform(np.array([0.6, -0.5, 0.2]))
goal_config = drm_planner.GetClosestNonCollidingConfigurationsByPose(goal_pose.GetAsMatrix4(), 100, 0.2)[0]
succ, plan = drm_planner.Plan(start_config, goal_config,csd_vox, radius)
plot_triad(goal_pose, meshcat, 'goal_pose', size= 0.1)

wps = densify_waypoints(plan, plant, plant_context)
input("Press 'Enter' to play the motion plan")
for w in wps:
    plant.SetPositions(plant_context, w)
    diagram.ForcedPublish(diagram_context)
    time.sleep(0.01)
print('here')
