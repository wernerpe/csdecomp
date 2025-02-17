import os
import numpy as np

from pycsdecomp import (URDFParser, 
                    UniformSampleInHPolyhedraCuda,
                    )
from pycsdecomp import (
    URDFParser,
    DRM, 
    RoadmapOptions, 
    RoadmapBuilder, 
    Plant,
    Link,
    KinematicTree,
    UniformSampleInHPolyhedraCuda,
    HPolyhedron
    )

from utils import CSD_EXAMPLES_ROOT, get_drm_summary

parser = URDFParser()
parser.register_package("test_assets", "../csdecomp/tests/test_assets/")
parser.parse_directives("../csdecomp/tests/test_assets/directives/simple_kinova_sens_on_table.yml")
plant: Plant = parser.build_plant()

domain = HPolyhedron()
domain.MakeBox(plant.getPositionLowerLimits(), plant.getPositionUpperLimits())

rm_opts = RoadmapOptions()
rm_opts.robot_map_size_x = 1.5
rm_opts.robot_map_size_y = 2.
rm_opts.robot_map_size_z = 1.
rm_opts.map_center = np.array([0.3,0,0.5])
rm_opts.max_task_space_distance_between_nodes = 0.85
rm_opts.max_configuration_distance_between_nodes = 4.0
rm_opts.nodes_processed_before_debug_statement = 300
rm_opts.offline_voxel_resolution = 0.02
rm_opts.edge_step_size = 0.1     

configs = UniformSampleInHPolyhedraCuda([domain], 
                                        domain.ChebyshevCenter().reshape(-1,1), 
                                        8000, 
                                        100)[0]

# Select the frame to build the pose map for.
kt : KinematicTree = plant.getKinematicTree()
links : list[Link] = kt.get_links()

print("Available Link names for pose map")
for l in links:
    print(l.name)

rm_builder = RoadmapBuilder(plant,"kinova::end_effector_link", rm_opts)
rm_builder.add_nodes_manual(configs)
rm_builder.build_roadmap(max_neighbors= 20)


#rm_builder.read('tmp/kinova40k.map')
drm : DRM =rm_builder.get_drm()
get_drm_summary(drm, do_plot=False)
try:
    os.mkdir(CSD_EXAMPLES_ROOT+'/tmp')
except FileExistsError:
    print("Directory already exists")
rm_builder.write('tmp/kinova_rm_tmp_pre_pose_map.map')
rm_builder.build_pose_map()
rm_builder.write('tmp/kinova_rm_tmp_post_pose_map.map')
rm_builder.build_collision_map()
rm_builder.write('tmp/kinova_rm.map')
