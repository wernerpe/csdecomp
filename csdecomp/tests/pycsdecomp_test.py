import unittest
import pycsdecomp as csd
import numpy as np
import sys

from pydrake.all import (RobotDiagramBuilder,
                         LoadModelDirectives,
                         ProcessModelDirectives,
                         SceneGraphCollisionChecker,
                         QueryObject, 
                         SceneGraphInspector)

class TestBasic(unittest.TestCase):
    def test_link(self):
        link = csd.Link()
        link.name = "elbow"
        self.assertEqual(link.name, "elbow")
        print('hello world my python bindings are working :))')
        print(f" numpy version {np.__version__}")
        print(f"location of python interpreter {sys.executable}")

class TestDrakeCSDBridge(unittest.TestCase):
    def test_drake_csd_bridge(self):
        asset_path = "csdecomp/tests/test_assets"
        directives_path = asset_path + "/directives/simple_kinova_sens_on_table.yml"
        rbuilder = RobotDiagramBuilder()
        plant = rbuilder.plant()
        #builder = rbuilder.builder()
        parser = rbuilder.parser()
        scene_graph = rbuilder.scene_graph()
        parser.package_map().Add("test_assets", asset_path)
        parser.package_map().Add("kortex_description", asset_path + "/kinova/kortex_description")
        parser.package_map().Add("robotiq_arg85_description", asset_path + "/robotiq")
        directives = LoadModelDirectives(directives_path)
        models = ProcessModelDirectives(directives, plant, parser)
        plant.Finalize()
        diagram = rbuilder.Build()
        kin_idx = plant.GetModelInstanceByName("kinova")
        rob_idx = plant.GetModelInstanceByName("robotiq")
        checker = SceneGraphCollisionChecker(model = diagram,
                                             robot_model_instances = [kin_idx, rob_idx],
                                             edge_step_size = 0.1
                                            )
        diagram_context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyMutableContextFromRoot(diagram_context)
        diagram.ForcedPublish(diagram_context)
        plant_context = plant.GetMyContextFromRoot(diagram_context)
        scene_graph_context = scene_graph.GetMyContextFromRoot(diagram_context)
        query : QueryObject = scene_graph.get_query_output_port().Eval(scene_graph_context)
        inspector : SceneGraphInspector = query.inspector()
        csd_plant = csd.convert_drake_plant_to_csd_plant(plant, plant_context, inspector)

        #generate a bunch of random configurations and test that we get the same results
        domain = csd.HPolyhedron()
        domain.MakeBox(csd_plant.getPositionLowerLimits(), csd_plant.getPositionUpperLimits())
        N = 100
        samples = csd.UniformSampleInHPolyhedraCuda([domain], 
                                                     domain.ChebyshevCenter(), 
                                                     N, 
                                                     100, 
                                                     1337)[0]

        res_csd = csd.CheckCollisionFreeCuda(samples, csd_plant.getMinimalPlant())
        res_drake = checker.CheckConfigsCollisionFree(samples.T, parallelize=True)
        for r_csd, r_drake in zip(res_csd, res_drake):
            self.assertEqual(r_csd, r_drake)

if __name__ == "__main__":
  unittest.main()