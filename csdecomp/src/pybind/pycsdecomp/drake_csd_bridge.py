from pydrake.all import (MultibodyPlant,
                         SceneGraphInspector,
                         ModelInstanceIndex,
                         Role,
                         Box,
                         Sphere)
from typing import Union
import pycsdecomp as csd
import numpy as np


def set_joint_type_by_drake_name(drake_joint_type_name: str)->csd.JointType:
    if drake_joint_type_name == 'weld':
        return csd.JointType(0)
    if drake_joint_type_name == 'revolute':
        return csd.JointType(1)
    if drake_joint_type_name == 'prismatic':
        return csd.JointType(2)
    raise NotImplementedError(f"The joint type \
                              conversions from {drake_joint_type_name}\
                to CSD compatible joints has not been implemented yet.")

def extract_shape_dimensions(shape : Union[Box, Sphere]):
        if isinstance(shape, Sphere):
            return np.array([shape.radius(),0,0]).reshape(3,1)
        if isinstance(shape, Box):
            return shape.size().reshape(3,1)
        raise NotImplementedError(f" Shape {shape} is not supported")

def extract_shape_info(shape : Union[Box, Sphere]):
    if isinstance(shape, Sphere):
        return csd.ShapePrimitive.SPHERE, \
            np.array([shape.radius(),0,0]).reshape(3,1)
   
    if isinstance(shape, Box):
        return csd.ShapePrimitive.BOX, shape.size().reshape(3,1)
    raise NotImplementedError(f" Shape {shape} is not supported")

def get_compound_name(model_name, body_name):
    if model_name == "WorldModelInstance":
        return body_name
    if model_name == "DefaultModelInstance":
        return body_name
    else:
        return f"{model_name}::{body_name}"
    
def convert_drake_plant_to_csd_plant(drake_plant : MultibodyPlant,
                                     inspector : SceneGraphInspector)\
                                    ->csd.Plant:    
    csd_KT = csd.KinematicTree()
    model_instance_indices = [ModelInstanceIndex(i) \
                              for i in range(
                                  drake_plant.num_model_instances())]
    all_drake_bodies = []
    csd_links = []
    for miidx in model_instance_indices:
        bidxs = drake_plant.GetBodyIndices(miidx)
        for bid in bidxs:
            all_drake_bodies.append(drake_plant.get_body(bid))
            if all_drake_bodies[-1].is_floating():
                raise NotImplementedError('floating bases not \
                                          supported yet.')
            csd_link = csd.Link()
            #will be set later when processing the joints
            csd_link.parent_joint = -1
            csd_link.name = get_compound_name(
                drake_plant.GetModelInstanceName(miidx), 
                all_drake_bodies[-1].name())
            csd_links.append(csd_link)

    for csdl in csd_links:
        csd_KT.add_link(csdl)

    joint_idx = drake_plant.GetJointIndices()
    csd_joints = []
    all_drake_joints = []
    for jid in joint_idx:
        joint = drake_plant.get_joint(jid)
        lower = joint.position_lower_limits()
        upper = joint.position_upper_limits()
        
        all_drake_joints.append(joint)
        local_parent_id = all_drake_bodies.index(joint.parent_body())
        local_child_id = all_drake_bodies.index(joint.child_body())
        # print(f"drake parent body {all_drake_bodies[local_parent_id]} \
        #       csd parent {csd_links[local_parent_id].name}")
        csd_joint = csd.Joint()
        csd_joint.name = joint.name()
        csd_joint.type = set_joint_type_by_drake_name(joint.type_name())
        if csd_joint.type != csd.JointType.FIXED:
            csd_joint.position_lower_limit = lower
            csd_joint.position_upper_limit = upper
            
        csd_joint.parent_link = csd_KT.get_link_index(
            csd_links[local_parent_id].name)
        csd_joint.child_link = csd_KT.get_link_index(
            csd_links[local_child_id].name)
        if csd_joint.type == csd.JointType.REVOLUTE:
            csd_joint.axis = joint.revolute_axis()
            assert np.isclose(np.linalg.norm(csd_joint.axis), 1) 
        if csd_joint.type == csd.JointType.PRISMATIC:
            csd_joint.axis = joint.translation_axis()
            assert np.isclose(np.linalg.norm(csd_joint.axis), 1) 
        frame_idx = joint.frame_on_parent().index()
        csd_joint.X_PL_J = drake_plant.get_frame(frame_idx)\
            .GetFixedPoseInBodyFrame().GetAsMatrix4()
        csd_joints.append(csd_joint)
        csd_links[local_parent_id].child_joints.append(len(csd_joints)-1)
        csd_links[local_child_id].parent_joint = len(csd_joints)-1
    for j in csd_joints:
        csd_KT.add_joint(j)

    csd_KT.finalize()
    from pydrake.all import Role

    frame_ids = inspector.GetAllFrameIds()
    link_name_to_geom_ids = {}
    geometries = {}

    scene_collision_geometries =[]

    for id in frame_ids:
        name = inspector.GetName(id)
        link_name_to_geom_ids[name] = inspector.GetGeometries(
            id, Role.kProximity) 
        link_index = csd_KT.get_link_index(name)
        for g in link_name_to_geom_ids[name]:
            shape = inspector.GetShape(g)
            X_L_B = inspector.GetPoseInFrame(g).GetAsMatrix4()
            geometries[g] = [shape, X_L_B]
            csd_g = csd.CollisionGeometry()
            csd_g.link_index = link_index
            csd_g.X_L_B = X_L_B
            sh_type, sh_dimensions = extract_shape_info(shape)
            csd_g.dimensions = sh_dimensions
            csd_g.type = sh_type
            scene_collision_geometries.append(csd_g)

    # Hijacking parser functionalities to simplify the extracted tree using 
    # a finalize_scene call.
    csd_parser = csd.URDFParser()
    csd_parser.override_kinematic_tree(csd_KT)
    csd_parser.override_scene_collision_geometries(
        scene_collision_geometries)
    csd_parser.finalize_scene()
    csd_plant = csd_parser.build_plant()
    return csd_plant