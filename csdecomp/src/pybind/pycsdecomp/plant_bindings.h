#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "urdf_parser.h"
#include "voxel_wrapper.h"

namespace py = pybind11;
using namespace csdecomp;

void add_plant_bindings(py::module &m) {
  py::class_<URDFParser>(m, "URDFParser")
      .def(py::init<>())
      .def("parse_urdf", &URDFParser::parseURDF)
      .def("parse_urdf_string", &URDFParser::parseURDFString)
      .def("register_package", &URDFParser::registerPackage)
      .def("parse_directives", &URDFParser::parseDirectives)
      .def("get_minimal_plant", &URDFParser::getMinimalPlant)
      .def("get_kinematic_tree", &URDFParser::getKinematicTree)
      .def("get_minimal_kinematic_tree", &URDFParser::getMinimalKinematicTree)
      .def("get_scene_collision_geometries",
           &URDFParser::getSceneCollisionGeometries,
           py::return_value_policy::reference_internal)
      .def("get_scene_inspector", &URDFParser::getSceneInspector)
      .def("override_kinematic_tree", &URDFParser::overrideKinematicTree)
      .def("override_scene_collision_geometries",
           &URDFParser::overrideSceneCollisionGeometries)
      .def("finalize_scene", &URDFParser::finalizeScene)
      .def("build_plant", &URDFParser::buildPlant);

  py::enum_<JointType>(m, "JointType")
      .value("FIXED", JointType::FIXED)
      .value("REVOLUTE", JointType::REVOLUTE)
      .value("PRISMATIC", JointType::PRISMATIC)
      .export_values();

  py::class_<KinematicTree>(m, "KinematicTree")
      .def(py::init<>())
      .def("add_link", &KinematicTree::addLink)
      .def("add_joint", &KinematicTree::addJoint)
      .def("finalize", &KinematicTree::finalize)
      .def("get_link_index", &KinematicTree::getLinkIndexByName)
      .def("get_link_name", &KinematicTree::getLinkNameByIndex)
      .def("get_link", &KinematicTree::getLink)
      .def("get_joint_index", &KinematicTree::getJointIndex)
      .def("get_joint", &KinematicTree::getJoint)
      .def("is_finalized", &KinematicTree::isFinalized)
      .def("get_leaf_link_indices", &KinematicTree::getLeafLinkIndices)
      .def("get_links", &KinematicTree::getLinks,
           py::return_value_policy::reference_internal)
      .def("get_joints", &KinematicTree::getJoints,
           py::return_value_policy::reference_internal)
      .def("get_position_lower_limits", &KinematicTree::getPositionLowerLimits)
      .def("get_position_upper_limits", &KinematicTree::getPositionUpperLimits)
      .def("compute_link_frame_to_world_transforms",
           &KinematicTree::computeLinkFrameToWorldTransforms)
      .def("get_minimal_kinematic_tree",
           &KinematicTree::getMyMinimalKinematicTreeStruct)
      .def("num_links", &KinematicTree::numLinks);

  py::class_<Joint>(m, "Joint")
      .def(py::init<>())
      .def_readwrite("name", &Joint::name)
      .def_readwrite("type", &Joint::type)
      .def_readwrite("X_PL_J", &Joint::X_PL_J)
      .def_readwrite("parent_link", &Joint::parent_link)
      .def_readwrite("child_link", &Joint::child_link)
      .def_readwrite("axis", &Joint::axis)
      .def_readwrite("position_lower_limit", &Joint::position_lower_limit)
      .def_readwrite("position_upper_limit", &Joint::position_upper_limit);

  py::class_<Link>(m, "Link")
      .def(py::init<>())
      .def_readwrite("name", &Link::name)
      .def_readwrite("parent_joint", &Link::parent_joint)
      .def_readwrite("child_joints", &Link::child_joints);

  py::class_<MinimalJoint>(m, "MinimalJoint")
      .def(py::init<>())
      .def_readwrite("type", &MinimalJoint::type)
      .def_readwrite("X_PL_J", &MinimalJoint::X_PL_J)
      .def_readwrite("index", &MinimalJoint::index)
      .def_readwrite("parent_link", &MinimalJoint::parent_link)
      .def_readwrite("child_link", &MinimalJoint::child_link)
      .def_readwrite("axis", &MinimalJoint::axis);

  py::class_<MinimalLink>(m, "MinimalLink")
      .def(py::init<>())
      .def_readwrite("index", &MinimalLink::index)
      .def_readwrite("parent_joint_index", &MinimalLink::parent_joint_index)
      .def_property(
          "child_joint_index_array",
          [](MinimalLink &l) {
            return py::array_t<int>(MAX_CHILD_JOINT_COUNT,
                                    l.child_joint_index_array);
          },
          [](MinimalLink &l, py::array_t<int> arr) {
            if (arr.size() > MAX_CHILD_JOINT_COUNT)
              throw std::runtime_error("Array too large");
            std::copy(arr.data(), arr.data() + arr.size(),
                      l.child_joint_index_array);
          })
      .def_readwrite("num_child_joints", &MinimalLink::num_child_joints);

  py::class_<MinimalKinematicTree>(m, "MinimalKinematicTree")
      .def(py::init<>())
      .def_property(
          "joints",
          [](MinimalKinematicTree &t) {
            py::list joints;
            for (int i = 0; i < t.num_joints; ++i) {
              joints.append(t.joints[i]);
            }
            return joints;
          },
          [](MinimalKinematicTree &t, py::list joints) {
            if (joints.size() > MAX_JOINTS_PER_TREE)
              throw std::runtime_error("List too large");
            t.num_joints = static_cast<int>(joints.size());
            for (int i = 0; i < t.num_joints; ++i) {
              t.joints[i] = joints[i].cast<MinimalJoint>();
            }
          })
      .def_property(
          "links",
          [](MinimalKinematicTree &t) {
            py::list links;
            for (int i = 0; i < t.num_links; ++i) {
              links.append(t.links[i]);
            }
            return links;
          },
          [](MinimalKinematicTree &t, py::list links) {
            if (links.size() > MAX_LINKS_PER_TREE)
              throw std::runtime_error("List too large");
            t.num_links = static_cast<int>(links.size());
            for (int i = 0; i < t.num_links; ++i) {
              t.links[i] = links[i].cast<MinimalLink>();
            }
          })
      .def_property(
          "joint_traversal_sequence",
          [](MinimalKinematicTree &t) {
            return py::array_t<int>(MAX_JOINTS_PER_TREE,
                                    t.joint_traversal_sequence);
          },
          [](MinimalKinematicTree &t, py::array_t<int> arr) {
            if (arr.size() > MAX_JOINTS_PER_TREE)
              throw std::runtime_error("Array too large");
            std::copy(arr.data(), arr.data() + arr.size(),
                      t.joint_traversal_sequence);
          })
      .def_property(
          "joint_idx_to_config_idx",
          [](MinimalKinematicTree &t) {
            return py::array_t<int>(MAX_JOINTS_PER_TREE,
                                    t.joint_idx_to_config_idx);
          },
          [](MinimalKinematicTree &t, py::array_t<int> arr) {
            if (arr.size() > MAX_JOINTS_PER_TREE)
              throw std::runtime_error("Array too large");
            std::copy(arr.data(), arr.data() + arr.size(),
                      t.joint_idx_to_config_idx);
          })
      .def_readwrite("num_configuration_variables",
                     &MinimalKinematicTree::num_configuration_variables)
      .def_readwrite("num_joints", &MinimalKinematicTree::num_joints)
      .def_readwrite("num_links", &MinimalKinematicTree::num_links);

  m.def(
      "computeLinkFrameToWorldTransformsMinimal",
      [](const Eigen::VectorXf &joint_values,
         const MinimalKinematicTree &minimal_tree) {
        Eigen::MatrixXf transforms =
            Eigen::MatrixXf::Zero(4 * minimal_tree.num_links, 4);
        computeLinkFrameToWorldTransformsMinimal(joint_values, minimal_tree,
                                                 &transforms);
        return transforms;
      },
      "Compute link frame to world transforms for a minimal kinematic tree",
      py::arg("joint_values"), py::arg("tree"));

  // Bind the ShapePrimitive enum
  py::enum_<ShapePrimitive>(m, "ShapePrimitive")
      .value("BOX", ShapePrimitive::BOX)
      .value("SPHERE", ShapePrimitive::SPHERE)
      .value("CYLINDER", ShapePrimitive::CYLINDER)
      .value("CAPSULE", ShapePrimitive::CAPSULE);

  // Bind the CollisionGroup enum
  py::enum_<CollisionGroup>(m, "CollisionGroup")
      .value("ROBOT", CollisionGroup::ROBOT)
      .value("WORLD", CollisionGroup::WORLD)
      .value("VOXELS", CollisionGroup::VOXELS);

  // Bind the CollisionGeometry struct
  py::class_<CollisionGeometry>(m, "CollisionGeometry")
      .def(py::init<>())
      .def_readwrite("type", &CollisionGeometry::type)
      .def_readwrite("dimensions", &CollisionGeometry::dimensions)
      .def_readwrite("link_index", &CollisionGeometry::link_index)
      .def_readwrite("group", &CollisionGeometry::group)
      .def_readwrite("X_L_B", &CollisionGeometry::X_L_B);

  // Bind the SceneInspector struct
  py::class_<SceneInspector>(m, "SceneInspector")
      .def(py::init<>())
      .def_readwrite(
          "link_index_to_scene_collision_geometry_indices",
          &SceneInspector::link_index_to_scene_collision_geometry_indices)
      .def_readwrite("robot_to_world_collision_pairs",
                     &SceneInspector::robot_to_world_collision_pairs)
      .def_readwrite("robot_geometry_ids", &SceneInspector::robot_geometry_ids)
      .def("get_collision_pair_matrix",
           &SceneInspector::getCollisionPairMatrix);

  // Bind the CollisionPairMatrix type
  py::class_<CollisionPairMatrix>(m, "CollisionPairMatrix",
                                  py::buffer_protocol())
      .def_buffer([](CollisionPairMatrix &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(), sizeof(GeometryIndex),
            py::format_descriptor<GeometryIndex>::format(), 2,
            {m.rows(), m.cols()},
            {sizeof(GeometryIndex), sizeof(GeometryIndex) * m.rows()});
      });

  py::class_<VoxelsWrapper>(m, "Voxels")
      .def(py::init<Voxels>())
      .def(py::init<>())
      .def("set_matrix", &VoxelsWrapper::setMatrix)
      .def("get_matrix", &VoxelsWrapper::getMatrix);

  py::class_<MinimalPlant>(m, "MinimalPlant")
      .def(py::init<>())
      .def_readwrite("kin_tree", &MinimalPlant::kin_tree)
      .def_property_readonly(
          "scene_collision_geometries",
          [](MinimalPlant &mp) {
            py::list result;
            for (int i = 0; i < mp.num_scene_geometries; ++i) {
              result.append(mp.scene_collision_geometries[i]);
            }
            return result;
          })
      .def_property_readonly("collision_pairs_flat",
                             [](MinimalPlant &mp) {
                               return py::array_t<GeometryIndex>(
                                   {mp.num_collision_pairs * 2},
                                   {sizeof(GeometryIndex)},
                                   mp.collision_pairs_flat, py::cast(mp));
                             })
      .def_readwrite("num_scene_geometries",
                     &MinimalPlant::num_scene_geometries)
      .def_readwrite("num_collision_pairs", &MinimalPlant::num_collision_pairs);

  py::class_<Plant>(m, "Plant")
      .def(py::init<const KinematicTree &, const SceneInspector &,
                    const std::vector<CollisionGeometry> &>())
      .def("numLinks", &Plant::numLinks)
      .def("numJoints", &Plant::numJoints)
      .def("getPositionLowerLimits", &Plant::getPositionLowerLimits)
      .def("getPositionUpperLimits", &Plant::getPositionUpperLimits)
      .def("numConfigurationVariables", &Plant::numConfigurationVariables)
      .def("getMinimalPlant", &Plant::getMinimalPlant)
      .def("getKinematicTree", &Plant::getKinematicTree,
           py::return_value_policy::reference_internal)
      .def("getSceneInspector", &Plant::getSceneInspector,
           py::return_value_policy::reference_internal)
      .def("getRobotGeometryIds", &Plant::getRobotGeometryIds,
           py::return_value_policy::reference_internal)
      .def("getSceneCollisionGeometries", &Plant::getSceneCollisionGeometries,
           py::return_value_policy::reference_internal)
      .def("getCollisionPairMatrix", &Plant::getCollisionPairMatrix)
      .def("getSceneCollisionGeometryNames",
           &Plant::getSceneCollisionGeometryNames)
      .def("getSceneCollisionGeometryIndexByName",
           &Plant::getSceneCollisionGeometryIndexByName);
}