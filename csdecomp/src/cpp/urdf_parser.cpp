#include "urdf_parser.h"

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace csdecomp {
namespace {
// bool isFixedJoint(const Joint &joint) { return joint.type ==
// JointType::FIXED; }
Eigen::MatrixXi createAdjacencyMatrix(const std::vector<Joint> &joints,
                                      const std::vector<Link> &links) {
  int num_links = links.size();
  Eigen::MatrixXi Xi = Eigen::MatrixXi::Zero(num_links, num_links);

  for (size_t i = 0; i < joints.size(); ++i) {
    const Joint &joint = joints[i];
    int parent = joint.parent_link;
    int child = joint.child_link;

    if (parent >= 0 && parent < num_links && child >= 0 && child < num_links) {
      Xi(parent, child) =
          i + 1;  // Store joint index + 1 (to avoid confusion with 0)
      Xi(child, parent) = i + 1;  // Assuming undirected graph
    }
  }

  return Xi;
}

std::vector<int> findPath(const Eigen::MatrixXi &adjacencyMatrix, int start,
                          int end) {
  int n = adjacencyMatrix.rows();
  std::vector<bool> visited(n, false);
  std::vector<int> parent(n, -1);
  std::queue<int> q;

  q.push(start);
  visited[start] = true;

  while (!q.empty()) {
    int current = q.front();
    q.pop();

    if (current == end) {
      // Path found, reconstruct it
      std::vector<int> path;
      for (int v = end; v != -1; v = parent[v]) {
        path.push_back(v);
      }
      std::reverse(path.begin(), path.end());
      return path;
    }

    for (int neighbor = 0; neighbor < n; ++neighbor) {
      if (adjacencyMatrix(current, neighbor) && !visited[neighbor]) {
        q.push(neighbor);
        visited[neighbor] = true;
        parent[neighbor] = current;
      }
    }
  }

  // No path found
  return {};
}

}  // namespace

URDFParser::URDFParser() {}

bool URDFParser::parseURDF(const std::string &urdf_file) {
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(urdf_file.c_str()) != tinyxml2::XML_SUCCESS) {
    std::string str;
    str = fmt::format("Failed to load URDF file: {}", urdf_file);
    throw std::runtime_error(str);
  }
  tinyxml2::XMLPrinter printer;
  doc.Print(&printer);
  return parseURDFString(std::string(printer.CStr()));
}

bool URDFParser::parseURDFString(const std::string &file_contents) {
  tinyxml2::XMLDocument doc;
  tinyxml2::XMLError error = doc.Parse(file_contents.c_str());
  if (error != tinyxml2::XML_SUCCESS) {
    std::string str;
    str = fmt::format("Failed to load URDF file: {}", file_contents);
    throw std::runtime_error(str);
  }

  const tinyxml2::XMLElement *robot = doc.RootElement();
  if (!robot || std::string(robot->Name()) != "robot") {
    std::cerr << "Root element is not 'robot'" << std::endl;
    return false;
  }
  Link world_link;
  world_link.name = "world";
  world_link.parent_joint = -1;
  kinematic_tree_.addLink(world_link);

  // First pass: parse links
  for (const tinyxml2::XMLElement *link = robot->FirstChildElement("link");
       link; link = link->NextSiblingElement("link")) {
    if (!parseLink(link)) {
      return false;
    }
  }

  // Second pass: parse joints
  for (const tinyxml2::XMLElement *joint = robot->FirstChildElement("joint");
       joint; joint = joint->NextSiblingElement("joint")) {
    if (!parseJoint(joint)) {
      return false;
    }
  }
  bool connected_to_world = false;
  auto joints = kinematic_tree_.getJoints();
  for (auto j : joints) {
    if (j.parent_link == 0) {
      connected_to_world = true;
      break;
    }
  }
  if (!connected_to_world) {
    throw std::runtime_error(
        "[CSDECOMP:URDFPARSER] The provided urdf is not connected to the "
        "'world' "
        "link via a joint. This is not supported. Please load this model via "
        "directives, or add a joint, e.g. a fixed joint connecting the model "
        "to the world frame.");
  }

  for (auto l : kinematic_tree_.getLinks()) {
    if (l.name != "world") {
      if (l.parent_joint < 0 || (size_t)l.parent_joint >= joints.size()) {
        std::string tmp;
        fmt::format(
            "[CSDECOMP:URDFPARSER] The provided urdf  has a branch of the "
            "kinematic tree with no valid parent. The link {} has "
            "parent joint {}, the number of joints is {}"
            "link via a joint. Make sure that all models are connected "
            "to the world frame.",
            l.name, l.parent_joint, joints.size());
        throw std::runtime_error(tmp);
      }
    }
  }
  kinematic_tree_.finalize();
  finalizeScene();
  return true;
}

KinematicTree URDFParser::getKinematicTree() const { return kinematic_tree_; }

const MinimalKinematicTree URDFParser::getMinimalKinematicTree() const {
  return kinematic_tree_.getMyMinimalKinematicTreeStruct();
}

const MinimalPlant URDFParser::getMinimalPlant() const {
  MinimalKinematicTree mtree = getMinimalKinematicTree();
  CollisionPairMatrix cpm = inspector_.getCollisionPairMatrix();

  // Calculate the number of geometries to copy
  int num_geometries =
      std::min(static_cast<int>(scene_collision_geometries.size()),
               MAX_NUM_STATIC_COLLISION_GEOMETRIES);

  // Create a MinimalPlant instance
  MinimalPlant plant;

  // Correctly copy the MinimalKinematicTree
  std::memcpy(plant.kin_tree.joints, mtree.joints,
              sizeof(MinimalJoint) * MAX_JOINTS_PER_TREE);
  std::memcpy(plant.kin_tree.links, mtree.links,
              sizeof(MinimalLink) * MAX_LINKS_PER_TREE);
  std::memcpy(plant.kin_tree.joint_traversal_sequence,
              mtree.joint_traversal_sequence,
              sizeof(int) * MAX_JOINTS_PER_TREE);
  std::memcpy(plant.kin_tree.joint_idx_to_config_idx,
              mtree.joint_idx_to_config_idx, sizeof(int) * MAX_JOINTS_PER_TREE);
  plant.kin_tree.num_configuration_variables =
      mtree.num_configuration_variables;
  plant.kin_tree.num_joints = mtree.num_joints;
  plant.kin_tree.num_links = mtree.num_links;
  plant.num_collision_pairs = cpm.cols();
  assert(plant.num_collision_pairs <=
         MAX_NUM_STATIC_COLLISION_GEOMETRIES *
             (MAX_NUM_STATIC_COLLISION_GEOMETRIES - 1) / 2);

  std::copy(cpm.data(), cpm.data() + 2 * cpm.cols(),
            plant.collision_pairs_flat);
  plant.num_scene_geometries = num_geometries;

  // Copy the collision geometries
  std::copy(scene_collision_geometries.begin(),
            scene_collision_geometries.begin() + num_geometries,
            plant.scene_collision_geometries);

  return plant;
}

bool URDFParser::parseJoint(const tinyxml2::XMLElement *joint_element) {
  std::string name = joint_element->FindAttribute("name")->Value();
  std::string type_str = joint_element->FindAttribute("type")->Value();

  JointType type;
  if (type_str == "fixed")
    type = JointType::FIXED;
  else if (type_str == "revolute")
    type = JointType::REVOLUTE;
  else if (type_str == "prismatic")
    type = JointType::PRISMATIC;
  else {
    std::cerr << "Unsupported joint type: " << type_str << std::endl;
    return false;
  }

  const tinyxml2::XMLElement *origin =
      joint_element->FirstChildElement("origin");
  // origin is the transform from the joint frame to the parent joint frame
  Eigen::Matrix4f X_PL_J = parseOrigin(origin);
  Eigen::Vector3f axis = Eigen::Vector3f::Zero();

  if (!type == JointType::FIXED) {
    const tinyxml2::XMLElement *axis_element =
        joint_element->FirstChildElement("axis");
    axis = parseAxis(axis_element);
  }

  const tinyxml2::XMLElement *parent =
      joint_element->FirstChildElement("parent");
  const tinyxml2::XMLElement *child = joint_element->FirstChildElement("child");

  if (!parent || !child) {
    std::cerr << "Joint is missing parent or child element" << std::endl;
    return false;
  }

  std::string parent_link = parent->FindAttribute("link")->Value();
  std::string child_link = child->FindAttribute("link")->Value();

  Joint joint;
  joint.type = type;
  joint.name = name;
  joint.X_PL_J = X_PL_J;
  joint.parent_link = kinematic_tree_.getLinkIndexByName(parent_link);
  joint.child_link = kinematic_tree_.getLinkIndexByName(child_link);
  joint.axis = axis;

  const tinyxml2::XMLElement *limits_elem =
      joint_element->FirstChildElement("limit");
  if (!limits_elem) {
    // No limit element, assume no limits
    joint.position_lower_limit = -std::numeric_limits<float>::max();
    joint.position_upper_limit = std::numeric_limits<float>::max();
  } else {
    double lower = 0.0, upper = 0.0;
    bool has_lower = limits_elem->QueryDoubleAttribute("lower", &lower) ==
                     tinyxml2::XML_SUCCESS;
    bool has_upper = limits_elem->QueryDoubleAttribute("upper", &upper) ==
                     tinyxml2::XML_SUCCESS;
    if (has_lower) {
      joint.position_lower_limit = lower;
    }
    if (has_upper) {
      joint.position_upper_limit = upper;
    }
  }

  kinematic_tree_.addJoint(joint);
  return true;
}

bool URDFParser::parseLink(const tinyxml2::XMLElement *link_element) {
  Link link;
  std::string name = link_element->Attribute("name");
  // indices of joints will be added later when the joints are  parsed
  link.name = name;
  link.parent_joint = -1;  // Will be set when adding joints

  kinematic_tree_.addLink(link);
  const int link_idx = kinematic_tree_.getLinkIndexByName(link.name);
  // Iterate through all collision elements
  if (link_element->FindAttribute("origin") != nullptr) {
    throw std::runtime_error(
        "[CSDECOMP:URDFParser:ParseLink] Global origin tags are not supported "
        "in "
        "links, please apply the transformation directly to the visual and "
        "collision geometries and remove the global origin tag for the link. ");
  }

  int num_geoms_link = 0;
  for (const tinyxml2::XMLElement *collision =
           link_element->FirstChildElement("collision");
       collision != nullptr;
       collision = collision->NextSiblingElement("collision")) {
    std::string col_name;
    if (collision->FindAttribute("name") != nullptr) {
      col_name = collision->FindAttribute("name")->Value() +
                 fmt::format("::{}", num_geoms_link + 1);
    } else {
      col_name = name + "::col_geom::" + fmt::format("{}", num_geoms_link + 1);
    }
    CollisionGeometry col_geom = parseCollisionGeometry(collision);
    col_geom.link_index = link_idx;
    scene_collision_geometry_name_to_id[col_name] =
        scene_collision_geometries.size();
    scene_collision_geometries.push_back(col_geom);
    num_geoms_link++;
  }

  return true;
}

Eigen::Matrix4f URDFParser::parseOrigin(
    const tinyxml2::XMLElement *origin_element) {
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  Eigen::Vector3f translation = Eigen::Vector3f::Zero();
  Eigen::Vector3f rotation = Eigen::Vector3f::Zero();

  if (origin_element) {
    const char *xyzStr = origin_element->FindAttribute("xyz")->Value();
    if (xyzStr) {
      std::istringstream xyzStream(xyzStr);
      float x, y, z;
      xyzStream >> x >> y >> z;
      translation.x() = x;
      translation.y() = y;
      translation.z() = z;

    } else {
      throw std::runtime_error(
          "[CSDECOMP:URDFPARSER] no position specified for joint origin");
    }
    if (origin_element->FindAttribute("rpy")) {
      const char *rpyStr = origin_element->FindAttribute("rpy")->Value();
      if (rpyStr) {
        std::istringstream rpyStream(rpyStr);
        float r, p, y;
        rpyStream >> r >> p >> y;
        rotation.x() = r;
        rotation.y() = p;
        rotation.z() = y;
      } else {
        throw std::runtime_error(
            "[CSDECOMP:URDFPARSER] error in orientation (rpy) "
            "specification for joint");
      }
    }  // else default to rpy = 0,0,0

  } else {
    rotation.x() = 0;
    rotation.y() = 0;
    rotation.z() = 0;
    translation.x() = 0;
    translation.y() = 0;
    translation.z() = 0;

    // throw std::runtime_error(
    //     "[CSDECOMP:URDFPARSER] failed to parse joint element check
    //     spelling");
  }

  Eigen::AngleAxisf rollAngle(rotation.x(), Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf pitchAngle(rotation.y(), Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf yawAngle(rotation.z(), Eigen::Vector3f::UnitZ());

  Eigen::Matrix3f rotationMatrix =
      (yawAngle * pitchAngle * rollAngle).toRotationMatrix();

  transform.block<3, 3>(0, 0) = rotationMatrix;
  transform.block<3, 1>(0, 3) = translation;

  return transform;
}

Eigen::Vector3f URDFParser::parseAxis(
    const tinyxml2::XMLElement *axis_element) {
  Eigen::Vector3f axis = Eigen::Vector3f::Identity();
  if (axis_element) {
    const char *xyzStr = axis_element->FindAttribute("xyz")->Value();
    std::istringstream xyzStream(xyzStr);
    float x, y, z;
    xyzStream >> x >> y >> z;
    axis.x() = x;
    axis.y() = y;
    axis.z() = z;
    axis.normalize();
  } else {
    throw std::runtime_error(
        "[CSDECOMP:URDFPARSER] Error: One or more revolute or "
        "prismatic joints have no axis specified!");
  }
  return axis;
}

CollisionGeometry URDFParser::parseCollisionGeometry(
    const tinyxml2::XMLElement *collision_element) {
  CollisionGeometry col_obj;

  // Default values
  col_obj.type = ShapePrimitive::BOX;
  col_obj.dimensions = Eigen::Vector3f::Zero();
  col_obj.group = CollisionGroup::ROBOT;
  col_obj.X_L_B = Eigen::Matrix4f::Identity();

  // Parse origin if present
  const tinyxml2::XMLElement *origin_element =
      collision_element->FirstChildElement("origin");
  if (origin_element) {
    col_obj.X_L_B = parseOrigin(origin_element);
  }
  // Parse geometry
  const tinyxml2::XMLElement *geometry_element =
      collision_element->FirstChildElement("geometry");
  if (geometry_element) {
    const tinyxml2::XMLElement *shape_element =
        geometry_element->FirstChildElement();
    if (shape_element) {
      std::string shape_name = shape_element->Name();

      if (shape_name == "box") {
        col_obj.type = ShapePrimitive::BOX;
        const char *xyzStr = shape_element->FindAttribute("size")->Value();
        std::istringstream xyzStream(xyzStr);
        float x, y, z;
        xyzStream >> x >> y >> z;
        col_obj.dimensions[0] = x;
        col_obj.dimensions[1] = y;
        col_obj.dimensions[2] = z;

      } else if (shape_name == "sphere") {
        col_obj.type = ShapePrimitive::SPHERE;
        float radius;
        shape_element->QueryFloatAttribute("radius", &radius);
        col_obj.dimensions = Eigen::Vector3f(radius, -1, -1);
      } else if (shape_name == "cylinder") {
        throw std::runtime_error("cylider not yet implemented");
        col_obj.type = ShapePrimitive::CYLINDER;
        float radius, length;
        shape_element->QueryFloatAttribute("radius", &radius);
        shape_element->QueryFloatAttribute("length", &length);
        col_obj.dimensions = Eigen::Vector3f(radius, length, radius);
      } else if (shape_name == "capsule") {
        col_obj.type = ShapePrimitive::CAPSULE;
        float radius, length;
        shape_element->QueryFloatAttribute("radius", &radius);
        shape_element->QueryFloatAttribute("length", &length);
        col_obj.dimensions = Eigen::Vector3f(radius, length, radius);
      }
    } else {
      throw std::runtime_error(
          "No valid geometry type specified. Spheres, "
          "Boxes, Cylinders and Capsules are supported.");
    }
  }

  return col_obj;
}

const std::vector<CollisionGeometry> &URDFParser::getSceneCollisionGeometries()
    const {
  return scene_collision_geometries;
}

const SceneInspector URDFParser::getSceneInspector() const {
  return inspector_;
}

const Plant URDFParser::buildPlant() const {
  if (!kinematic_tree_.isFinalized()) {
    throw std::runtime_error(
        "No URDF has been parsed yet, cannot build plant!");
  }
  return Plant(kinematic_tree_, inspector_, scene_collision_geometries);
}

void URDFParser::finalizeScene() {
  // 1. FINALIZE GEOMETRIES
  // 1.1 ensure all fields are correct (is parent link world etc)
  // 1.2 build scene inspector link to geom map
  for (int idx = 0; idx < kinematic_tree_.numLinks(); ++idx) {
    std::vector<int> col_geom_indices;
    inspector_.link_index_to_scene_collision_geometry_indices.push_back(
        col_geom_indices);
  }

  int geom_index = 0;
  for (CollisionGeometry co : scene_collision_geometries) {
    inspector_.link_index_to_scene_collision_geometry_indices.at(co.link_index)
        .push_back(geom_index);
    ++geom_index;
  }

  // Set collision groups for robot and world.
  // For every geometry if it is attached to the world frame with a fixed
  // joint we count it as part of the world and ignore its collisions with
  // voxels.

  for (CollisionGeometry &co : scene_collision_geometries) {
    Link co_link = kinematic_tree_.getLink(co.link_index);
    Joint prev_joint = kinematic_tree_.getJoint(co_link.parent_joint);
    while (true) {
      if (prev_joint.type != JointType::FIXED) {
        co.group = CollisionGroup::ROBOT;
        break;
      }
      if (prev_joint.parent_link == 0) {
        // if we arrive here we traced the kinematic tree to the world link
        // only through fixed joints
        co.group = CollisionGroup::WORLD;
        break;
      }
      int prev_joint_idx =
          kinematic_tree_.getLink(prev_joint.parent_link).parent_joint;
      prev_joint = kinematic_tree_.getJoint(prev_joint_idx);
    }
  }
  // find and store all ids of the robot geometries in the inspector
  int id = 0;
  for (auto cg : scene_collision_geometries) {
    if (cg.group == CollisionGroup::ROBOT) {
      inspector_.robot_geometry_ids.push_back(id);
    }
    ++id;
  }

  auto links_search = kinematic_tree_.getLinks();
  auto joints_search = kinematic_tree_.getJoints();
  Eigen::MatrixXi adj_kin_tree =
      createAdjacencyMatrix(joints_search, links_search);

  if (scene_collision_geometries.size()) {
    // Compute collision pairs for robot to world. For now they are unordered.
    for (std::size_t geom_A = 0; geom_A < scene_collision_geometries.size() - 1;
         ++geom_A) {
      for (std::size_t geom_B = geom_A + 1;
           geom_B < scene_collision_geometries.size(); ++geom_B) {
        CollisionGeometry coA = scene_collision_geometries.at(geom_A);
        CollisionGeometry coB = scene_collision_geometries.at(geom_B);

        // check if both are world geoms
        auto gr_A = scene_collision_geometries.at(geom_A).group;
        auto gr_B = scene_collision_geometries.at(geom_B).group;

        if (gr_A != CollisionGroup::WORLD || gr_B != CollisionGroup::WORLD) {
          // Either A or B is a robot geometry and (A,B) is potentially a
          // valid pair. If A and B are not part of the same link or part of
          // successive links the pair is valid.

          int parent_link_idx_A =
              kinematic_tree_
                  .getJoint(
                      kinematic_tree_.getLink(coA.link_index).parent_joint)
                  .parent_link;
          int parent_link_idx_B =
              kinematic_tree_
                  .getJoint(
                      kinematic_tree_.getLink(coB.link_index).parent_joint)
                  .parent_link;

          bool not_successive =
              parent_link_idx_A !=
                  scene_collision_geometries.at(geom_B).link_index &&
              parent_link_idx_B !=
                  scene_collision_geometries.at(geom_A).link_index;

          // traverse from geomA to geomB, if they are connected by a chain of
          // fixed links then filter the collisions as well.

          std::vector<bool> visited(kinematic_tree_.numLinks(), false);

          // std::cout<< adj_kin_tree<<std::endl;
          std::vector<int> link_idx_path =
              findPath(adj_kin_tree, coA.link_index, coB.link_index);

          bool connected_by_fixed_joints = true;
          if (link_idx_path.size()) {
            for (size_t s = 0; s < link_idx_path.size() - 1; ++s) {
              int adj_val =
                  adj_kin_tree(link_idx_path.at(s), link_idx_path.at(s + 1));
              JointType type = joints_search.at(adj_val - 1).type;
              if (type != JointType::FIXED) {
                connected_by_fixed_joints = false;
                break;
              }
            }
          }

          bool not_same = coA.link_index != coB.link_index;

          if (not_successive && not_same && !connected_by_fixed_joints) {
            // push back valid collision pair
            std::pair<int, int> col_pair(geom_A, geom_B);
            inspector_.robot_to_world_collision_pairs.push_back(col_pair);
          }
        }
      }
    }
  }
  inspector_.scene_collision_geometry_name_to_geometry_index =
      scene_collision_geometry_name_to_id;
}

bool URDFParser::parseDirectives(const std::string &directives_file) {
  YAML::Node config = YAML::LoadFile(directives_file);

  if (!config["directives"]) {
    std::cerr << "Error: No 'directives' found in the YAML file." << std::endl;
    return false;
  }

  std::stringstream combined_urdf;
  combined_urdf << "<robot name=\"combined_robot\">\n";

  for (const auto &directive : config["directives"]) {
    if (directive["add_model"]) {
      auto model = directive["add_model"];
      std::string name = model["name"].as<std::string>();
      std::string file = model["file"].as<std::string>();

      // Parse the individual URDF file
      tinyxml2::XMLDocument doc;
      if (doc.LoadFile(resolvePackagePath(file).c_str()) !=
          tinyxml2::XML_SUCCESS) {
        std::cout << "[CSDECOMP:ParseDirectives] Error adsfasdfasdf loading "
                     "URDF file: "
                  << file << " that maps to: " << resolvePackagePath(file)
                  << std::endl;
        std::cerr << "[CSDECOMP:ParseDirectives] Error loading URDF file: "
                  << file << " that maps to: " << resolvePackagePath(file)
                  << std::endl;
        return false;
      }

      tinyxml2::XMLPrinter printer;
      // doc.Print(&printer);
      // std::cout << "pre-rename\n\n" << printer.CStr() << std::endl;

      // Rename links and joints
      renameElements(doc.RootElement(), name);
      doc.Print(&printer);
      // std::cout << "post-rename\n\n" << printer.CStr() << std::endl;
      // Add the renamed content to the combined URDF
      for (const tinyxml2::XMLElement *element =
               doc.RootElement()->FirstChildElement();
           element != nullptr; element = element->NextSiblingElement()) {
        printer.ClearBuffer();
        element->Accept(&printer);
        combined_urdf << printer.CStr() << "\n";
      }
    } else if (directive["add_weld"]) {
      auto weld = directive["add_weld"];
      std::string parent = weld["parent"].as<std::string>();
      std::string child = weld["child"].as<std::string>();

      combined_urdf << "  <joint name=\"weld_" << parent << "_to_" << child
                    << "\" type=\"fixed\">\n";
      combined_urdf << "    <parent link=\"" << parent << "\"/>\n";
      combined_urdf << "    <child link=\"" << child << "\"/>\n";
      combined_urdf << "  </joint>\n";
    } else if (directive["add_frame"]) {
      auto frame = directive["add_frame"];
      std::string name = frame["name"].as<std::string>();
      auto X_PF = frame["X_PF"];
      std::string base_frame = X_PF["base_frame"].as<std::string>();
      auto translation = X_PF["translation"].as<std::vector<double>>();
      auto rotation = X_PF["rotation"]["deg"].as<std::vector<double>>();

      combined_urdf << "  <link name=\"" << name << "\"/>\n";
      combined_urdf << "  <joint name=\"" << base_frame << "_to_" << name
                    << "\" type=\"fixed\">\n";
      combined_urdf << "    <parent link=\"" << base_frame << "\"/>\n";
      combined_urdf << "    <child link=\"" << name << "\"/>\n";
      combined_urdf << "    <origin xyz=\"" << translation[0] << " "
                    << translation[1] << " " << translation[2] << "\" ";
      combined_urdf << "rpy=\"" << rotation[0] * M_PI / 180.0 << " "
                    << rotation[1] * M_PI / 180.0 << " "
                    << rotation[2] * M_PI / 180.0 << "\"/>\n";
      combined_urdf << "  </joint>\n";
    }
  }

  combined_urdf << "</robot>";

  // Parse the combined URDF
  bool result = parseURDFString(combined_urdf.str());

  return result;
}

// Helper function to resolve package paths
std::string URDFParser::resolvePackagePath(const std::string &path) {
  if (path.substr(0, 10) == "package://") {
    size_t pos = path.find("/", 10);
    std::string package_name = path.substr(10, pos - 10);
    std::string relative_path = path.substr(pos + 1);
    auto it = package_name_to_path_map.find(package_name);
    if (it != package_name_to_path_map.end()) {
      return it->second + "/" + relative_path;
    }
  }
  return path;
}

// Helper function to rename elements in the URDF
void URDFParser::renameElements(tinyxml2::XMLElement *element,
                                const std::string &prefix) {
  if (element == nullptr) return;

  // Rename the element itself if it has a name attribute
  const char *name = element->Attribute("name");
  if (name != nullptr) {
    std::string new_name = prefix + "::" + name;
    element->SetAttribute("name", new_name.c_str());
  }

  // If this is a joint element, update the parent and child link names
  if (std::string(element->Name()) == "joint") {
    tinyxml2::XMLElement *parent = element->FirstChildElement("parent");
    tinyxml2::XMLElement *child = element->FirstChildElement("child");

    if (parent && parent->Attribute("link")) {
      std::string parent_link = parent->Attribute("link");
      if (parent_link == "world") {
        std::string tmp = fmt::format(
            "[CSDECOMP:URDFParser] Directly specifying the world link as "
            "parent "
            "in "
            "one of the URDFS (found in {}) is not supported due to "
            "potential "
            "ambiguity. To "
            "weld the robot to the world specify this in a directive using "
            "the "
            "-add_weld directive.",
            name);
        throw std::runtime_error(tmp);
      }
      parent->SetAttribute("link", (prefix + "::" + parent_link).c_str());
    }

    if (child && child->Attribute("link")) {
      std::string child_link = child->Attribute("link");
      child->SetAttribute("link", (prefix + "::" + child_link).c_str());
    }
  }

  // Recursively process child elements
  for (tinyxml2::XMLElement *child = element->FirstChildElement();
       child != nullptr; child = child->NextSiblingElement()) {
    renameElements(child, prefix);
  }
}

bool URDFParser::registerPackage(const std::string &package_name,
                                 const std::string &package_path) {
  // Check if the package name is empty
  if (package_name.empty()) {
    std::cerr << "Error: Package name cannot be empty." << std::endl;
    return false;
  }

  // Check if the package path is empty
  if (package_path.empty()) {
    std::cerr << "Error: Package path cannot be empty." << std::endl;
    return false;
  }

  // Check if the package path exists (by attempting to open a file handle)
  std::ifstream path_check(package_path);
  if (!path_check.good()) {
    std::cerr << "Error: Package path does not exist or is not accessible: "
              << package_path << std::endl;
    return false;
  }
  path_check.close();

  // Check if the package name is already registered
  if (package_name_to_path_map.find(package_name) !=
      package_name_to_path_map.end()) {
    std::cerr << "Warning: Package '" << package_name
              << "' is already registered. Overwriting." << std::endl;
  }

  // Register the package
  package_name_to_path_map[package_name] = package_path;

  std::cout << "Successfully registered package '" << package_name
            << "' with path: " << package_path << std::endl;
  return true;
}
}  // namespace csdecomp