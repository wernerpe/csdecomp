#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "collision_geometry.h"
#include "kinematic_tree.h"
#include "minimal_plant.h"
#include "tinyxml2.h"

namespace csdecomp {
/**
 * @brief A class for parsing URDF (Unified Robot Description Format) files.
 */
class URDFParser {
 public:
  /**
   * @brief Constructs a URDFParser object.
   */
  URDFParser();

  /**
   * @brief Parses a URDF file.
   * @param urdf_file The path to the URDF file.
   * @return True if parsing was successful, false otherwise.
   *
   * @note If a single urdf file is to be parsed, then make sure to add a joint
   * connecting the robot to the 'world' link which will be created by default
   * internally.
   */
  bool parseURDF(const std::string &urdf_file);

  /**
   * @brief Parses a URDF from a string containing the URDF contents.
   *
   * @param file_contents A string containing the full URDF XML content.
   * @return True if parsing was successful, false otherwise.
   *
   * This method takes a string containing the URDF XML content, parses it,
   * and populates the internal data structures of the URDFParser with the
   * robot's description. This is useful when the URDF content is available
   * in memory or generated dynamically, rather than stored in a file.
   *
   * @note The same considerations apply as with parseURDF() regarding the
   * 'world' link connection.
   */
  bool parseURDFString(const std::string &file_contents);

  /**
   * @brief registers a package. Adds package name and path pair to a map
   * @param package_name Name of the package.
   * @param package_path Path to the location of the package.
   */
  bool registerPackage(const std::string &package_name,
                       const std::string &package_path);
  /**
   * @brief Parses a directives file, and creates a temporary single urdf
   * combining all models.
   * @param directives_file The path to the directives.yaml file.
   * @return True if parsing was successful, false otherwise.
   *
   * @note Make sure that all branches of the kinematic tree connect to the
   * world frame.
   */
  bool parseDirectives(const std::string &urdf_file);

  /**
   * @brief Gets the minimal representation of the plant. This includes both a
   * minimal kinematic tree, the static scene geometries (not voxels) and the
   * collision pair matrix.
   * @return The MinimalPlant object.
   */
  const MinimalPlant getMinimalPlant() const;

  /**
   * @brief Gets the constructed KinematicTree.
   * @return The KinematicTree object representing the parsed robot structure.
   */
  KinematicTree getKinematicTree() const;

  /**
   * @brief Gets the minimal representation of the kinematic tree.
   * @return The MinimalKinematicTree object.
   */
  const MinimalKinematicTree getMinimalKinematicTree() const;

  /**
   * @brief Gets the collision geometries of the scene.
   * @return A vector of CollisionGeometry representing the scene's collision
   * geometries.
   */
  const std::vector<CollisionGeometry> &getSceneCollisionGeometries() const;

  /**
   * @brief Gets the SceneInspector object.
   * @return The SceneInspector object containing scene information.
   */
  const SceneInspector getSceneInspector() const;

  /**
   * @brief overrides SceneCollisionGeometries that have been parsed sofar.
   */
  void overrideSceneCollisionGeometries(
      const std::vector<CollisionGeometry> &col_geoms_update) {
    scene_collision_geometries.clear();
    for (const auto &g : col_geoms_update) {
      scene_collision_geometries.push_back(g);
    }
  };

  /**
   * @brief overrides the kinematic tree in the parser.
   */
  void overrideKinematicTree(const KinematicTree &kinematic_tree) {
    kinematic_tree_ = kinematic_tree;
  };

  const Plant buildPlant() const;

  /**
   * @brief Finalizes the scene setup, constructing the inspector and setting up
   * collision pairs.
   */
  void finalizeScene();

 private:
  KinematicTree kinematic_tree_;
  SceneInspector inspector_;
  std::vector<CollisionGeometry> scene_collision_geometries;
  std::unordered_map<std::string, int> scene_collision_geometry_name_to_id;
  std::unordered_map<std::string, std::string> package_name_to_path_map;

  std::string resolvePackagePath(const std::string &path);
  void renameElements(tinyxml2::XMLElement *element, const std::string &prefix);

  /**
   * @brief Parses a joint element from the URDF.
   * @param joint_element The XMLElement representing the joint.
   * @return True if parsing was successful, false otherwise.
   */
  bool parseJoint(const tinyxml2::XMLElement *joint_element);

  /**
   * @brief Parses a link element from the URDF.
   * @param link_element The XMLElement representing the link.
   * @return True if parsing was successful, false otherwise.
   */
  bool parseLink(const tinyxml2::XMLElement *link_element);

  /**
   * @brief Parses an origin element from the URDF.
   * @param origin_element The XMLElement representing the origin.
   * @return An Eigen::Matrix4f representing the parsed origin transform.
   */
  Eigen::Matrix4f parseOrigin(const tinyxml2::XMLElement *origin_element);

  /**
   * @brief Parses an axis element from the URDF.
   * @param axis_element The XMLElement representing the axis.
   * @return An Eigen::Vector3f representing the parsed axis.
   */
  Eigen::Vector3f parseAxis(const tinyxml2::XMLElement *axis_element);

  /**
   * @brief Parses a collision object from a geometry element in the URDF.
   * @param geometry_element The XMLElement representing the geometry.
   * @return A CollisionGeometry representing the parsed geometry.
   */
  CollisionGeometry parseCollisionGeometry(
      const tinyxml2::XMLElement *geometry_element);
};
}  // namespace csdecomp