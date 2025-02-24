// csdecomp/tests/urdf_parser_test.cpp
#include "urdf_parser.h"

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "kinematic_tree.h"
using namespace csdecomp;

bool areMinimalLinksEqual(const MinimalLink &link1, const MinimalLink &link2) {
  // Compare basic members
  if (link1.index != link2.index ||
      link1.parent_joint_index != link2.parent_joint_index ||
      link1.num_child_joints != link2.num_child_joints) {
    return false;
  }

  // Compare only the initialized part of child_joint_index_array
  return memcmp(link1.child_joint_index_array, link2.child_joint_index_array,
                sizeof(int) * link1.num_child_joints) == 0;
}

bool areMinimalJointsEqual(const MinimalJoint &j1, const MinimalJoint &j2) {
  if (j1.type != j2.type) {
    std::cout << "Joint type mismatch\n";
    return false;
  }

  if (!j1.X_PL_J.isApprox(j2.X_PL_J)) {
    std::cout << "Transform mismatch\n";
    return false;
  }

  if (j1.index != j2.index) {
    std::cout << "Index mismatch\n";
    return false;
  }

  if (j1.parent_link != j2.parent_link) {
    std::cout << "Parent link mismatch\n";
    return false;
  }

  if (j1.child_link != j2.child_link) {
    std::cout << "Child link mismatch\n";
    return false;
  }

  if (!j1.axis.isApprox(j2.axis)) {
    std::cout << "Axis mismatch\n";
    return false;
  }

  return true;
}

bool AreMinimalKinematicTreesEqual(const MinimalKinematicTree &tree1,
                                   const MinimalKinematicTree &tree2) {
  if (tree1.num_configuration_variables != tree2.num_configuration_variables)
    return false;
  if (tree1.num_joints != tree2.num_joints) return false;
  if (tree1.num_links != tree2.num_links) return false;
  for (int i = 0; i < tree1.num_joints; ++i) {
    if (!areMinimalJointsEqual(tree1.joints[i], tree2.joints[i])) return false;
    if (tree1.joint_traversal_sequence[i] != tree2.joint_traversal_sequence[i])
      return false;
    if (tree1.joint_idx_to_config_idx[i] != tree2.joint_idx_to_config_idx[i])
      return false;
  }
  std::cout << "checking links  "
            << " same\n";
  for (int i = 0; i < tree1.num_links; ++i) {
    if (!areMinimalLinksEqual(tree1.links[i], tree2.links[i])) return false;
  }

  return true;
}

class URDFParserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    parser.registerPackage("test_assets", prefix);
    kTol = 1e-6;
  }

  void TearDown() override {
    // Remove the temporary file
    std::string tmp = prefix + "temp_test.urdf";
    std::remove(tmp.c_str());
  }

  std::string urdf_content;
  std::string prefix = "csdecomp/tests/test_assets/";
  URDFParser parser;
  float kTol;
};

TEST_F(URDFParserTest, ParseValidURDF) {
  // Create a simple URDF string for testing
  urdf_content = R"(
<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
  </link>

  <link name="right_leg">
    <visual>
      <geometry>
        <box size="0.6 0.1 0.2"/>
      </geometry>
    </visual>
  </link>
  <joint name="world_to_base" type="fixed">
    <parent link="world"/>
    <child link="base_link"/>
    <origin xyz="0 0.0 0.25"/>
  </joint>

  <joint name="base_to_right_leg" type="revolute">
    <parent link="base_link"/>
    <child link="right_leg"/>
    <origin xyz="0 -0.22 0.25"/>
    <axis xyz="0 0 1"/>
  </joint>

  
</robot>
        )";

  // Write the URDF content to a temporary file
  std::string tmp = prefix + "temp_test.urdf";
  std::ofstream temp_file(tmp);
  temp_file << urdf_content;
  temp_file.close();
  ASSERT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();
  // CollisionGeometry collision = parser.getCollisionGeometry();

  // Check if the correct number of links and joints were parsed
  EXPECT_EQ(tree.getLinks().size(), 3);
  EXPECT_EQ(tree.getJoints().size(), 2);

  // Check if the joint was parsed correctly
  const auto &joints = tree.getJoints();
  EXPECT_EQ(joints[0].name, "world_to_base");
  EXPECT_EQ(joints[0].type, JointType::FIXED);
  EXPECT_EQ(joints[0].parent_link, 0);  // Assuming base_link is the first link
  EXPECT_EQ(joints[0].child_link, 1);   // Assuming right_leg is the second link
  EXPECT_EQ(joints[1].name, "base_to_right_leg");
  EXPECT_EQ(joints[1].type, JointType::REVOLUTE);
  EXPECT_EQ(joints[1].parent_link, 1);  // Assuming base_link is the first link
  EXPECT_EQ(joints[1].child_link, 2);   // Assuming right_leg is the second link
  // You can add more specific checks here based on your implementation
}

TEST_F(URDFParserTest, ParseInvalidURDF) {
  // Try to parse a non-existent file
  EXPECT_THROW(parser.parseURDF("non_existent_file.urdf"), std::runtime_error);
}
TEST_F(URDFParserTest, AddLinkAfterFinalized) {
  std::string tmp = prefix + "2_branch_5dof.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();
  EXPECT_TRUE(tree.isFinalized());
  Joint joint;
  Link link;
  EXPECT_THROW(tree.addJoint(joint), std::runtime_error);
  EXPECT_THROW(tree.addLink(link), std::runtime_error);
}
TEST_F(URDFParserTest, LeafLinkIndices) {
  std::string tmp = prefix + "2_branch_5dof.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();
  EXPECT_TRUE(tree.isFinalized());
  auto indices = tree.getLeafLinkIndices();
  for (auto i : indices) {
    EXPECT_TRUE(i == 3 || i == 6 || i == 8);
  }
}
TEST_F(URDFParserTest, ParseTwoBranchURDF) {
  std::string tmp = prefix + "2_branch_5dof.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();

  const auto &joints = tree.getJoints();

  EXPECT_EQ(tree.getLinks().size(), 9);
  EXPECT_EQ(tree.getJoints().size(), 8);

  EXPECT_EQ(joints[0].name, "joint1");
  EXPECT_EQ(joints[0].type, JointType::REVOLUTE);
  EXPECT_EQ(joints[0].parent_link, 0);
  EXPECT_EQ(joints[0].child_link, 1);

  EXPECT_EQ(joints[1].name, "joint2");
  EXPECT_EQ(joints[1].type, JointType::REVOLUTE);
  EXPECT_EQ(joints[1].parent_link, 1);
  EXPECT_EQ(joints[1].child_link, 2);

  EXPECT_EQ(joints[2].name, "joint3");
  EXPECT_EQ(joints[2].type, JointType::REVOLUTE);
  EXPECT_EQ(joints[2].parent_link, 2);
  EXPECT_EQ(joints[2].child_link, 3);

  EXPECT_EQ(joints[3].name, "joint4");
  EXPECT_EQ(joints[3].type, JointType::REVOLUTE);
  EXPECT_EQ(joints[3].parent_link, 1);
  EXPECT_EQ(joints[3].child_link, 4);

  EXPECT_EQ(joints[4].name, "joint5");
  EXPECT_EQ(joints[4].type, JointType::REVOLUTE);
  EXPECT_EQ(joints[4].parent_link, 4);
  EXPECT_EQ(joints[4].child_link, 5);

  EXPECT_EQ(joints[5].name, "joint6");
  EXPECT_EQ(joints[5].type, JointType::FIXED);
  EXPECT_EQ(joints[5].parent_link, 5);
  EXPECT_EQ(joints[5].child_link, 6);

  for (auto l : tree.getPositionLowerLimits()) {
    EXPECT_NEAR(l, -3.14, kTol);
  }

  for (auto l : tree.getPositionUpperLimits()) {
    EXPECT_NEAR(l, 3.14, kTol);
  }
}

TEST_F(URDFParserTest, ParseTwoBranchURDFCollisionGeomsAndPairs) {
  std::string tmp = prefix + "2_branch_5dof.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();
  const std::vector<CollisionGeometry> scene_col_geoms =
      parser.getSceneCollisionGeometries();

  SceneInspector inspector = parser.getSceneInspector();
  int index;
  index = tree.getLinkIndexByName("link1");
  auto geom_ids =
      inspector.link_index_to_scene_collision_geometry_indices.at(index);
  for (auto id : geom_ids) {
    CollisionGeometry geom_of_interest = scene_col_geoms.at(id);
    EXPECT_EQ(geom_of_interest.type, ShapePrimitive::BOX);
    EXPECT_EQ(geom_of_interest.link_index, index);
    if (id == 1) {
      Eigen::AngleAxisf rollAngle(1, Eigen::Vector3f::UnitX());
      Eigen::AngleAxisf pitchAngle(0, Eigen::Vector3f::UnitY());
      Eigen::AngleAxisf yawAngle(0, Eigen::Vector3f::UnitZ());

      Eigen::Matrix3f rotationMatrix =
          (yawAngle * pitchAngle * rollAngle).toRotationMatrix();

      Eigen::Matrix4f X_L_B = Eigen::Matrix4f::Identity();
      X_L_B.block<3, 3>(0, 0) = rotationMatrix;
      X_L_B(0, 3) = 0.5;

      EXPECT_TRUE(geom_of_interest.X_L_B.isApprox(X_L_B));
    }
  }
  index = tree.getLinkIndexByName("link2");
  geom_ids = inspector.link_index_to_scene_collision_geometry_indices.at(index);
  for (auto id : geom_ids) {
    CollisionGeometry geom_of_interest = scene_col_geoms.at(id);
    EXPECT_EQ(geom_of_interest.type, ShapePrimitive::BOX);
    EXPECT_EQ(geom_of_interest.link_index, index);
  }
  index = tree.getLinkIndexByName("link3");
  geom_ids = inspector.link_index_to_scene_collision_geometry_indices.at(index);
  for (auto id : geom_ids) {
    CollisionGeometry geom_of_interest = scene_col_geoms.at(id);
    EXPECT_EQ(geom_of_interest.type, ShapePrimitive::BOX);
    EXPECT_EQ(geom_of_interest.link_index, index);
  }
  index = tree.getLinkIndexByName("link4");
  geom_ids = inspector.link_index_to_scene_collision_geometry_indices.at(index);
  for (auto id : geom_ids) {
    CollisionGeometry geom_of_interest = scene_col_geoms.at(id);
    EXPECT_EQ(geom_of_interest.type, ShapePrimitive::BOX);
    EXPECT_EQ(geom_of_interest.link_index, index);
  }
  index = tree.getLinkIndexByName("link5");
  geom_ids = inspector.link_index_to_scene_collision_geometry_indices.at(index);
  for (auto id : geom_ids) {
    CollisionGeometry geom_of_interest = scene_col_geoms.at(id);
    EXPECT_EQ(geom_of_interest.type, ShapePrimitive::BOX);
    EXPECT_EQ(geom_of_interest.link_index, index);
  }
  index = tree.getLinkIndexByName("link6");
  geom_ids = inspector.link_index_to_scene_collision_geometry_indices.at(index);
  for (auto id : geom_ids) {
    CollisionGeometry geom_of_interest = scene_col_geoms.at(id);
    EXPECT_EQ(geom_of_interest.type, ShapePrimitive::SPHERE);
    EXPECT_EQ(geom_of_interest.link_index, index);
  }
  index = tree.getLinkIndexByName("box1");
  geom_ids = inspector.link_index_to_scene_collision_geometry_indices.at(index);
  for (auto id : geom_ids) {
    CollisionGeometry geom_of_interest = scene_col_geoms.at(id);
    EXPECT_EQ(geom_of_interest.type, ShapePrimitive::BOX);
    EXPECT_EQ(geom_of_interest.link_index, index);
  }
  index = tree.getLinkIndexByName("box2");
  geom_ids = inspector.link_index_to_scene_collision_geometry_indices.at(index);
  for (auto id : geom_ids) {
    CollisionGeometry geom_of_interest = scene_col_geoms.at(id);
    EXPECT_EQ(geom_of_interest.type, ShapePrimitive::BOX);
    EXPECT_EQ(geom_of_interest.link_index, index);
  }

  // expected collision pairs
  //  Vector of expected pairs
  std::vector<std::pair<int, int>> expected_pairs = {
      {0, 3}, {0, 5}, {0, 6}, {0, 7}, {0, 8}, {1, 3}, {1, 5}, {1, 6}, {1, 7},
      {1, 8}, {2, 4}, {2, 5}, {2, 6}, {2, 7}, {2, 8}, {3, 4}, {3, 5}, {3, 6},
      {3, 7}, {3, 8}, {4, 6}, {4, 7}, {4, 8}, {5, 7}, {5, 8}, {6, 7}, {6, 8}};

  // Check if all expected pairs are in robot_to_world_collision_pairs
  for (const auto &pair : expected_pairs) {
    EXPECT_NE(std::find(inspector.robot_to_world_collision_pairs.begin(),
                        inspector.robot_to_world_collision_pairs.end(), pair),
              inspector.robot_to_world_collision_pairs.end())
        << "Pair {" << pair.first << "," << pair.second << "} not found";
  }
}

TEST_F(URDFParserTest, TestMinimalPlant) {
  std::string tmp = prefix + "2_branch_5dof.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();
  const std::vector<CollisionGeometry> scene_col_geoms =
      parser.getSceneCollisionGeometries();

  SceneInspector inspector = parser.getSceneInspector();
  CollisionPairMatrix cpm = inspector.getCollisionPairMatrix();
  MinimalPlant plant = parser.getMinimalPlant();
  MinimalKinematicTree mtree = tree.getMyMinimalKinematicTreeStruct();

  EXPECT_EQ(scene_col_geoms.size(), plant.num_scene_geometries);
  const CollisionPairMatrix cpm_plant = Eigen::Map<const CollisionPairMatrix>(
      plant.collision_pairs_flat, 2, (int)plant.num_collision_pairs);
  EXPECT_TRUE(cpm_plant.isApprox(cpm));
  int idx = 0;
  for (auto co : scene_col_geoms) {
    const CollisionGeometry co_plant = plant.scene_collision_geometries[idx];
    EXPECT_EQ(co.type, co_plant.type);
    EXPECT_EQ(co.link_index, co_plant.link_index);
    EXPECT_TRUE(co.X_L_B.isApprox(co_plant.X_L_B));
    EXPECT_EQ(co.dimensions[0], co_plant.dimensions[0]);
    EXPECT_EQ(co.dimensions[1], co_plant.dimensions[1]);
    EXPECT_EQ(co.dimensions[2], co_plant.dimensions[2]);
    ++idx;
  }

  EXPECT_TRUE(AreMinimalKinematicTreesEqual(mtree, plant.kin_tree));
}

TEST_F(URDFParserTest, Parse3DofPrismatic) {
  std::string tmp = prefix + "3dofrobot_prismatic.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree = parser.getKinematicTree();
  const auto &joints = tree.getJoints();

  EXPECT_EQ(tree.getLinks().size(), 4);
  EXPECT_EQ(tree.getJoints().size(), 3);

  EXPECT_EQ(joints[0].name, "joint1");
  EXPECT_EQ(joints[0].type, JointType::REVOLUTE);
  EXPECT_EQ(joints[0].parent_link, 0);
  EXPECT_EQ(joints[0].child_link, 1);

  EXPECT_EQ(joints[1].name, "joint2");
  EXPECT_EQ(joints[1].type, JointType::REVOLUTE);
  EXPECT_EQ(joints[1].parent_link, 1);
  EXPECT_EQ(joints[1].child_link, 2);

  EXPECT_EQ(joints[2].name, "joint3");
  EXPECT_EQ(joints[2].type, JointType::PRISMATIC);
  EXPECT_EQ(joints[2].parent_link, 2);
  EXPECT_EQ(joints[2].child_link, 3);
}
// Make sure parser fails if there are multiple links with the same name
TEST_F(URDFParserTest, ReturnFalseOnDuplicateLink) {
  std::string tmp = prefix + "3dof_doublename_link.urdf";
  EXPECT_THROW(parser.parseURDF(tmp), std::runtime_error);
}
// Make sure parser fails if there are multiple joints with the same name
TEST_F(URDFParserTest, ReturnFalseOnDuplicateJoint) {
  std::string tmp = prefix + "3dof_doublename_joint.urdf";
  EXPECT_THROW(parser.parseURDF(tmp), std::runtime_error);
}
// Test parsing of joint origins
TEST_F(URDFParserTest, JointAxis) {
  std::string tmp = prefix + "3dofrobot_prismatic.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));

  KinematicTree tree;
  Eigen::Vector3f axis_expected;

  tree = parser.getKinematicTree();
  const std::vector<Joint> joints = tree.getJoints();
  auto world_to_link1 = joints.at(tree.getJointIndex("joint1"));

  Eigen::Vector3f axis = world_to_link1.axis;
  axis_expected.x() = 0;
  axis_expected.y() = 1;
  axis_expected.z() = 0;
  EXPECT_EQ(axis, axis_expected);

  auto l2_to_l3 = joints.at(tree.getJointIndex("joint3"));
  EXPECT_TRUE(world_to_link1.type = JointType::REVOLUTE);
  EXPECT_TRUE(l2_to_l3.type = JointType::PRISMATIC);
}

// Test parsing of joint origins
TEST_F(URDFParserTest, ParseKinova) {
  std::string tmp = prefix + "kinova.urdf";
  EXPECT_THROW(parser.parseURDF(tmp), std::runtime_error);
}

// checkst that the parent joints are valid and initialized for all but the
// world link
TEST_F(URDFParserTest, TestSimpleDirectives) {
  std::string tmp = prefix + "directives/simple.yml";
  EXPECT_TRUE(parser.parseDirectives(tmp));

  auto kt = parser.getKinematicTree();
  auto joints = kt.getJoints();
  auto links = kt.getLinks();
  EXPECT_EQ(kt.getLinks().size(), 18);
  EXPECT_EQ(kt.getJoints().size(), 17);
  auto inspector = parser.getSceneInspector();
  auto scene_geometries = parser.getSceneCollisionGeometries();
  EXPECT_EQ(scene_geometries.size(), 10);
  int table_origin_idx = kt.getLinkIndexByName("table_origin");
  const Eigen::Matrix4f tf_world_table =
      joints.at(links.at(table_origin_idx).parent_joint).X_PL_J;
  Eigen::Matrix4f X_W_table_expected;
  //clang-format off
  X_W_table_expected << 1, 0, 0, 0.4, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  //clang-format on
  EXPECT_TRUE(X_W_table_expected.isApprox(tf_world_table));
}

// checks that parser throws if one of the branches is not attached to the root
TEST_F(URDFParserTest, ThrowOnDirectivesWithFloatingModel) {
  std::string tmp = prefix + "directives/simple_no_anchor.yml";
  EXPECT_THROW(parser.parseDirectives(tmp), std::runtime_error);
}

// checkst that the parent joints are valid and initialized for all but the
// world link
TEST_F(URDFParserTest, TestLoadKinovaTableAndRobotiq) {
  std::string tmp = prefix + "directives/kinova_sens_on_table.yml";
  // making this throw to remind me to implement the test
  EXPECT_TRUE(parser.parseDirectives(tmp));
  auto kt = parser.getKinematicTree();
  auto joints = kt.getJoints();
  auto links = kt.getLinks();
  auto geoms = parser.getSceneCollisionGeometries();
  EXPECT_EQ(kt.getLinks().size(), 33);
  EXPECT_EQ(kt.getJoints().size(), 32);
  EXPECT_EQ(geoms.size(), 20);
  Eigen::VectorXf lower(7);
  lower << -3.1, -2.24, -3.1, -2.57, -3.1, -2.09, -3.1;

  Eigen::VectorXf upper(7);
  upper << 3.1, 2.24, 3.1, 2.57, 3.1, 2.09, 3.1;

  EXPECT_TRUE(lower.isApprox(kt.getPositionLowerLimits()));
  EXPECT_TRUE(upper.isApprox(kt.getPositionUpperLimits()));
  SceneInspector insp = parser.getSceneInspector();
  CollisionPairMatrix cpm = insp.getCollisionPairMatrix();
  bool pair7_9_present = false;
  for (int col = 0; col < cpm.cols(); ++col) {
    if (cpm(0, col) == 7 && cpm(1, col) == 9) {
      pair7_9_present = true;  // Found the pair (7,9)
    }
  }
  EXPECT_FALSE(pair7_9_present);
}

GTEST_TEST(PlantTest, TestPlant) {
  std::string prefix = "csdecomp/tests/test_assets/";
  URDFParser parser;
  parser.registerPackage("test_assets", prefix);
  std::string tmp = prefix + "directives/kinova_sens_on_table.yml";
  EXPECT_THROW(parser.buildPlant(), std::runtime_error);
  EXPECT_TRUE(parser.parseDirectives(tmp));
  Plant plant = parser.buildPlant();
  SceneInspector insp = parser.getSceneInspector();

  MinimalPlant mplant = plant.getMinimalPlant();
  MinimalKinematicTree parser_mkt = parser.getMinimalKinematicTree();
  MinimalKinematicTree mplant_mkt = mplant.kin_tree;
  EXPECT_TRUE(AreMinimalKinematicTreesEqual(mplant_mkt, parser_mkt));

  std::vector<std::string> col_geom_names =
      plant.getSceneCollisionGeometryNames();
  auto it = std::find(col_geom_names.begin(), col_geom_names.end(),
                      "kinova::half_arm_2_link_collision::1");
  int idx_plant = plant.getSceneCollisionGeometryIndexByName(
      "kinova::half_arm_2_link_collision::1");
  EXPECT_TRUE(it != col_geom_names.end());
  EXPECT_EQ(insp.scene_collision_geometry_name_to_geometry_index
                ["kinova::half_arm_2_link_collision::1"],
            idx_plant);
  std::vector<CollisionGeometry> col_geoms =
      plant.getSceneCollisionGeometries();
  EXPECT_EQ(col_geoms.at(idx_plant).type, ShapePrimitive::BOX);
}

GTEST_TEST(ParserTest, ParseCapsules) {
  std::string prefix = "csdecomp/tests/test_assets/";
  URDFParser parser;
  std::string tmp = prefix + "movable_capsule.urdf";
  EXPECT_TRUE(parser.parseURDF(tmp));
  Plant p = parser.buildPlant();
  int index1 =
      p.getSceneCollisionGeometryIndexByName("movable_capsule::col_geom::1");
  int index2 =
      p.getSceneCollisionGeometryIndexByName("static_capsule::col_geom::1");
  std::vector<CollisionGeometry> col_geoms = p.getSceneCollisionGeometries();
  CollisionGeometry movable = col_geoms.at(index1);
  CollisionGeometry static_geom = col_geoms.at(index2);
  EXPECT_FLOAT_EQ(movable.dimensions[0], 0.1);
  EXPECT_FLOAT_EQ(movable.dimensions[1], 0.3);
  EXPECT_FLOAT_EQ(static_geom.dimensions[0], 0.2);
  EXPECT_FLOAT_EQ(static_geom.dimensions[1], 0.5);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}