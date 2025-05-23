import pycsdecomp as csd
import numpy as np

urdf_string = f"""
<?xml version="1.0"?>
<robot name="two_link_robot">

  <!-- Link 1 -->
  <link name="link1">
    <visual>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 0.1 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <visual>
      <origin xyz="0.5 0 0" rpy="1 0 0"/>
      <geometry>
        <box size="1 0.1 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 0.1 0.1"/>
      </geometry>
    </collision>
    <collision>
      <origin xyz="0.5 0 0" rpy="1 0 0"/>
      <geometry>
        <box size="1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint 1 -->
  <joint name="joint1" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="world"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
  </joint>

  <!-- Link 2 -->
  <link name="link2">
    <visual>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 0.1 0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint 2 -->
  <joint name="joint2" type="revolute">
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
  </joint>

  <!-- Link 3 -->
  <link name="link3">
    <visual>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 0.1 0.1"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint 2 -->
  <joint name="joint3" type="revolute">
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <parent link="link2"/>
    <child link="link3"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
  </joint>

<!-- Link 1 -->
<link name="link4">
  <visual>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 0.1 0.1"/>
    </geometry>
    <material name="purp">
      <color rgba="1 0 1 1"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 0.1 0.1"/>
    </geometry>
  </collision>
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>

<!-- Joint 1 -->
<joint name="joint4" type="revolute">
  <origin xyz="0 -1 0" rpy="0 0 0"/>
  <parent link="link1"/>
  <child link="link4"/>
  <axis xyz="0 0 1"/>
  <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
</joint>

<link name="link5">
  <visual>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 0.1 0.1"/>
    </geometry>
    <material name="af">
      <color rgba="0 1 1 1"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 0.1 0.1"/>
    </geometry>
  </collision>
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>

<joint name="joint5" type="revolute">
  <origin xyz="1 0 0" rpy="0 0 0"/>
  <parent link="link4"/>
  <child link="link5"/>
  <axis xyz="0 0 1"/>
  <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
</joint>


<link name="link6">
  <visual>
    <origin xyz="0.0 0 0" rpy="0 0 0"/>
    <geometry>
      <sphere radius="0.5"/>
    </geometry>
    <material name="somth">
      <color rgba="1 1 1 1"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0.0 0 0" rpy="0 0 0"/>
    <geometry>
      <sphere radius="0.5"/>
    </geometry>
  </collision>
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>

<joint name="joint6" type="fixed">
  <origin xyz="1.5 0 0" rpy="0 0 0"/>
  <parent link="link5"/>
  <child link="link6"/>
</joint>

<!-- box 1 -->
<link name="box1">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <material name="black">
      <color rgba="0 0 0 1"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
  <inertial>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>

<!-- box 2 -->
<link name="box2">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <material name="black">
      <color rgba="0 0 0 1"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
  <inertial>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>

 <!-- Joint w box1 -->
 <joint name="jointwb1" type="fixed">
  <origin xyz="0 0 1" rpy="0 0 0"/>
  <parent link="world"/>
  <child link="box1"/>
 </joint>

 <!-- Joint box1 box2 -->
 <joint name="jointb1b2" type="fixed">
  <origin xyz="1 0 0" rpy="0 0 0"/>
  <parent link="box1"/>
  <child link="box2"/>
 </joint>

</robot>

"""

parser = csd.URDFParser()
parser.parse_urdf_string(urdf_string)
plant = parser.build_plant()

configs = np.array([[0.6714, -0.9096, 0.79, -0.1806, 0.],
[0.6714, -0.9096, 0.78, -0.1806, 0.],
[0.0204, 0.3714, 0.4714, 1.0344, -1.2816],
[0.0204, 0.3714, 0.4714, 1.0874, -1.2816],]).T

expected_results = [True, False, True, False]

results =csd.CheckCollisionFreeCuda(configs, plant.getMinimalPlant())

for r, e in zip(results, expected_results):
    assert r==e

print("Success")