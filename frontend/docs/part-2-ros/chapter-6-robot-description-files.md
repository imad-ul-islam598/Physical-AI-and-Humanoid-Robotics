---
sidebar_position: 6
---

# Chapter 6: Robot Description Files

## Introduction

Robot Description Format (URDF) and its extension Xacro are essential tools for defining robot models in ROS 2. These XML-based formats allow for precise specification of a robot's physical and kinematic properties, including links, joints, inertial properties, and visual appearance. Understanding these formats is crucial for simulating, visualizing, and controlling robotic systems, particularly humanoid robots with complex kinematic structures.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the structure and components of URDF files
- Create URDF models for robotic systems with multiple links and joints
- Use Xacro to simplify complex robot descriptions with macros and parameters
- Define joint properties and constraints for robotic systems
- Create a complete humanoid robot model from scratch

## Key Concepts

- **URDF (Unified Robot Description Format)**: XML-based format for describing robot models
- **Xacro**: XML macro language that extends URDF with macros and parameters
- **Links**: Rigid bodies in the robot model with physical properties
- **Joints**: Connections between links with specific degrees of freedom
- **Inertial Properties**: Mass, center of mass, and moment of inertia
- **Visual and Collision Models**: Geometric representations for visualization and collision detection

## Technical Explanation

URDF (Unified Robot Description Format) is an XML-based format used extensively in ROS for representing robot models. It provides a standardized way to describe a robot's physical structure, including its links (rigid bodies), joints (connections between links), and their properties. URDF files are fundamental to robot simulation, visualization, and kinematic analysis.

A URDF robot model consists of:

1. **Links**: Represent rigid bodies in the robot. Each link has:
   - Visual properties (shape, color, material) for visualization
   - Collision properties (shape) for collision detection
   - Inertial properties (mass, center of mass, inertia matrix) for dynamics simulation

2. **Joints**: Define the connection between two links. Each joint has:
   - Type (fixed, continuous, revolute, prismatic, etc.)
   - Limits (position, velocity, effort)
   - Origin (position and orientation relative to parent link)
   - Axis (for rotational or prismatic joints)

The most common joint types are:
- **Fixed**: No movement between links
- **Revolute**: Rotational movement with limits
- **Continuous**: Unlimited rotational movement
- **Prismatic**: Linear sliding movement with limits

URDF alone can become unwieldy for complex robots with many similar components. Xacro (XML Macros) extends URDF by providing:
- Macros for reusing common structures
- Property definitions for parameters
- Mathematical expressions
- Conditional blocks

For humanoid robots, URDF/Xacro models typically include:
- A base/torso link
- Multiple joint chains for arms and legs
- Proper inertial properties for dynamic simulation
- Visual and collision models for each link
- Transmission elements for actuator modeling

## Diagrams written as text descriptions

**Diagram 1: URDF Robot Model Structure**
```
Robot Model (Root Link: base_link)
├── base_link (torso)
│   ├── head_link
│   ├── left_shoulder_link
│   │   ├── left_elbow_link
│   │   │   └── left_wrist_link
│   │   │       └── left_hand_link
│   ├── right_shoulder_link
│   │   ├── right_elbow_link
│   │   │   └── right_wrist_link
│   │   │       └── right_hand_link
│   ├── left_hip_link
│   │   ├── left_knee_link
│   │   │   └── left_ankle_link
│   │   │       └── left_foot_link
│   └── right_hip_link
│       ├── right_knee_link
│       │   └── right_ankle_link
│       │       └── right_foot_link
```

**Diagram 2: Link and Joint Relationship**
```
Parent Link (A)
     │
     │ Joint (Type: Revolute)
     │ Limits: -90° to 90°
     │ Axis: Z-axis
     ▼
Child Link (B)
     │
     │ Joint (Type: Revolute)
     │ Limits: -45° to 45°
     │ Axis: Y-axis
     ▼
Child Link (C)
```

**Diagram 3: Xacro Macro Structure**
```
Xacro File
├── Properties (Parameters)
│   ├── mass_arm: 2.5
│   ├── length_upper_arm: 0.3
│   └── length_lower_arm: 0.25
├── Macros (Reusable Components)
│   ├── macro: simple_arm
│   │   ├── input: name, side
│   │   └── output: arm links and joints
│   └── macro: leg_chain
│       ├── input: name, side
│       └── output: leg links and joints
└── Robot Definition
    ├── Use macro: simple_arm (left)
    ├── Use macro: simple_arm (right)
    ├── Use macro: leg_chain (left)
    ├── Use macro: leg_chain (right)
    └── Base link with sensors
```

## Code Examples

Here's an example of a simple robot URDF:

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Link connected by a revolute joint -->
  <joint name="base_to_arm" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0.25 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <link name="arm_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <origin xyz="0 0 0.2" rpy="1.57 0 0"/>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <origin xyz="0 0 0.2" rpy="1.57 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
</robot>
```

Here's a more complex example using Xacro for a humanoid arm:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_arm">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="arm_mass" value="2.0"/>
  <xacro:property name="upper_arm_length" value="0.3"/>
  <xacro:property name="lower_arm_length" value="0.25"/>

  <!-- Macro for creating an arm -->
  <xacro:macro name="simple_arm" params="side parent *origin">
    <!-- Shoulder link -->
    <link name="${side}_shoulder_link">
      <visual>
        <geometry>
          <sphere radius="0.05"/>
        </geometry>
        <material name="${side}_shoulder_material">
          <color rgba="0.5 0.5 0.5 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <sphere radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_shoulder_link"/>
      <xacro:insert_block name="origin"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="2.0"/>
    </joint>

    <!-- Upper arm link -->
    <link name="${side}_upper_arm_link">
      <visual>
        <geometry>
          <cylinder radius="0.04" length="${upper_arm_length}"/>
        </geometry>
        <origin xyz="0 0 ${upper_arm_length/2}" rpy="1.57 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.04" length="${upper_arm_length}"/>
        </geometry>
        <origin xyz="0 0 ${upper_arm_length/2}" rpy="1.57 0 0"/>
      </collision>
      <inertial>
        <mass value="${arm_mass * 0.4}"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_shoulder_link"/>
      <child link="${side}_upper_arm_link"/>
      <origin xyz="0 0 ${upper_arm_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-M_PI/2}" upper="${M_PI}" effort="100" velocity="2.0"/>
    </joint>

    <!-- Lower arm link -->
    <link name="${side}_lower_arm_link">
      <visual>
        <geometry>
          <cylinder radius="0.03" length="${lower_arm_length}"/>
        </geometry>
        <origin xyz="0 0 ${lower_arm_length/2}" rpy="1.57 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.03" length="${lower_arm_length}"/>
        </geometry>
        <origin xyz="0 0 ${lower_arm_length/2}" rpy="1.57 0 0"/>
      </collision>
      <inertial>
        <mass value="${arm_mass * 0.3}"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_wrist_joint" type="revolute">
      <parent link="${side}_upper_arm_link"/>
      <child link="${side}_lower_arm_link"/>
      <origin xyz="0 0 ${lower_arm_length}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="50" velocity="3.0"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro to create both arms -->
  <xacro:simple_arm side="left" parent="base_link">
    <origin xyz="0.1 0.2 0" rpy="0 0 0"/>
  </xacro:simple_arm>

  <xacro:simple_arm side="right" parent="base_link">
    <origin xyz="0.1 -0.2 0" rpy="0 0 0"/>
  </xacro:simple_arm>
</robot>
```

## Exercises

1. **URDF Creation**: Create a URDF file for a simple 2-wheeled robot with a caster wheel, including proper visual, collision, and inertial properties.

2. **Xacro Macro Design**: Create a Xacro macro for a humanoid leg with hip, knee, and ankle joints, then instantiate both left and right legs.

3. **Joint Limit Analysis**: For a humanoid arm with shoulder, elbow, and wrist joints, determine appropriate joint limits based on human anatomy.

4. **Inertial Property Estimation**: Calculate approximate inertial properties for a humanoid torso assuming it's a rectangular box with dimensions 0.4m x 0.3m x 0.6m and mass of 20kg.

## Summary

URDF and Xacro provide the essential framework for describing robot models in ROS 2. Understanding these formats is crucial for developing, simulating, and controlling robotic systems. URDF defines the kinematic and physical properties of robots, while Xacro extends its capabilities with macros and parameters for managing complex models efficiently. For humanoid robots with their intricate structures, these tools enable precise modeling of the mechanical system, which is fundamental to successful control and simulation. As we continue our exploration of Physical AI, these robot description files will serve as the foundation for simulation, visualization, and control system development.