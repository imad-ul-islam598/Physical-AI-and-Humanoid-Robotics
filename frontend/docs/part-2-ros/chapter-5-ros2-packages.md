---
sidebar_position: 5
---

# Chapter 5: ROS 2 Packages

## Introduction

ROS 2 packages are the fundamental units of code organization in the Robot Operating System 2 ecosystem. They provide a structured way to organize, build, and distribute robot software components. Understanding package structure, launch files, and parameter management is essential for developing well-organized and maintainable robotic applications. This chapter explores the anatomy of ROS 2 packages and how they facilitate the creation of complex robotic systems.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the structure and organization of ROS 2 packages
- Create and configure ROS 2 packages using colcon build system
- Use launch files to start multiple nodes with specific configurations
- Manage parameters for ROS 2 nodes effectively
- Organize code and resources within packages for optimal maintainability

## Key Concepts

- **Package**: A container for organizing ROS 2 code, data, and resources
- **colcon**: The build system used for ROS 2 packages
- **Launch Files**: XML or Python files that define how to start multiple nodes
- **Parameters**: Configuration values that can be set at runtime
- **ament**: The ROS 2 build system and package management framework
- **Resource Management**: Proper organization of configuration files, models, and other resources

## Technical Explanation

ROS 2 packages serve as the primary organizational unit for ROS 2 software. Each package contains a specific functionality or component of a larger robotic system. A well-structured package includes source code, configuration files, launch files, and documentation, all organized in a standardized directory structure.

The typical structure of a ROS 2 package is:

```
package_name/
├── CMakeLists.txt or package.xml (for C++)
├── setup.py or pyproject.toml (for Python)
├── package.xml (package manifest)
├── src/ (source code)
├── launch/ (launch files)
├── config/ (configuration files)
├── test/ (test files)
├── include/ (header files for C++)
├── scripts/ (executable scripts)
└── README.md (documentation)
```

The `package.xml` file is the package manifest that contains metadata about the package, including its name, version, description, maintainers, license, dependencies, and other information required for building and using the package.

The `CMakeLists.txt` file (for C++ packages) or `setup.py`/`pyproject.toml` (for Python packages) contains build instructions for the package. These files specify how to compile the code, what dependencies to include, and what executables or libraries to create.

Launch files provide a way to start multiple nodes with specific configurations simultaneously. They can be written in XML or Python and allow for complex system initialization with proper parameter setting and node organization. Launch files can include other launch files, set parameters, remap topics, and specify node-specific configurations.

Parameters in ROS 2 allow for runtime configuration of nodes. They can be set through launch files, command-line arguments, or parameter files. Parameters are key-value pairs that control node behavior without requiring code changes. Common parameters include sensor calibration values, control gains, and operational modes.

The colcon build system is responsible for building ROS 2 packages. It provides a unified interface for building packages in different languages and handles dependencies between packages. Colcon can build individual packages or entire workspaces, making it suitable for both development and deployment scenarios.

## Diagrams written as text descriptions

**Diagram 1: ROS 2 Package Structure**
```
robot_package/                    # Package root
├── package.xml                   # Package manifest
├── CMakeLists.txt               # Build instructions (C++)
├── setup.py                     # Build instructions (Python)
├── src/                         # Source code
│   ├── node1.py                 # Python node
│   └── node2.cpp                # C++ node
├── launch/                      # Launch files
│   ├── system.launch.py         # System launch
│   └── sensors.launch.xml       # Sensor launch
├── config/                      # Configuration
│   ├── params.yaml              # Parameters
│   └── calibration.json         # Calibration
├── test/                        # Test files
├── scripts/                     # Executable scripts
└── README.md                    # Documentation
```

**Diagram 2: Package Dependency and Build Flow**
```
Workspace (src/)
├── package_A/    ──┐
│   ├── src/        │
│   └── package.xml │
├── package_B/    ──┤ (depends on A)
│   ├── src/        │
│   └── package.xml │
└── package_C/    ──┘ (depends on B)
    ├── src/
    └── package.xml

Build with: colcon build
```

**Diagram 3: Launch File Structure**
```
Launch File
├── Node A (with parameters)
│   ├── Parameter 1: value1
│   └── Parameter 2: value2
├── Node B (with parameters)
│   ├── Parameter 3: value3
│   └── Parameter 4: value4
├── Remappings: topic1 → topic2
└── Conditions: if variable == true
```

## Code Examples

Here's an example of a simple ROS 2 package structure with a launch file:

Example `package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robot_control</name>
  <version>0.0.1</version>
  <description>Robot control package for Physical AI</description>
  <maintainer email="maintainer@todo.todo">maintainer</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <exec_depend>ros2launch</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Example Python launch file:
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('robot_control')

    # Load parameters from file
    params_file = os.path.join(pkg_share, 'config', 'robot_params.yaml')

    return LaunchDescription([
        Node(
            package='robot_control',
            executable='joint_controller',
            name='joint_controller',
            parameters=[params_file],
            output='screen'
        ),
        Node(
            package='robot_control',
            executable='sensor_processor',
            name='sensor_processor',
            parameters=[
                {'sensor_rate': 50.0},
                {'use_sim_time': False}
            ],
            output='screen'
        ),
        Node(
            package='robot_control',
            executable='motion_planner',
            name='motion_planner',
            parameters=[params_file],
            remappings=[
                ('/cmd_vel', '/robot/cmd_vel'),
                ('/odom', '/robot/odom')
            ],
            output='screen'
        )
    ])
```

Example parameters file (`config/robot_params.yaml`):
```yaml
/**:  # Applies to all nodes
  ros__parameters:
    use_sim_time: false
    control_rate: 100.0
    max_velocity: 1.0
    acceleration_limit: 2.0

joint_controller:
  ros__parameters:
    kp: 10.0
    ki: 0.1
    kd: 0.01
    max_effort: 100.0

sensor_processor:
  ros__parameters:
    sensor_rate: 50.0
    filter_cutoff: 10.0
    enable_filtering: true
```

Example of a node that uses parameters:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class ParameterizedController(Node):
    def __init__(self):
        super().__init__('parameterized_controller')

        # Declare parameters with default values
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('control_rate', 50.0)

        # Get parameter values
        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.control_rate = self.get_parameter('control_rate').value

        # Create publisher
        self.control_pub = self.create_publisher(Float64, 'control_output', 10)

        # Create timer
        self.timer = self.create_timer(1.0/self.control_rate, self.control_callback)

        self.get_logger().info(f'Controller initialized with KP={self.kp}, KI={self.ki}, KD={self.kd}')

    def control_callback(self):
        # Simple control logic
        control_msg = Float64()
        control_msg.data = 0.0  # Placeholder control value
        self.control_pub.publish(control_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = ParameterizedController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercises

1. **Package Creation**: Create a new ROS 2 package named "humanoid_interfaces" that contains message and service definitions for humanoid robot control.

2. **Launch File Design**: Design a launch file that starts a complete humanoid robot system including sensor processing, control, and visualization nodes with appropriate parameters.

3. **Parameter Management**: Create a parameter file that defines different configurations for simulation vs. real robot operation, including joint limits, control gains, and safety parameters.

4. **Dependency Analysis**: Analyze the dependencies of a typical humanoid robot package and explain why each dependency is necessary.

## Summary

ROS 2 packages provide the essential organizational structure for developing complex robotic systems. Understanding package structure, launch files, and parameter management is crucial for creating maintainable and configurable robotic applications. The modular approach of packages enables code reuse, independent development, and systematic testing of robotic components. As we continue exploring Physical AI and humanoid robotics, this package-based organization will serve as the foundation for building sophisticated systems with multiple interacting components.