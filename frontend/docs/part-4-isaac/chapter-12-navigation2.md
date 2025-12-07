---
sidebar_position: 12
---

# Chapter 12: Navigation 2

## Introduction

Navigation 2 (Nav2) is the next-generation navigation framework for ROS 2, designed to provide robust, flexible, and efficient path planning and navigation capabilities for mobile robots. Built from the ground up for ROS 2, Nav2 offers improved performance, better maintainability, and enhanced features compared to its predecessor. For humanoid robots, Nav2 provides the foundation for autonomous navigation with specialized considerations for bipedal locomotion and complex terrain traversal. This chapter explores the architecture, components, and implementation of Nav2 for Physical AI applications.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the architecture and components of the Navigation 2 framework
- Configure and customize Nav2 for specific robot platforms
- Implement path planning algorithms suitable for humanoid robots
- Handle obstacles and dynamic environments in navigation
- Address the unique challenges of bipedal locomotion planning

## Key Concepts

- **Navigation 2 (Nav2)**: The ROS 2 navigation framework
- **Path Planning**: Algorithms for finding optimal paths from start to goal
- **Costmaps**: Grid-based representations of environment traversability
- **Controllers**: Algorithms that follow planned paths with robot dynamics
- **Biped Locomotion**: Two-legged walking motion for humanoid robots
- **ZMP (Zero Moment Point)**: Stability criterion for bipedal walking
- **Dynamic Obstacle Avoidance**: Real-time obstacle detection and avoidance

## Technical Explanation

Navigation 2 represents a complete re-architecture of the ROS navigation system, taking advantage of ROS 2's improved communication model, lifecycle management, and overall system design. The framework is built around a behavior tree architecture that provides flexible and robust navigation execution.

The core architecture of Nav2 consists of several interconnected components:

1. **Behavior Tree Executor**: The central orchestrator that manages navigation tasks using behavior trees. Behavior trees provide a flexible way to define complex navigation behaviors and handle various scenarios and failure conditions.

2. **Path Planner**: Responsible for generating global paths from the start location to the goal. Nav2 includes multiple planner implementations including the NavFn (Dijkstra-based), Global Planner (A*), and newer approaches like RRT and other sampling-based planners.

3. **Path Controller**: Executes the global plan by generating local commands to follow the path while considering robot dynamics and local obstacles. The controller runs at high frequency to ensure smooth path following.

4. **Costmap 2D**: Provides a grid-based representation of the environment with information about obstacles, free space, and inflation for safety margins. Nav2 uses both global and local costmaps with different update frequencies and purposes.

5. **Recovery Behaviors**: Handles situations where the robot gets stuck or encounters unexpected obstacles. These include spinning in place, backing up, and other recovery actions.

For humanoid robots, Nav2 requires special considerations:

- **Bipedal Kinematics**: Humanoid robots have different kinematic constraints compared to wheeled robots, requiring specialized path following algorithms.
- **Stability Constraints**: Bipedal locomotion requires maintaining balance, which affects how paths can be executed.
- **Footstep Planning**: For humanoid robots, navigation involves planning where to place each foot, not just planning a path for a point robot.
- **Terrain Adaptation**: Humanoid robots can potentially navigate more complex terrain than wheeled robots but require careful foot placement.

The path planning process in Nav2 involves:

1. **Global Path Planning**: Finding a collision-free path from start to goal using the global costmap
2. **Local Path Planning**: Generating safe trajectories in the local costmap while following the global path
3. **Path Execution**: Controlling the robot to follow the planned path while avoiding local obstacles

Costmaps in Nav2 are grid-based representations that store information about the environment:

- **Static Layer**: Fixed obstacles from the map
- **Obstacle Layer**: Dynamic obstacles from sensors
- **Inflation Layer**: Safety margins around obstacles
- **Voxel Layer**: 3D obstacle information (for 3D navigation)

The controller in Nav2 uses various approaches to follow paths:

- **Pure Pursuit**: Follows a look-ahead point on the path
- **DWB (Dynamic Window Approach)**: Considers robot dynamics and constraints
- **MPC (Model Predictive Control)**: Uses predictive models for optimal control

## Diagrams written as text descriptions

**Diagram 1: Navigation 2 Architecture**
```
Navigation 2 System
├── Behavior Tree Executor
│   ├── Global Planner (A*, NavFn, etc.)
│   ├── Local Planner (DWB, etc.)
│   ├── Controller (Pure Pursuit, etc.)
│   └── Recovery Behaviors
├── Global Costmap
│   ├── Static Layer (Map)
│   ├── Inflation Layer
│   └── Obstacle Layer (from sensors)
├── Local Costmap
│   ├── Obstacle Layer (local sensors)
│   ├── Inflation Layer
│   └── Voxel Layer (3D obstacles)
├── Transform System (TF2)
│   ├── Robot Pose
│   ├── Map to Odom
│   └── Sensor Frames
└── Action Interfaces
    ├── Navigate To Pose
    ├── Follow Path
    └── Compute Path
```

**Diagram 2: Path Planning Process**
```
Start Pose → Global Planner → Global Path → Local Planner → Local Trajectory
     │           │              │              │              │
     │           ▼              ▼              ▼              ▼
     │      Plan Path      Update Costmap   Control Robot  Execute Motion
     │           │              │              │              │
     │           ▼              ▼              ▼              ▼
Goal Pose ← Success/ ← Check Collision ← Check Dynamics ← Robot Motion
             Fail
```

**Diagram 3: Costmap Layers**
```
Grid Cell [i,j]
├── Static Cost: Static obstacle map
├── Obstacle Cost: From sensor data
├── Inflation Cost: Safety margin
├── Voxel Cost: 3D obstacle data
└── Total Cost: Combined cost value
     │
     ▼
Navigation Decision (Traversable/Not)
```

## Code Examples

Here's an example of a Nav2 configuration file:

```yaml
# Nav2 Configuration for Humanoid Robot
# File: config/nav2_params.yaml

amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.1
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "/odom"
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Specify the default behavior tree XML file
    default_nav_to_pose_bt_xml: "navigate_w_replanning_and_recovery.xml"
    default_nav_through_poses_bt_xml: "navigate_w_replanning_and_recovery.xml"

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # Controller parameters
    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      # Inner controller
      inner_controller: "FollowPath/DWBLocalPlanner"
      # Rotation shim parameters
      rotation_speed_thresh: 0.3
      min_to_turn_angle: 1.0
      max_to_turn_angle: 10.0

    FollowPath/DWBLocalPlanner:
      plugin: "nav2_dwb_controller::DWBLocalPlanner"
      debug_trajectory_details: False
      min_vel_x: 0.0
      min_vel_y: 0.0
      max_vel_x: 0.5
      max_vel_y: 0.0
      max_vel_theta: 1.0
      min_speed_xy: 0.0
      max_speed_xy: 0.5
      min_speed_theta: 0.0
      acc_lim_x: 2.5
      acc_lim_y: 0.0
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_y: 0.0
      decel_lim_theta: -3.2
      vx_samples: 20
      vy_samples: 5
      vtheta_samples: 20
      sim_time: 1.7
      linear_granularity: 0.05
      angular_granularity: 0.025
      transform_tolerance: 0.2
      xy_goal_tolerance: 0.25
      trans_stopped_velocity: 0.25
      short_circuit_trajectory_evaluation: True
      stateful: True
      critics: ["RotateToGoal", "Oscillation", "BaseObstacle", "GoalAlign", "PathAlign", "PathDist", "GoalDist"]
      BaseObstacle.scale: 0.02
      PathAlign.scale: 32.0
      PathAlign.forward_point_distance: 0.1
      GoalAlign.scale: 24.0
      GoalAlign.forward_point_distance: 0.1
      PathDist.scale: 32.0
      GoalDist.scale: 24.0
      RotateToGoal.scale: 32.0
      RotateToGoal.slowing_factor: 5.0
      RotateToGoal.lookahead_time: -1.0

global_costmap:
  ros__parameters:
    update_frequency: 1.0
    publish_frequency: 1.0
    global_frame: "map"
    robot_base_frame: "base_link"
    use_sim_time: True
    rolling_window: false
    width: 200
    height: 200
    resolution: 0.05
    origin_x: -50.0
    origin_y: -50.0
    always_send_full_costmap: True
    plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
    obstacle_layer:
      plugin: "nav2_costmap_2d::ObstacleLayer"
      enabled: True
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_max_range: 3.0
        raytrace_min_range: 0.0
        obstacle_max_range: 2.5
        obstacle_min_range: 0.0
    static_layer:
      plugin: "nav2_costmap_2d::StaticLayer"
      map_subscribe_transient_local: True
    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      cost_scaling_factor: 3.0
      inflation_radius: 0.55

local_costmap:
  ros__parameters:
    update_frequency: 5.0
    publish_frequency: 2.0
    global_frame: "odom"
    robot_base_frame: "base_link"
    use_sim_time: True
    rolling_window: True
    width: 3
    height: 3
    resolution: 0.05
    always_send_full_costmap: False
    plugins: ["obstacle_layer", "voxel_layer", "inflation_layer"]
    obstacle_layer:
      plugin: "nav2_costmap_2d::ObstacleLayer"
      enabled: True
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_max_range: 3.0
        raytrace_min_range: 0.0
        obstacle_max_range: 2.5
        obstacle_min_range: 0.0
    voxel_layer:
      plugin: "nav2_costmap_2d::VoxelLayer"
      enabled: True
      publish_voxel_map: True
      origin_z: 0.0
      z_resolution: 0.2
      z_voxels: 10
      max_obstacle_height: 2.0
      mark_threshold: 0
      observation_sources: pointcloud
      pointcloud:
        topic: /pointcloud
        max_obstacle_height: 2.0
        min_obstacle_height: 0.0
        clearing: True
        marking: True
        data_type: "PointCloud2"
        raytrace_max_range: 3.0
        raytrace_min_range: 0.0
        obstacle_max_range: 2.5
        obstacle_min_range: 0.0
    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      cost_scaling_factor: 3.0
      inflation_radius: 0.6
```

Example of a Nav2 launch file:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml
import os

def generate_launch_description():
    # Launch configuration variables
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    default_bt_xml_filename = LaunchConfiguration('default_bt_xml_filename')
    map_subscribe_transient_local = LaunchConfiguration('map_subscribe_transient_local')

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare launch arguments
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            get_package_share_directory('my_robot_navigation'),
            'config',
            'nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    declare_bt_xml_cmd = DeclareLaunchArgument(
        'default_bt_xml_filename',
        default_value=os.path.join(
            get_package_share_directory('nav2_bt_navigator'),
            'behavior_trees',
            'navigate_w_replanning_and_recovery.xml'),
        description='Full path to the behavior tree xml file to use')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack')

    declare_map_subscribe_transient_local_cmd = DeclareLaunchArgument(
        'map_subscribe_transient_local',
        default_value='false',
        description='Whether to set the map subscriber QoS to transient local')

    # Map fully qualified names to relative ones so the node's namespace can be prepended.
    # In case of the transforms (tf), currently, there doesn't seem to be a better alternative
    # https://github.com/ros/geometry2/issues/32
    # https://github.com/ros/robot_state_publisher/pull/30
    # TODO(orduno) Substitute with `PushNodeRemapping`
    #              https://github.com/ros2/launch_ros/issues/56
    remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'default_bt_xml_filename': default_bt_xml_filename,
        'autostart': autostart,
        'map_subscribe_transient_local': map_subscribe_transient_local}

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True)

    # Nodes launching commands
    start_lifecycle_manager_cmd = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        namespace=namespace,
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                   {'autostart': autostart},
                   {'node_names': ['map_server',
                                  'planner_server',
                                  'controller_server',
                                  'bt_navigator',
                                  'waypoint_follower',
                                  'velocity_smoother']}])

    # Localization (AMCL)
    start_localization_cmd = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        namespace=namespace,
        output='screen',
        parameters=[configured_params],
        remappings=remappings)

    # Map Server
    start_map_server_cmd = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace=namespace,
        output='screen',
        parameters=[configured_params],
        remappings=remappings)

    # Planner Server
    start_planner_cmd = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        namespace=namespace,
        output='screen',
        parameters=[configured_params],
        remappings=remappings)

    # Controller Server
    start_controller_cmd = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        namespace=namespace,
        output='screen',
        parameters=[configured_params],
        remappings=remappings)

    # Navigation Server
    start_nav_server_cmd = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        namespace=namespace,
        output='screen',
        parameters=[configured_params],
        remappings=remappings)

    # Waypoint Follower Server
    start_waypoint_follower_cmd = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        namespace=namespace,
        output='screen',
        parameters=[configured_params],
        remappings=remappings)

    # Velocity Smoother Server
    start_velocity_smoother_cmd = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        namespace=namespace,
        output='screen',
        parameters=[configured_params],
        remappings=remappings)

    # Add the actions to launch all of the navigation nodes
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_bt_xml_cmd)
    ld.add_action(declare_map_subscribe_transient_local_cmd)

    ld.add_action(start_lifecycle_manager_cmd)
    ld.add_action(start_localization_cmd)
    ld.add_action(start_map_server_cmd)
    ld.add_action(start_planner_cmd)
    ld.add_action(start_controller_cmd)
    ld.add_action(start_nav_server_cmd)
    ld.add_action(start_waypoint_follower_cmd)
    ld.add_action(start_velocity_smoother_cmd)

    return ld
```

Example of a custom path planner for humanoid robots:

```python
import rclpy
from rclpy.node import Node
from nav2_msgs.action import ComputePathToPose
from nav2_msgs.srv import GetCostmap
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from builtin_interfaces.msg import Duration
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from scipy.spatial import distance
import math

class HumanoidPathPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_path_planner')

        # Action server for path computation
        self._action_server = ActionServer(
            self,
            ComputePathToPose,
            'compute_path_to_pose',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            cancel_callback=self.cancel_callback)

        # Service client for costmap
        self.costmap_client = self.create_client(
            GetCostmap,
            'costmap/get_costmap'
        )

        # Parameters for humanoid navigation
        self.foot_separation = 0.2  # Distance between feet
        self.step_height = 0.1      # Maximum step height for humanoid
        self.max_step_length = 0.4  # Maximum step length

        self.get_logger().info('Humanoid Path Planner initialized')

    def execute_callback(self, goal_handle):
        """Execute path planning for humanoid robot"""
        self.get_logger().info('Received path planning request')

        start_pose = goal_handle.request.start
        goal_pose = goal_handle.request.goal

        # Get current costmap
        costmap = self.get_current_costmap()
        if costmap is None:
            self.get_logger().error('Failed to get costmap')
            goal_handle.abort()
            return ComputePathToPose.Result()

        # Plan path considering humanoid constraints
        path = self.plan_humanoid_path(start_pose, goal_pose, costmap)

        if path is None:
            self.get_logger().error('Failed to find valid path')
            goal_handle.abort()
            return ComputePathToPose.Result()

        # Create result
        result = ComputePathToPose.Result()
        result.path = path

        goal_handle.succeed()
        self.get_logger().info('Path planning completed successfully')
        return result

    def get_current_costmap(self):
        """Get current costmap from costmap server"""
        if not self.costmap_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Costmap service not available')
            return None

        request = GetCostmap.Request()
        request.layers = []  # Get all layers

        future = self.costmap_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        try:
            response = future.result()
            return response.map
        except Exception as e:
            self.get_logger().error(f'Failed to get costmap: {str(e)}')
            return None

    def plan_humanoid_path(self, start_pose, goal_pose, costmap):
        """Plan path considering humanoid robot constraints"""
        # Convert poses to grid coordinates
        start_x = int((start_pose.pose.position.x - costmap.info.origin.position.x) / costmap.info.resolution)
        start_y = int((start_pose.pose.position.y - costmap.info.origin.position.y) / costmap.info.resolution)
        goal_x = int((goal_pose.pose.position.x - costmap.info.origin.position.x) / costmap.info.resolution)
        goal_y = int((goal_pose.pose.position.y - costmap.info.origin.position.y) / costmap.info.resolution)

        # Check if start and goal are within map bounds
        if (start_x < 0 or start_x >= costmap.info.width or
            start_y < 0 or start_y >= costmap.info.height or
            goal_x < 0 or goal_x >= costmap.info.width or
            goal_y < 0 or goal_y >= costmap.info.height):
            self.get_logger().error('Start or goal pose outside map bounds')
            return None

        # Check if start and goal are in free space
        start_idx = start_y * costmap.info.width + start_x
        goal_idx = goal_y * costmap.info.width + goal_x

        if (costmap.data[start_idx] >= 50 or costmap.data[goal_idx] >= 50):  # 50 is obstacle threshold
            self.get_logger().error('Start or goal pose in obstacle space')
            return None

        # Implement A* path planning algorithm with humanoid constraints
        path = self.a_star_with_constraints(
            (start_x, start_y),
            (goal_x, goal_y),
            costmap,
            self.max_step_length / costmap.info.resolution  # Convert to grid units
        )

        if path is None:
            return None

        # Convert grid path to world coordinates
        world_path = Path()
        world_path.header.frame_id = "map"
        world_path.header.stamp = self.get_clock().now().to_msg()

        for x, y in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x * costmap.info.resolution + costmap.info.origin.position.x
            pose.pose.position.y = y * costmap.info.resolution + costmap.info.origin.position.y
            pose.pose.position.z = 0.0  # Assume flat terrain for now
            pose.pose.orientation.w = 1.0  # No rotation
            world_path.poses.append(pose)

        return world_path

    def a_star_with_constraints(self, start, goal, costmap, max_step_cells):
        """A* algorithm with humanoid movement constraints"""
        import heapq

        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        def get_neighbors(pos):
            """Get valid neighbors considering step constraints"""
            neighbors = []
            # Consider 8-connected neighborhood but with step length constraints
            for dx in range(-int(max_step_cells), int(max_step_cells) + 1):
                for dy in range(-int(max_step_cells), int(max_step_cells) + 1):
                    if dx == 0 and dy == 0:
                        continue
                    # Check step length constraint
                    step_dist = math.sqrt(dx*dx + dy*dy)
                    if step_dist <= max_step_cells and step_dist > 0:
                        new_pos = (pos[0] + dx, pos[1] + dy)
                        # Check bounds
                        if (0 <= new_pos[0] < costmap.info.width and
                            0 <= new_pos[1] < costmap.info.height):
                            # Check if path to this neighbor is traversable
                            if self.is_path_clear(pos, new_pos, costmap):
                                neighbors.append(new_pos)
            return neighbors

        def is_path_clear(self, start, end, costmap):
            """Check if path between two points is clear of obstacles"""
            # Use Bresenham's line algorithm to check path
            x0, y0 = start
            x1, y1 = end

            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            x, y = x0, y0
            while True:
                # Check if current cell is traversable
                idx = y * costmap.info.width + x
                if idx < len(costmap.data) and costmap.data[idx] >= 50:  # Obstacle
                    return False

                if x == x1 and y == y1:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

            return True

        # A* algorithm implementation
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current_cost, current = heapq.heappop(frontier)

            if current == goal:
                break

            for next_pos in get_neighbors(current):
                new_cost = cost_so_far[current] + heuristic(current, next_pos)

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        # Reconstruct path
        if goal not in came_from:
            return None

        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()

        return path

    def cancel_callback(self, goal_handle):
        """Handle action cancellation"""
        self.get_logger().info('Path planning canceled')
        return CancelResponse.ACCEPT

def main(args=None):
    rclpy.init(args=args)
    planner = HumanoidPathPlanner()

    try:
        executor = MultiThreadedExecutor()
        rclpy.spin(planner, executor=executor)
    except KeyboardInterrupt:
        planner.get_logger().info('Shutting down Humanoid Path Planner...')
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercises

1. **Nav2 Configuration**: Configure Nav2 for a humanoid robot with appropriate parameters for bipedal locomotion.

2. **Path Planning**: Implement a custom path planner that considers humanoid-specific constraints like step height and length.

3. **Costmap Tuning**: Adjust costmap parameters for humanoid navigation considering the robot's size and stability requirements.

4. **Obstacle Avoidance**: Implement dynamic obstacle avoidance for a humanoid robot navigating in a crowded environment.

## Summary

Navigation 2 provides a robust and flexible framework for autonomous navigation that can be adapted for humanoid robots with specialized considerations for bipedal locomotion. The behavior tree architecture, combined with modular components for path planning, control, and recovery, enables sophisticated navigation behaviors. For humanoid robots, special attention must be paid to kinematic constraints, stability considerations, and the unique challenges of two-legged locomotion. As we continue our exploration of Physical AI, Nav2 will serve as the foundation for enabling autonomous navigation capabilities in humanoid robotic systems.