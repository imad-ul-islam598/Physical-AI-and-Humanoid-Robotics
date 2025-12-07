---
sidebar_position: 18
---

# Chapter 18: Manipulation

## Introduction

Manipulation is a fundamental capability for humanoid robots, enabling them to interact with objects in their environment through grasping, lifting, moving, and placing operations. Unlike simple grippers or specialized end-effectors, humanoid manipulation involves complex whole-body coordination, integrating arm movements, torso positioning, and sometimes even stepping to achieve manipulation goals. This chapter explores the principles of humanoid manipulation, grasp planning, coordinated control strategies, and the integration of perception and action for dexterous manipulation.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the principles of dexterous manipulation in humanoid robots
- Implement grasp planning and execution for various object types
- Design coordinated manipulation strategies using multiple degrees of freedom
- Integrate perception and manipulation for object interaction
- Apply force control and compliance for safe manipulation

## Key Concepts

- **Dexterous Manipulation**: Skillful manipulation using multi-fingered hands
- **Grasp Planning**: Determining optimal grasp points and hand configurations
- **Coordinated Control**: Whole-body coordination for manipulation tasks
- **Force Control**: Controlling interaction forces during manipulation
- **Compliance Control**: Allowing controlled flexibility in manipulation
- **Prehensile Grasping**: Grasping with fingers to hold objects
- **Non-prehensile Manipulation**: Manipulating without grasping (pushing, sliding)
- **Bimanual Coordination**: Using both hands together for complex tasks

## Technical Explanation

Humanoid manipulation is significantly more complex than manipulation with simple robotic arms due to the need for whole-body coordination and the anthropomorphic nature of the manipulator. Humanoid robots must coordinate multiple degrees of freedom across arms, torso, and sometimes legs to achieve manipulation goals while maintaining balance and stability.

**Grasp Planning** in humanoid robots involves multiple considerations:

1. **Geometric Analysis**: Understanding object shape, size, and surface properties
2. **Force Analysis**: Ensuring the grasp can support the object's weight and applied forces
3. **Accessibility**: Ensuring the hand can reach the planned grasp points
4. **Stability**: Maintaining balance during the manipulation task
5. **Dexterity**: Planning finger configurations for complex manipulation

The grasp planning process typically includes:
- Object recognition and pose estimation
- Surface analysis for potential grasp points
- Force closure analysis to ensure stable grasps
- Collision checking to avoid self-collision
- Kinematic feasibility verification

**Coordinated Manipulation Control** involves several key components:

1. **Task Prioritization**: Managing multiple simultaneous objectives (end-effector position, balance, joint limits)
2. **Null Space Projection**: Using redundancy to achieve secondary objectives while maintaining primary tasks
3. **Whole-Body Control**: Coordinating all available degrees of freedom for optimal performance
4. **Balance Integration**: Maintaining stability during manipulation

The control hierarchy typically follows:
- High-level task planning (what to manipulate, how to approach)
- Mid-level coordination (arm, torso, and balance coordination)
- Low-level joint control (individual joint trajectories)

**Force and Compliance Control** is essential for safe and effective manipulation:

- **Impedance Control**: Controlling the robot's mechanical impedance to environmental interactions
- **Admittance Control**: Controlling motion in response to applied forces
- **Hybrid Force/Position Control**: Combining position and force control for different task aspects

**Bimanual Coordination** adds another layer of complexity, requiring:
- Task allocation between hands
- Coordination of motion and forces
- Management of kinematic constraints
- Communication between control systems

The mathematical framework for humanoid manipulation typically uses:
- **Task Space**: Cartesian space for end-effector positions and forces
- **Joint Space**: Configuration space for joint angles
- **Operational Space**: Space where forces and motions are controlled

The relationship between joint torques τ and task forces F is given by:
```
τ = J^T * F + τ_null
```

Where J is the Jacobian matrix and τ_null represents torques in the null space of the primary task.

## Diagrams written as text descriptions

**Diagram 1: Humanoid Manipulation Control Hierarchy**
```
High-Level Task Planner
         │
         ▼
Mid-Level Coordinated Controller
         │
    ┌────┴────┐
    │         │
Arm Control  Balance Control
    │         │
    ▼         ▼
Low-Level Joint Controllers
    │         │
    └────┬────┘
         │
         ▼
Robot Hardware
```

**Diagram 2: Grasp Planning Process**
```
Object Recognition → Pose Estimation → Surface Analysis → Grasp Candidate Generation
         │                   │                  │                    │
         ▼                   ▼                  ▼                    ▼
   3D Model       Object Pose      Surface Normals    Grasp Points & Orientations
         │                   │                  │                    │
         ▼                   ▼                  ▼                    ▼
Force Analysis ← Stability Check ← Collision Check ← Kinematic Feasibility
         │                   │                  │                    │
         └───────────────────┼──────────────────┼────────────────────┘
                             ▼                  ▼
                        Grasp Quality    Grasp Selection
                        Evaluation       (Best Grasp)
```

**Diagram 3: Bimanual Coordination Example**
```
     Left Hand Goal ←─────────────────────────→ Right Hand Goal
           │                                           │
           ▼                                           ▼
    Object Interaction                    Object Interaction
    (e.g., holding)                       (e.g., turning)
           │                                           │
           └───────────────────────────────────────────┘
                             │
                             ▼
                     Coordinated Motion
                     (Synchronized action)
```

## Code Examples

Here's an example of a humanoid manipulation system with grasp planning and coordinated control:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class HumanoidManipulationSystem:
    def __init__(self):
        """
        Initialize the humanoid manipulation system
        """
        # Robot configuration
        self.arm_dof = 7  # 7 DOF arms
        self.torso_dof = 3  # 3 DOF torso (pitch, roll, yaw)

        # Hand configuration (simplified)
        self.hand_fingers = 5  # 5 fingers per hand
        self.finger_joints = 3  # 3 joints per finger (simplified)

        # Initialize kinematic models
        self.right_arm_kin = self.initialize_arm_kinematics('right')
        self.left_arm_kin = self.initialize_arm_kinematics('left')

        # Initialize grasp planners
        self.right_grasp_planner = GraspPlanner('right')
        self.left_grasp_planner = GraspPlanner('left')

        # Initialize controllers
        self.impedance_controller = ImpedanceController()
        self.coordinated_controller = CoordinatedController()

        # Robot state
        self.current_joint_angles = {
            'right_arm': np.zeros(self.arm_dof),
            'left_arm': np.zeros(self.arm_dof),
            'torso': np.zeros(self.torso_dof),
            'right_hand': np.zeros(self.hand_fingers * self.finger_joints),
            'left_hand': np.zeros(self.hand_fingers * self.finger_joints)
        }

    def initialize_arm_kinematics(self, arm_type):
        """
        Initialize kinematic model for arm
        """
        class ArmKinematics:
            def __init__(self, arm_type):
                self.arm_type = arm_type
                # DH parameters would be defined here for actual implementation
                self.dh_params = self.get_dh_params(arm_type)

            def get_dh_params(self, arm_type):
                # Simplified DH parameters for 7-DOF arm
                if arm_type == 'right':
                    return [
                        [0.0, np.pi/2, 0.2, 0.0],      # Shoulder yaw
                        [0.0, -np.pi/2, 0.0, 0.0],     # Shoulder pitch
                        [0.0, np.pi/2, 0.25, 0.0],     # Shoulder roll
                        [0.0, -np.pi/2, 0.0, 0.0],     # Elbow pitch
                        [0.0, np.pi/2, 0.25, 0.0],     # Wrist pitch
                        [0.0, -np.pi/2, 0.0, 0.0],     # Wrist roll
                        [0.0, 0.0, 0.1, 0.0]           # Wrist yaw
                    ]
                else:  # left arm
                    return [
                        [0.0, np.pi/2, 0.2, 0.0],      # Shoulder yaw
                        [0.0, -np.pi/2, 0.0, 0.0],     # Shoulder pitch
                        [0.0, -np.pi/2, 0.25, 0.0],    # Shoulder roll (different sign)
                        [0.0, -np.pi/2, 0.0, 0.0],     # Elbow pitch
                        [0.0, np.pi/2, 0.25, 0.0],     # Wrist pitch
                        [0.0, -np.pi/2, 0.0, 0.0],     # Wrist roll
                        [0.0, 0.0, 0.1, 0.0]           # Wrist yaw
                    ]

            def forward_kinematics(self, joint_angles):
                # Simplified forward kinematics
                # In practice, this would use DH parameters and transformation matrices
                position = np.array([0.5, 0.3, 0.2])  # Placeholder
                orientation = np.array([0, 0, 0, 1])  # Placeholder quaternion

                return {
                    'position': position,
                    'orientation': orientation
                }

        return ArmKinematics(arm_type)

    def plan_grasp(self, object_info, hand_type='right'):
        """
        Plan a grasp for the specified object
        object_info: dict with object properties
        hand_type: 'right' or 'left'
        """
        if hand_type == 'right':
            grasp_planner = self.right_grasp_planner
        else:
            grasp_planner = self.left_grasp_planner

        # Plan grasp based on object properties
        grasp = grasp_planner.plan_grasp(object_info)

        return grasp

    def execute_manipulation_task(self, task_description):
        """
        Execute a manipulation task
        task_description: dict describing the manipulation task
        """
        # Parse task description
        task_type = task_description.get('type', 'grasp')
        target_object = task_description.get('object', {})
        target_pose = task_description.get('target_pose', None)
        hand_preference = task_description.get('hand', 'right')

        if task_type == 'grasp':
            return self.execute_grasp_task(target_object, hand_preference)
        elif task_type == 'place':
            return self.execute_place_task(target_pose, hand_preference)
        elif task_type == 'move':
            return self.execute_move_task(target_object, target_pose, hand_preference)
        elif task_type == 'bimanual':
            return self.execute_bimanual_task(task_description)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def execute_grasp_task(self, object_info, hand_type='right'):
        """
        Execute a grasping task
        """
        # Plan the grasp
        grasp_plan = self.plan_grasp(object_info, hand_type)
        if not grasp_plan:
            raise ValueError("Could not plan grasp for object")

        # Get current arm configuration
        current_arm_angles = self.current_joint_angles[f'{hand_type}_arm']

        # Calculate approach pose (pre-grasp)
        approach_pose = self.calculate_approach_pose(grasp_plan['pose'],
                                                   grasp_plan['approach_distance'])

        # Execute approach
        success = self.move_to_pose(hand_type, approach_pose)
        if not success:
            return False

        # Execute grasp
        success = self.execute_grasp(hand_type, grasp_plan)
        if not success:
            return False

        # Lift object slightly
        lift_offset = np.array([0, 0, 0.05])  # Lift 5cm
        lift_pose = self.offset_pose(grasp_plan['pose'], lift_offset)
        success = self.move_to_pose(hand_type, lift_pose)

        return success

    def execute_place_task(self, target_pose, hand_type='right'):
        """
        Execute a placing task
        """
        # Move to target pose
        success = self.move_to_pose(hand_type, target_pose)
        if not success:
            return False

        # Open hand to release object
        success = self.open_hand(hand_type)

        # Move away from placed object
        retract_offset = np.array([0, 0, 0.1])  # Move up 10cm
        retract_pose = self.offset_pose(target_pose, retract_offset)
        self.move_to_pose(hand_type, retract_pose)

        return success

    def move_to_pose(self, hand_type, target_pose):
        """
        Move the specified hand to target pose
        """
        # Calculate inverse kinematics
        if hand_type == 'right':
            arm_kin = self.right_arm_kin
            current_angles = self.current_joint_angles['right_arm']
        else:
            arm_kin = self.left_arm_kin
            current_angles = self.current_joint_angles['left_arm']

        # In practice, this would solve inverse kinematics
        # For now, we'll simulate the movement
        target_angles = current_angles + 0.1  # Placeholder

        # Update joint angles
        self.current_joint_angles[f'{hand_type}_arm'] = target_angles

        print(f"Moved {hand_type} hand to pose: {target_pose}")
        return True

    def execute_grasp(self, hand_type, grasp_plan):
        """
        Execute the grasp with the specified hand
        """
        # Close fingers to grasp object
        success = self.close_hand(hand_type, grasp_plan['grasp_type'])

        if success:
            print(f"Successfully grasped object with {hand_type} hand")

        return success

    def close_hand(self, hand_type, grasp_type='power'):
        """
        Close the hand with specified grasp type
        """
        # Set finger joint angles based on grasp type
        if grasp_type == 'power':
            # Power grasp - close all fingers firmly
            finger_angles = np.full(self.hand_fingers * self.finger_joints, 1.0)
        elif grasp_type == 'precision':
            # Precision grasp - use fingertips
            finger_angles = np.full(self.hand_fingers * self.finger_joints, 0.5)
        else:
            finger_angles = np.zeros(self.hand_fingers * self.finger_joints)

        self.current_joint_angles[f'{hand_type}_hand'] = finger_angles

        return True

    def open_hand(self, hand_type):
        """
        Open the hand
        """
        self.current_joint_angles[f'{hand_type}_hand'] = np.zeros(
            self.hand_fingers * self.finger_joints)
        return True

    def calculate_approach_pose(self, grasp_pose, approach_distance):
        """
        Calculate approach pose for pre-grasp
        """
        # Approach from the opposite direction of the grasp approach
        approach_direction = grasp_pose.get('approach_direction', [0, 0, -1])
        approach_vec = np.array(approach_direction) * approach_distance

        approach_pose = grasp_pose.copy()
        approach_pose['position'] = grasp_pose['position'] - approach_vec

        return approach_pose

    def offset_pose(self, pose, offset):
        """
        Apply offset to pose
        """
        new_pose = pose.copy()
        new_pose['position'] = pose['position'] + np.array(offset)
        return new_pose

    def execute_bimanual_task(self, task_description):
        """
        Execute a task requiring both hands
        """
        # Parse bimanual task
        action = task_description.get('action', 'hold')
        object_info = task_description.get('object', {})

        if action == 'hold':
            # Grasp object with both hands
            right_grasp = self.plan_grasp(object_info, 'right')
            left_grasp = self.plan_grasp(object_info, 'left')

            # Execute coordinated grasp
            success_right = self.execute_grasp('right', right_grasp)
            success_left = self.execute_grasp('left', left_grasp)

            return success_right and success_left

        elif action == 'assemble':
            # Use one hand to hold, other to manipulate
            primary_hand = task_description.get('primary_hand', 'right')
            secondary_hand = 'left' if primary_hand == 'right' else 'right'

            # Grasp object with primary hand
            primary_grasp = self.plan_grasp(object_info, primary_hand)
            primary_success = self.execute_grasp(primary_hand, primary_grasp)

            if primary_success:
                # Perform secondary action with other hand
                secondary_action = task_description.get('secondary_action', {})
                # Execute secondary action
                return True  # Placeholder

        return False

class GraspPlanner:
    def __init__(self, hand_type):
        self.hand_type = hand_type
        self.object_properties = {
            'cylindrical': {
                'preferred_grasps': ['cylindrical', 'top_grasp'],
                'approach_distance': 0.15
            },
            'box': {
                'preferred_grasps': ['top_grasp', 'side_grasp'],
                'approach_distance': 0.20
            },
            'spherical': {
                'preferred_grasps': ['spherical', 'pinch'],
                'approach_distance': 0.10
            }
        }

    def plan_grasp(self, object_info):
        """
        Plan grasp based on object properties
        """
        obj_type = object_info.get('type', 'unknown')
        obj_size = object_info.get('size', [0.1, 0.1, 0.1])
        obj_weight = object_info.get('weight', 0.5)
        obj_pose = object_info.get('pose', {
            'position': [0, 0, 0],
            'orientation': [0, 0, 0, 1]
        })

        # Determine best grasp type based on object properties
        if obj_type in self.object_properties:
            grasp_options = self.object_properties[obj_type]['preferred_grasps']
            approach_distance = self.object_properties[obj_type]['approach_distance']
        else:
            # Default for unknown objects
            grasp_options = ['power', 'pinch']
            approach_distance = 0.15

        # Select grasp type (in practice, this would be more sophisticated)
        selected_grasp = grasp_options[0]

        # Create grasp plan
        grasp_plan = {
            'pose': obj_pose,
            'grasp_type': selected_grasp,
            'approach_distance': approach_distance,
            'required_force': min(50.0, obj_weight * 20.0),  # Force based on weight
            'stability_score': 0.9  # Placeholder
        }

        return grasp_plan

class ImpedanceController:
    def __init__(self):
        # Impedance control parameters
        self.position_gain = 1000.0  # N/m
        self.velocity_gain = 200.0   # Ns/m
        self.orientation_gain = 100.0  # Nm/rad
        self.angular_velocity_gain = 20.0  # Nms/rad

    def control_impedance(self, desired_pose, current_pose, desired_wrench=None):
        """
        Control the mechanical impedance of the end-effector
        """
        if desired_wrench is None:
            desired_wrench = np.zeros(6)  # No desired external force

        # Calculate position and orientation errors
        pos_error = desired_pose['position'] - current_pose['position']

        # For orientation, we need to calculate orientation error
        # This is a simplified approach
        orientation_error = np.zeros(3)  # Placeholder

        # Calculate required forces to achieve desired impedance
        force = (self.position_gain * pos_error -
                self.velocity_gain * current_pose.get('velocity', np.zeros(3)))

        torque = (self.orientation_gain * orientation_error -
                 self.angular_velocity_gain * current_pose.get('angular_velocity', np.zeros(3)))

        # Combine with desired external wrench
        total_wrench = np.concatenate([force, torque]) + desired_wrench

        return total_wrench

class CoordinatedController:
    def __init__(self):
        # Priorities for different tasks
        self.task_priorities = {
            'balance': 1,      # Highest priority
            'end_effector': 2, # High priority
            'avoid_collision': 3, # Medium priority
            'posture': 4       # Lowest priority
        }

    def coordinate_tasks(self, tasks, robot_state):
        """
        Coordinate multiple tasks with different priorities
        """
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda x: self.task_priorities[x['type']])

        # Initialize joint velocities
        joint_velocities = np.zeros(robot_state['joint_angles'].shape)

        for task in sorted_tasks:
            if task['type'] == 'balance':
                # Balance task - adjust COM position
                balance_vel = self.balance_task(task, robot_state)
                joint_velocities += balance_vel

            elif task['type'] == 'end_effector':
                # End-effector task - move to desired position
                ee_vel = self.end_effector_task(task, robot_state)
                joint_velocities += ee_vel

            elif task['type'] == 'avoid_collision':
                # Collision avoidance task
                collision_vel = self.collision_avoidance_task(task, robot_state)
                joint_velocities += collision_vel

            elif task['type'] == 'posture':
                # Posture task - maintain comfortable joint configuration
                posture_vel = self.posture_task(task, robot_state)
                joint_velocities += posture_vel

        return joint_velocities

    def balance_task(self, task, robot_state):
        """
        Calculate joint velocities for balance maintenance
        """
        # This would integrate with balance control system
        return np.zeros(robot_state['joint_angles'].shape[0])

    def end_effector_task(self, task, robot_state):
        """
        Calculate joint velocities for end-effector movement
        """
        # Calculate Jacobian for the end-effector
        # In practice, this would use the actual kinematic model
        jacobian = np.random.rand(6, len(robot_state['joint_angles']))  # Placeholder

        # Calculate desired end-effector velocity
        desired_ee_vel = task['desired_velocity']

        # Calculate joint velocities using Jacobian pseudoinverse
        joint_velocities = np.linalg.pinv(jacobian) @ desired_ee_vel

        return joint_velocities

    def collision_avoidance_task(self, task, robot_state):
        """
        Calculate joint velocities for collision avoidance
        """
        # This would use distance sensors and obstacle information
        return np.zeros(robot_state['joint_angles'].shape[0])

    def posture_task(self, task, robot_state):
        """
        Calculate joint velocities for posture maintenance
        """
        # Move towards desired joint configuration
        desired_joints = task['desired_joint_angles']
        current_joints = robot_state['joint_angles']

        joint_error = desired_joints - current_joints
        joint_velocities = 0.1 * joint_error  # Simple proportional control

        return joint_velocities

# Example usage
def main():
    # Initialize manipulation system
    manipulator = HumanoidManipulationSystem()

    # Define an object to manipulate
    object_info = {
        'type': 'cylindrical',
        'size': [0.05, 0.05, 0.15],  # diameter, diameter, height
        'weight': 0.3,  # kg
        'pose': {
            'position': np.array([0.5, 0.2, 0.1]),
            'orientation': np.array([0, 0, 0, 1])  # quaternion
        }
    }

    # Create a manipulation task
    grasp_task = {
        'type': 'grasp',
        'object': object_info,
        'hand': 'right'
    }

    print("Executing grasp task...")
    success = manipulator.execute_manipulation_task(grasp_task)
    print(f"Grasp task success: {success}")

    # Create a placing task
    place_pose = {
        'position': np.array([0.6, -0.1, 0.1]),
        'orientation': np.array([0, 0, 0, 1])
    }

    place_task = {
        'type': 'place',
        'target_pose': place_pose,
        'hand': 'right'
    }

    print("Executing place task...")
    success = manipulator.execute_manipulation_task(place_task)
    print(f"Place task success: {success}")

    # Create a bimanual task
    bimanual_task = {
        'type': 'bimanual',
        'action': 'hold',
        'object': object_info
    }

    print("Executing bimanual task...")
    success = manipulator.execute_manipulation_task(bimanual_task)
    print(f"Bimanual task success: {success}")

if __name__ == '__main__':
    main()
```

Example of force control and compliance in manipulation:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class ForceComplianceController:
    def __init__(self, robot_mass=50.0, gravity=9.81):
        """
        Initialize force and compliance controller
        """
        self.robot_mass = robot_mass
        self.gravity = gravity

        # Force control parameters
        self.force_gain = 100.0  # N/m for force error
        self.compliance_stiffness = 1000.0  # N/m for compliance
        self.compliance_damping = 200.0     # Ns/m for compliance

        # Safety limits
        self.max_force = 200.0  # Maximum allowed force (N)
        self.max_torque = 50.0  # Maximum allowed torque (Nm)

        # Current state
        self.current_contact_force = np.zeros(3)
        self.current_contact_torque = np.zeros(3)
        self.desired_contact_force = np.zeros(3)
        self.contact_surface_normal = np.array([0, 0, 1])  # Default: surface facing up

    def hybrid_force_position_control(self, position_error, force_error,
                                    force_control_axes, position_control_axes):
        """
        Hybrid force/position control
        force_control_axes: list of axes (0,1,2 for x,y,z) to control force
        position_control_axes: list of axes to control position
        """
        # Initialize control outputs
        force_command = np.zeros(3)
        position_command = np.zeros(3)

        # Position control for specified axes
        for axis in position_control_axes:
            position_command[axis] = 1000 * position_error[axis]  # High stiffness for position control

        # Force control for specified axes
        for axis in force_control_axes:
            force_command[axis] = self.force_gain * force_error[axis]

        # Combine into a single control command
        # This is simplified - in practice, you'd need to consider the robot's dynamics
        control_output = {
            'position_command': position_command,
            'force_command': force_command
        }

        return control_output

    def admittance_control(self, applied_force, desired_admittance=0.001):
        """
        Admittance control: motion response to applied forces
        desired_admittance: how much motion per unit force (m/N)
        """
        # Calculate desired motion based on applied force
        desired_motion = desired_admittance * applied_force

        # This represents how the robot should move in response to forces
        return desired_motion

    def impedance_control(self, desired_pose, current_pose, desired_force=np.zeros(3)):
        """
        Impedance control: control robot's mechanical impedance
        """
        # Calculate pose error
        position_error = desired_pose['position'] - current_pose['position']
        orientation_error = self.calculate_orientation_error(
            desired_pose['orientation'], current_pose['orientation'])

        # Calculate velocities (if available)
        current_velocity = current_pose.get('velocity', np.zeros(3))
        current_angular_velocity = current_pose.get('angular_velocity', np.zeros(3))

        # Impedance control law: F = K(x_d - x) + D(v_d - v)
        position_stiffness = 1000.0  # N/m
        position_damping = 200.0     # Ns/m
        orientation_stiffness = 100.0  # Nm/rad
        orientation_damping = 20.0     # Nms/rad

        force_feedback = (position_stiffness * position_error -
                         position_damping * current_velocity)
        torque_feedback = (orientation_stiffness * orientation_error -
                          orientation_damping * current_angular_velocity)

        # Add desired force/torque
        total_force = force_feedback + desired_force
        total_torque = torque_feedback

        # Safety checks
        if np.linalg.norm(total_force) > self.max_force:
            total_force = total_force * self.max_force / np.linalg.norm(total_force)

        return {
            'force': total_force,
            'torque': total_torque,
            'motion_command': np.concatenate([position_error, orientation_error])
        }

    def calculate_orientation_error(self, desired_quat, current_quat):
        """
        Calculate orientation error between two quaternions
        """
        # Convert quaternions to rotation matrices
        desired_rot = R.from_quat(desired_quat).as_matrix()
        current_rot = R.from_quat(current_quat).as_matrix()

        # Calculate relative rotation
        relative_rot = desired_rot @ current_rot.T

        # Convert to axis-angle representation
        r = R.from_matrix(relative_rot)
        axis_angle = r.as_rotvec()

        return axis_angle

    def cartesian_impedance_control(self, desired_pose, current_pose,
                                  stiffness_matrix=None, damping_matrix=None):
        """
        Cartesian impedance control with 6-DOF stiffness and damping
        """
        if stiffness_matrix is None:
            stiffness_matrix = np.diag([1000, 1000, 1000, 100, 100, 100])  # [x,y,z,rx,ry,rz]

        if damping_matrix is None:
            damping_matrix = np.diag([200, 200, 200, 20, 20, 20])

        # Calculate pose error in Cartesian space
        pos_error = desired_pose['position'] - current_pose['position']

        # Calculate orientation error
        orientation_error = self.calculate_orientation_error(
            desired_pose['orientation'], current_pose['orientation'])

        pose_error = np.concatenate([pos_error, orientation_error])

        # Calculate velocity error if available
        current_twist = np.concatenate([
            current_pose.get('velocity', np.zeros(3)),
            current_pose.get('angular_velocity', np.zeros(3))
        ])
        desired_twist = np.zeros(6)  # For now, assume zero desired velocity

        velocity_error = desired_twist - current_twist

        # Apply impedance control law
        force_torque = stiffness_matrix @ pose_error + damping_matrix @ velocity_error

        # Extract force and torque
        force = force_torque[:3]
        torque = force_torque[3:]

        return {
            'wrench': force_torque,
            'force': force,
            'torque': torque
        }

    def safe_force_control(self, desired_force, current_force, safety_factor=0.8):
        """
        Apply safety-limited force control
        """
        # Calculate the force we want to apply
        force_error = desired_force - current_force
        force_command = 50.0 * force_error  # Proportional control

        # Apply safety limits
        force_magnitude = np.linalg.norm(force_command)
        if force_magnitude > self.max_force * safety_factor:
            force_command = force_command * (self.max_force * safety_factor) / force_magnitude

        return force_command

class TactileFeedbackIntegrator:
    def __init__(self):
        """
        Integrate tactile feedback for manipulation
        """
        self.tactile_sensors = {
            'right_hand': np.zeros(20),  # 20 taxels per hand
            'left_hand': np.zeros(20)
        }
        self.slip_detection_threshold = 0.1
        self.pressure_threshold = 0.05

    def process_tactile_data(self, hand, tactile_readings):
        """
        Process tactile sensor data
        """
        self.tactile_sensors[hand] = tactile_readings

        # Detect slip
        slip_detected = self.detect_slip(hand)

        # Detect object properties
        object_properties = self.estimate_object_properties(hand)

        # Calculate grasp stability
        grasp_stability = self.assess_grasp_stability(hand)

        return {
            'slip_detected': slip_detected,
            'object_properties': object_properties,
            'grasp_stability': grasp_stability,
            'tactile_map': tactile_readings
        }

    def detect_slip(self, hand):
        """
        Detect if object is slipping from grasp
        """
        tactile_data = self.tactile_sensors[hand]

        # Simple slip detection based on rapid changes in tactile readings
        if len(tactile_data) > 1:
            slip_score = np.std(np.diff(tactile_data))
            return slip_score > self.slip_detection_threshold

        return False

    def estimate_object_properties(self, hand):
        """
        Estimate object properties from tactile data
        """
        tactile_data = self.tactile_sensors[hand]

        # Estimate object size based on contact area
        contact_area = np.sum(tactile_data > self.pressure_threshold)

        # Estimate object compliance based on pressure distribution
        avg_pressure = np.mean(tactile_data[tactile_data > self.pressure_threshold])

        # Estimate object shape based on contact pattern
        # This is highly simplified
        object_shape = "unknown"
        if contact_area < 5:
            object_shape = "small"
        elif contact_area < 10:
            object_shape = "medium"
        else:
            object_shape = "large"

        return {
            'size': contact_area,
            'compliance': avg_pressure,
            'shape_category': object_shape
        }

    def assess_grasp_stability(self, hand):
        """
        Assess the stability of the current grasp
        """
        tactile_data = self.tactile_sensors[hand]

        # Calculate grasp stability based on pressure distribution
        active_taxels = tactile_data > self.pressure_threshold
        pressure_distribution = tactile_data[active_taxels]

        if len(pressure_distribution) == 0:
            return 0.0  # No contact

        # Stability is higher when pressure is distributed across multiple taxels
        stability = min(1.0, len(pressure_distribution) / 10.0)  # Normalize

        # Also consider the uniformity of pressure
        if len(pressure_distribution) > 1:
            pressure_variance = np.var(pressure_distribution)
            stability *= (1.0 - min(0.5, pressure_variance / 100.0))  # Penalize uneven pressure

        return stability

# Example of manipulation skill learning
class ManipulationSkillLearner:
    def __init__(self):
        self.skills = {}
        self.demonstrations = []
        self.current_skill = None

    def record_demonstration(self, skill_name, joint_trajectory, end_effector_trajectory):
        """
        Record a manipulation demonstration for learning
        """
        demonstration = {
            'skill_name': skill_name,
            'joint_trajectory': joint_trajectory,
            'end_effector_trajectory': end_effector_trajectory,
            'timestamp': np.datetime64('now')
        }

        self.demonstrations.append(demonstration)

        # Update or create skill model
        self.update_skill_model(skill_name)

    def update_skill_model(self, skill_name):
        """
        Update the model for a specific skill based on demonstrations
        """
        # Filter demonstrations for this skill
        skill_demos = [demo for demo in self.demonstrations if demo['skill_name'] == skill_name]

        if not skill_demos:
            return

        # Simple averaging approach (in practice, use more sophisticated methods like GMM/GMR)
        avg_joint_trajectory = np.mean([demo['joint_trajectory'] for demo in skill_demos], axis=0)
        avg_ee_trajectory = np.mean([demo['end_effector_trajectory'] for demo in skill_demos], axis=0)

        self.skills[skill_name] = {
            'avg_joint_trajectory': avg_joint_trajectory,
            'avg_ee_trajectory': avg_ee_trajectory,
            'num_demonstrations': len(skill_demos)
        }

    def execute_learned_skill(self, skill_name, start_state, goal_state):
        """
        Execute a learned manipulation skill
        """
        if skill_name not in self.skills:
            raise ValueError(f"Skill {skill_name} not learned")

        skill_model = self.skills[skill_name]

        # Adapt the learned trajectory to the current start and goal states
        adapted_trajectory = self.adapt_trajectory_to_context(
            skill_model['avg_ee_trajectory'], start_state, goal_state)

        return adapted_trajectory

    def adapt_trajectory_to_context(self, base_trajectory, start_state, goal_state):
        """
        Adapt a learned trajectory to new start and goal conditions
        """
        # This is a simplified adaptation
        # In practice, use techniques like Dynamic Movement Primitives (DMPs)

        # Calculate transformation from base start to current start
        base_start = base_trajectory[0] if len(base_trajectory) > 0 else np.zeros(6)
        base_goal = base_trajectory[-1] if len(base_trajectory) > 0 else np.zeros(6)

        # Simple linear scaling
        adapted_trajectory = []
        for point in base_trajectory:
            # Interpolate based on progress along the trajectory
            t = np.linspace(0, 1, len(base_trajectory))
            current_t = t[list(base_trajectory).index(point)] if point in base_trajectory else 0

            adapted_point = (1 - current_t) * start_state + current_t * goal_state
            adapted_trajectory.append(adapted_point)

        return np.array(adapted_trajectory)

def main():
    print("Humanoid Manipulation System with Force Control")
    print("=" * 50)

    # Initialize controllers
    force_ctrl = ForceComplianceController()
    tactile_integrator = TactileFeedbackIntegrator()
    skill_learner = ManipulationSkillLearner()

    # Example: Perform a compliant manipulation task
    desired_pose = {
        'position': np.array([0.5, 0.2, 0.3]),
        'orientation': np.array([0, 0, 0, 1])
    }

    current_pose = {
        'position': np.array([0.4, 0.15, 0.25]),
        'orientation': np.array([0, 0, 0, 1]),
        'velocity': np.array([0.01, 0.01, 0.02]),
        'angular_velocity': np.array([0.001, 0.001, 0.002])
    }

    print("Testing Cartesian Impedance Control...")
    impedance_result = force_ctrl.cartesian_impedance_control(desired_pose, current_pose)
    print(f"Required wrench: {impedance_result['wrench']}")

    # Example tactile feedback processing
    print("\nTesting Tactile Feedback Integration...")
    dummy_tactile_data = np.random.rand(20) * 0.5  # Random tactile readings
    tactile_result = tactile_integrator.process_tactile_data('right_hand', dummy_tactile_data)
    print(f"Grasp stability: {tactile_result['grasp_stability']:.2f}")
    print(f"Slip detected: {tactile_result['slip_detected']}")

    print("\nManipulation system ready for complex tasks!")

if __name__ == '__main__':
    main()
```

## Exercises

1. **Grasp Planning**: Implement a grasp planner that can handle different object shapes and generate stable grasps.

2. **Force Control**: Create a force control system that can perform compliant insertion tasks.

3. **Bimanual Coordination**: Design a bimanual manipulation task such as opening a jar or folding clothes.

4. **Tactile Integration**: Implement tactile feedback processing for grasp stability assessment.

5. **Skill Learning**: Create a system that learns manipulation skills from demonstrations.

## Summary

Humanoid manipulation combines complex whole-body coordination with dexterous hand control to enable robots to interact with their environment effectively. Successful manipulation requires integration of grasp planning, force control, tactile feedback, and coordinated motion across multiple degrees of freedom. The challenge lies in coordinating these elements while maintaining balance and achieving the manipulation goal safely. As we continue our exploration of Physical AI, mastering manipulation principles will be essential for creating humanoid robots capable of performing complex tasks in human environments.