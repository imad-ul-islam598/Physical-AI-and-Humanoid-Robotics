---
sidebar_position: 16
---

# Chapter 16: Humanoid Kinematics

## Introduction

Humanoid kinematics is the study of motion in humanoid robots without considering the forces that cause the motion. Understanding both forward and inverse kinematics is essential for controlling humanoid robots, enabling them to reach desired positions, maintain balance, and perform complex movements. This chapter explores the mathematical foundations of humanoid kinematics, implementation of kinematic algorithms, and practical considerations for controlling multi-degree-of-freedom humanoid systems.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the mathematical foundations of forward and inverse kinematics
- Implement forward kinematics for humanoid robot chains
- Solve inverse kinematics problems for humanoid manipulation and locomotion
- Apply Denavit-Hartenberg (DH) parameters to humanoid robot modeling
- Implement kinematic constraints and joint limits for realistic motion

## Key Concepts

- **Forward Kinematics**: Calculating end-effector position from joint angles
- **Inverse Kinematics**: Calculating joint angles to achieve desired end-effector position
- **Denavit-Hartenberg (DH) Parameters**: Standard method for describing robot kinematic chains
- **Jacobian Matrix**: Relating joint velocities to end-effector velocities
- **Kinematic Chains**: Sequential connections of joints and links
- **Degrees of Freedom (DOF)**: Independent movements a robot can perform
- **Workspace**: Volume in space reachable by the robot's end-effector

## Technical Explanation

Humanoid kinematics involves understanding the relationship between joint angles and the position and orientation of various parts of the robot, particularly the hands and feet. Humanoid robots typically have complex kinematic structures with multiple closed chains (when feet are on the ground) and redundant degrees of freedom.

**Forward Kinematics** calculates the position and orientation of the end-effector (hand, foot, etc.) given the joint angles. For a humanoid robot with n joints, forward kinematics transforms joint space coordinates (θ₁, θ₂, ..., θₙ) to Cartesian space coordinates (x, y, z, roll, pitch, yaw).

The forward kinematics solution uses transformation matrices to represent the relationship between consecutive links in a kinematic chain. Each joint-link pair contributes a transformation matrix based on its Denavit-Hartenberg parameters:

```
T = Rz(θ) * Tz(d) * Tx(a) * Rx(α)
```

Where:
- Rz(θ): Rotation around z-axis by angle θ
- Tz(d): Translation along z-axis by distance d
- Tx(a): Translation along x-axis by distance a
- Rx(α): Rotation around x-axis by angle α

For a complete kinematic chain, the overall transformation is the product of all individual transformations:

```
T_total = T₁ * T₂ * ... * Tₙ
```

**Inverse Kinematics** is more complex, as it involves solving for joint angles given a desired end-effector position and orientation. This is typically an underdetermined system for redundant robots (more DOF than necessary), requiring optimization techniques to select among possible solutions.

Common inverse kinematics approaches include:

1. **Analytical Methods**: Closed-form solutions for specific kinematic structures
2. **Numerical Methods**: Iterative approaches like Jacobian-based methods
3. **Optimization-Based Methods**: Formulating as an optimization problem with constraints

The Jacobian matrix is fundamental to numerical inverse kinematics:

```
J = ∂f/∂θ
```

Where f is the forward kinematics function and θ is the joint angle vector. The relationship between joint velocities and end-effector velocities is:

```
v = J * θ̇
```

For inverse kinematics:
```
θ̇ = J⁺ * v
```

Where J⁺ is the pseudoinverse of the Jacobian.

**Humanoid-Specific Considerations**:

Humanoid robots have several unique kinematic challenges:

- **Redundancy**: Multiple solutions exist for reaching the same position
- **Balance**: Kinematic solutions must consider center of mass and stability
- **Closed Kinematic Chains**: When feet are on the ground, the legs form closed chains
- **Multiple Tasks**: Need to satisfy multiple end-effector constraints simultaneously

**Kinematic Constraints** in humanoid robots include:

- Joint angle limits
- Velocity and acceleration limits
- Collision avoidance
- Balance constraints (ZMP - Zero Moment Point)
- Workspace boundaries

## Diagrams written as text descriptions

**Diagram 1: Humanoid Kinematic Structure**
```
        Head
          │
    ┌─────┴─────┐
    │           │
  Left Arm   Right Arm
    │           │
   (7DOF)      (7DOF)
    │           │
   Torso       Torso
    │           │
    └─────┬─────┘
          │
    ┌─────┴─────┐
    │           │
  Left Leg   Right Leg
    │           │
   (6DOF)      (6DOF)
    │           │
   Foot        Foot
```

**Diagram 2: Forward Kinematics Chain (Right Arm Example)**
```
Base Frame → Shoulder Joint → Elbow Joint → Wrist Joint → End-Effector
    │            │              │            │              │
    │ θ1         │ θ2           │ θ3         │ θ4           │
    │            │              │            │              │
   T1(θ1) →   T2(θ2) →      T3(θ3) →    T4(θ4) →    T_end_effector
```

**Diagram 3: Inverse Kinematics Problem**
```
Given: Desired End-Effector Pose (xd, yd, zd, θxd, θyd, θzd)
Find: Joint Angles (θ1, θ2, θ3, θ4, θ5, θ6, θ7)
Such that: ForwardKinematics(θ1, θ2, ..., θ7) ≈ (xd, yd, zd, θxd, θyd, θzd)
```

## Code Examples

Here's an example of implementing forward and inverse kinematics for a humanoid arm:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class HumanoidArmKinematics:
    def __init__(self, arm_type='right'):
        """
        Initialize kinematic model for humanoid arm
        Uses Denavit-Hartenberg parameters for a 7-DOF arm
        """
        self.arm_type = arm_type

        # DH parameters for a typical humanoid arm [a, alpha, d, theta_offset]
        # These would be calibrated for your specific robot
        if arm_type == 'right':
            self.dh_params = [
                [0.0, np.pi/2, 0.2, 0.0],      # Shoulder yaw
                [0.0, -np.pi/2, 0.0, 0.0],     # Shoulder pitch
                [0.0, np.pi/2, 0.25, 0.0],     # Shoulder roll
                [0.0, -np.pi/2, 0.0, 0.0],     # Elbow pitch
                [0.0, np.pi/2, 0.25, 0.0],     # Wrist pitch
                [0.0, -np.pi/2, 0.0, 0.0],     # Wrist roll
                [0.0, 0.0, 0.1, 0.0]           # Wrist yaw
            ]
        else:  # left arm (may have different offsets)
            self.dh_params = [
                [0.0, np.pi/2, 0.2, 0.0],      # Shoulder yaw
                [0.0, -np.pi/2, 0.0, 0.0],     # Shoulder pitch
                [0.0, -np.pi/2, 0.25, 0.0],    # Shoulder roll (different sign for left)
                [0.0, -np.pi/2, 0.0, 0.0],     # Elbow pitch
                [0.0, np.pi/2, 0.25, 0.0],     # Wrist pitch
                [0.0, -np.pi/2, 0.0, 0.0],     # Wrist roll
                [0.0, 0.0, 0.1, 0.0]           # Wrist yaw
            ]

        # Joint limits (in radians)
        self.joint_limits = [
            [-2.0, 2.0],    # Shoulder yaw
            [-2.0, 1.5],    # Shoulder pitch
            [-2.5, 1.0],    # Shoulder roll
            [-2.5, 0.5],    # Elbow pitch
            [-2.0, 2.0],    # Wrist pitch
            [-2.0, 2.0],    # Wrist roll
            [-2.0, 2.0]     # Wrist yaw
        ]

    def dh_transform(self, a, alpha, d, theta):
        """
        Calculate Denavit-Hartenberg transformation matrix
        """
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        st = np.sin(theta)
        ct = np.cos(theta)

        T = np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
        return T

    def forward_kinematics(self, joint_angles):
        """
        Calculate forward kinematics - get end-effector pose from joint angles
        """
        if len(joint_angles) != len(self.dh_params):
            raise ValueError("Number of joint angles must match number of joints")

        # Check joint limits
        for i, (angle, limits) in enumerate(zip(joint_angles, self.joint_limits)):
            if not (limits[0] <= angle <= limits[1]):
                print(f"Warning: Joint {i} angle {angle} out of limits [{limits[0]}, {limits[1]}]")

        # Calculate transformation matrices
        T_total = np.eye(4)  # Identity matrix

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            T_link = self.dh_transform(a, alpha, d, theta)
            T_total = T_total @ T_link

        # Extract position and orientation
        position = T_total[:3, 3]
        rotation_matrix = T_total[:3, :3]

        # Convert rotation matrix to quaternion for easier handling
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # [x, y, z, w]

        return {
            'position': position,
            'orientation_quat': quaternion,
            'orientation_matrix': rotation_matrix,
            'transform_matrix': T_total
        }

    def jacobian(self, joint_angles):
        """
        Calculate geometric Jacobian matrix
        J = [Jv; Jw] where Jv is linear velocity Jacobian and Jw is angular velocity Jacobian
        """
        n = len(joint_angles)
        J = np.zeros((6, n))

        # Get all transformation matrices
        T_current = np.eye(4)
        T_list = [T_current]

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            T_link = self.dh_transform(a, alpha, d, theta)
            T_current = T_current @ T_link
            T_list.append(T_current)

        # End-effector position
        end_pos = T_list[-1][:3, 3]

        # Calculate Jacobian columns
        for i in range(n):
            # Z-axis of joint i in base frame
            z_i = T_list[i][:3, 2]
            # Position of joint i in base frame
            p_i = T_list[i][:3, 3]

            # Linear velocity component
            J[:3, i] = np.cross(z_i, (end_pos - p_i))
            # Angular velocity component
            J[3:, i] = z_i

        return J

    def inverse_kinematics(self, target_pose, initial_angles=None, max_iterations=100, tolerance=1e-4):
        """
        Solve inverse kinematics using Jacobian transpose method
        """
        if initial_angles is None:
            initial_angles = np.zeros(len(self.dh_params))

        current_angles = np.array(initial_angles)

        for iteration in range(max_iterations):
            # Calculate current end-effector pose
            current_pose = self.forward_kinematics(current_angles)
            current_pos = current_pose['position']

            # Calculate error
            pos_error = target_pose['position'] - current_pos
            pos_error_norm = np.linalg.norm(pos_error)

            if pos_error_norm < tolerance:
                print(f"Converged after {iteration} iterations")
                break

            # Calculate Jacobian
            J = self.jacobian(current_angles)

            # Calculate joint angle update using Jacobian transpose
            # This is a simple Jacobian transpose method
            # For better results, use pseudoinverse: delta_theta = J_pinv @ error
            J_transpose = J[:3, :].T  # Only position part
            delta_theta = J_transpose @ pos_error * 0.1  # Learning rate

            # Update joint angles
            current_angles = current_angles + delta_theta

            # Apply joint limits
            for i, limits in enumerate(self.joint_limits):
                current_angles[i] = np.clip(current_angles[i], limits[0], limits[1])

        else:
            print(f"Warning: Did not converge after {max_iterations} iterations")

        return current_angles

    def inverse_kinematics_pinv(self, target_pose, initial_angles=None, max_iterations=100, tolerance=1e-4):
        """
        Solve inverse kinematics using Jacobian pseudoinverse method (more robust)
        """
        if initial_angles is None:
            initial_angles = np.zeros(len(self.dh_params))

        current_angles = np.array(initial_angles)

        for iteration in range(max_iterations):
            # Calculate current end-effector pose
            current_pose = self.forward_kinematics(current_angles)
            current_pos = current_pose['position']

            # Calculate error
            pos_error = target_pose['position'] - current_pos
            pos_error_norm = np.linalg.norm(pos_error)

            if pos_error_norm < tolerance:
                print(f"Converged after {iteration} iterations")
                break

            # Calculate Jacobian
            J = self.jacobian(current_angles)

            # Calculate joint angle update using pseudoinverse
            # Add damping for better numerical stability
            damping = 0.01
            I = np.eye(J.shape[1])
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * I)

            # Calculate desired end-effector velocity
            desired_vel = np.zeros(6)
            desired_vel[:3] = pos_error * 1.0  # Position error

            # Calculate joint velocity
            joint_vel = J_pinv @ desired_vel

            # Update joint angles
            current_angles = current_angles + joint_vel * 0.01  # Small time step

            # Apply joint limits
            for i, limits in enumerate(self.joint_limits):
                current_angles[i] = np.clip(current_angles[i], limits[0], limits[1])

        else:
            print(f"Warning: Did not converge after {max_iterations} iterations")

        return current_angles

# Example usage
def main():
    # Create kinematic model for right arm
    arm_kin = HumanoidArmKinematics(arm_type='right')

    # Example: Calculate forward kinematics
    joint_angles = [0.1, 0.2, 0.0, -0.5, 0.3, 0.1, 0.0]  # Example joint angles
    pose = arm_kin.forward_kinematics(joint_angles)

    print("Forward Kinematics Result:")
    print(f"Position: {pose['position']}")
    print(f"Orientation (quat): {pose['orientation_quat']}")

    # Example: Calculate Jacobian
    J = arm_kin.jacobian(joint_angles)
    print(f"\nJacobian shape: {J.shape}")
    print(f"Jacobian:\n{J}")

    # Example: Inverse kinematics
    target_pose = {
        'position': np.array([0.5, 0.3, 0.2]),  # Target position
        'orientation': np.array([0, 0, 0, 1])   # Target orientation (quat)
    }

    initial_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    solution = arm_kin.inverse_kinematics_pinv(target_pose, initial_angles)

    print(f"\nInverse Kinematics Solution: {solution}")

    # Verify solution
    final_pose = arm_kin.forward_kinematics(solution)
    print(f"Final position: {final_pose['position']}")
    print(f"Target position: {target_pose['position']}")

if __name__ == '__main__':
    main()
```

Example of a more complete humanoid kinematics system with multiple chains:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidKinematics:
    def __init__(self):
        """
        Complete humanoid kinematic model with multiple chains
        """
        # Define kinematic chains
        self.chains = {
            'right_arm': HumanoidArmKinematics(arm_type='right'),
            'left_arm': HumanoidArmKinematics(arm_type='left'),
            'right_leg': self.create_leg_kinematics('right'),
            'left_leg': self.create_leg_kinematics('left')
        }

        # Define base positions (relative to robot's root frame)
        self.chain_bases = {
            'right_arm': np.array([0.0, -0.1, 0.2]),  # Shoulder position
            'left_arm': np.array([0.0, 0.1, 0.2]),   # Shoulder position
            'right_leg': np.array([0.0, -0.05, 0.0]), # Hip position
            'left_leg': np.array([0.0, 0.05, 0.0])   # Hip position
        }

    def create_leg_kinematics(self, leg_type):
        """
        Create kinematic model for leg (simplified 6-DOF leg)
        """
        class LegKinematics:
            def __init__(self, leg_type):
                self.leg_type = leg_type
                # Simplified DH parameters for leg
                self.dh_params = [
                    [0.0, -np.pi/2, 0.0, 0.0],     # Hip yaw
                    [0.0, np.pi/2, 0.0, 0.0],      # Hip roll
                    [0.0, 0.0, 0.4, 0.0],          # Hip pitch (to knee)
                    [0.0, 0.0, 0.4, 0.0],          # Knee pitch (to ankle)
                    [0.0, -np.pi/2, 0.0, 0.0],     # Ankle pitch
                    [0.0, np.pi/2, 0.0, 0.0]       # Ankle roll
                ]

                # Joint limits
                self.joint_limits = [
                    [-0.5, 0.5],    # Hip yaw
                    [-0.5, 0.5],    # Hip roll
                    [-2.0, 0.5],    # Hip pitch
                    [-2.5, 0.0],    # Knee pitch
                    [-0.5, 0.5],    # Ankle pitch
                    [-0.5, 0.5]     # Ankle roll
                ]

            def dh_transform(self, a, alpha, d, theta):
                sa = np.sin(alpha)
                ca = np.cos(alpha)
                st = np.sin(theta)
                ct = np.cos(theta)

                T = np.array([
                    [ct, -st*ca, st*sa, a*ct],
                    [st, ct*ca, -ct*sa, a*st],
                    [0, sa, ca, d],
                    [0, 0, 0, 1]
                ])
                return T

            def forward_kinematics(self, joint_angles):
                if len(joint_angles) != len(self.dh_params):
                    raise ValueError("Wrong number of joint angles")

                T_total = np.eye(4)
                for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
                    theta = joint_angles[i] + theta_offset
                    T_link = self.dh_transform(a, alpha, d, theta)
                    T_total = T_total @ T_link

                position = T_total[:3, 3]
                rotation_matrix = T_total[:3, :3]
                r = R.from_matrix(rotation_matrix)
                quaternion = r.as_quat()

                return {
                    'position': position,
                    'orientation_quat': quaternion,
                    'transform_matrix': T_total
                }

        return LegKinematics(leg_type)

    def calculate_center_of_mass(self, joint_angles_dict, link_masses=None):
        """
        Calculate center of mass of the humanoid given joint angles
        This is a simplified calculation
        """
        if link_masses is None:
            # Default masses (kg)
            link_masses = {
                'torso': 10.0,
                'head': 2.0,
                'right_upper_arm': 1.5,
                'right_lower_arm': 1.0,
                'left_upper_arm': 1.5,
                'left_lower_arm': 1.0,
                'right_thigh': 3.0,
                'right_shin': 2.5,
                'left_thigh': 3.0,
                'left_shin': 2.5
            }

        total_mass = sum(link_masses.values())

        # Simplified CoM calculation
        # In practice, this would require detailed link models and positions
        com = np.zeros(3)

        # Base torso position (simplified)
        com += link_masses['torso'] * np.array([0, 0, 0.5])
        com += link_masses['head'] * np.array([0, 0, 0.8])

        # Add arm contributions based on their positions
        if 'right_arm' in joint_angles_dict:
            right_arm_fk = self.chains['right_arm'].forward_kinematics(joint_angles_dict['right_arm'])
            arm_pos = right_arm_fk['position'] + self.chain_bases['right_arm']
            com += link_masses['right_upper_arm'] * arm_pos * 0.6  # Simplified
            com += link_masses['right_lower_arm'] * arm_pos * 0.4  # Simplified

        if 'left_arm' in joint_angles_dict:
            left_arm_fk = self.chains['left_arm'].forward_kinematics(joint_angles_dict['left_arm'])
            arm_pos = left_arm_fk['position'] + self.chain_bases['left_arm']
            com += link_masses['left_upper_arm'] * arm_pos * 0.6  # Simplified
            com += link_masses['left_lower_arm'] * arm_pos * 0.4  # Simplified

        # Add leg contributions
        if 'right_leg' in joint_angles_dict:
            right_leg_fk = self.chains['right_leg'].forward_kinematics(joint_angles_dict['right_leg'])
            leg_pos = right_leg_fk['position'] + self.chain_bases['right_leg']
            com += link_masses['right_thigh'] * leg_pos * 0.6  # Simplified
            com += link_masses['right_shin'] * leg_pos * 0.4   # Simplified

        if 'left_leg' in joint_angles_dict:
            left_leg_fk = self.chains['left_leg'].forward_kinematics(joint_angles_dict['left_leg'])
            leg_pos = left_leg_fk['position'] + self.chain_bases['left_leg']
            com += link_masses['left_thigh'] * leg_pos * 0.6  # Simplified
            com += link_masses['left_shin'] * leg_pos * 0.4   # Simplified

        com = com / total_mass
        return com

    def check_workspace_feasibility(self, target_pos, chain_name, joint_angles):
        """
        Check if target position is within workspace of the chain
        """
        # Calculate current end-effector position
        current_fk = self.chains[chain_name].forward_kinematics(joint_angles)
        current_pos = current_fk['position']

        # Calculate maximum reach (simplified)
        max_reach = 0
        if chain_name in ['right_arm', 'left_arm']:
            # Arm reach: sum of link lengths
            max_reach = 0.2 + 0.25 + 0.25 + 0.1  # Approximate arm length
        elif chain_name in ['right_leg', 'left_leg']:
            # Leg reach
            max_reach = 0.4 + 0.4  # Approximate leg length

        # Check if target is within reach
        distance = np.linalg.norm(target_pos - current_pos)

        return distance <= max_reach

# Example of kinematic constraints and optimization
class ConstrainedKinematics:
    def __init__(self, humanoid_kin):
        self.humanoid_kin = humanoid_kin

    def solve_multi_task_ik(self, tasks, initial_angles_dict, weights=None):
        """
        Solve inverse kinematics for multiple tasks simultaneously
        tasks: list of tuples (chain_name, target_pose, priority_weight)
        """
        if weights is None:
            weights = [1.0] * len(tasks)

        # This is a simplified approach
        # In practice, you'd use more sophisticated multi-task optimization
        current_angles = initial_angles_dict.copy()

        # Solve each task sequentially with decreasing priority
        for i, (chain_name, target_pose, priority) in enumerate(tasks):
            # Get current chain angles
            chain_angles = current_angles[chain_name]

            # Solve IK for this chain
            if chain_name in ['right_arm', 'left_arm']:
                solution = self.humanoid_kin.chains[chain_name].inverse_kinematics_pinv(
                    target_pose, chain_angles, tolerance=1e-3
                )
            elif chain_name in ['right_leg', 'left_leg']:
                # For legs, you might want to consider balance constraints
                solution = self.humanoid_kin.chains[chain_name].inverse_kinematics(
                    target_pose, chain_angles, tolerance=1e-3
                )

            current_angles[chain_name] = solution

        return current_angles

    def enforce_balance_constraints(self, joint_angles_dict, support_polygons):
        """
        Enforce balance constraints during motion
        """
        # Calculate current CoM
        current_com = self.humanoid_kin.calculate_center_of_mass(joint_angles_dict)

        # Check if CoM is within support polygon
        # This is a simplified 2D check (x, y plane)
        com_2d = current_com[:2]

        # For each support polygon, check if CoM is inside
        valid_balance = False
        for polygon in support_polygons:
            if self.is_point_in_polygon(com_2d, polygon):
                valid_balance = True
                break

        if not valid_balance:
            print("Warning: Balance constraint violated!")
            # In practice, you'd adjust joint angles to restore balance

        return valid_balance

    def is_point_in_polygon(self, point, polygon):
        """
        Simple point-in-polygon test using ray casting
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
```

## Exercises

1. **Forward Kinematics**: Implement forward kinematics for a complete humanoid model with 2 arms and 2 legs.

2. **Inverse Kinematics**: Create an inverse kinematics solver that can handle multiple end-effectors simultaneously.

3. **DH Parameters**: Derive Denavit-Hartenberg parameters for a specific humanoid robot model.

4. **Workspace Analysis**: Analyze and visualize the workspace of different humanoid limbs.

5. **Constraint Handling**: Implement joint limit and collision avoidance constraints in kinematic solutions.

## Summary

Humanoid kinematics forms the mathematical foundation for controlling the complex movements of humanoid robots. Understanding both forward and inverse kinematics is essential for enabling robots to reach desired positions, maintain balance, and perform coordinated movements. The redundancy in humanoid systems provides flexibility but also requires sophisticated algorithms to select optimal solutions. As we continue our exploration of Physical AI, kinematic understanding will be crucial for developing controllers that enable humanoid robots to interact effectively with their environment while maintaining stability and safety.