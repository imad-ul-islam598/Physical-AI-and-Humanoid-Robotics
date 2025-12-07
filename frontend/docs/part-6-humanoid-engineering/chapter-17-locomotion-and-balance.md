---
sidebar_position: 17
---

# Chapter 17: Locomotion and Balance

## Introduction

Locomotion and balance are fundamental capabilities for humanoid robots, enabling them to move through environments while maintaining stability. Unlike wheeled or tracked robots, humanoid robots must solve the complex problem of bipedal locomotion, which requires precise control of the center of mass, coordinated joint movements, and dynamic balance strategies. This chapter explores the principles of humanoid locomotion, balance control methods, and the Zero Moment Point (ZMP) theory that forms the foundation for stable walking.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the principles of bipedal locomotion and balance control
- Implement Zero Moment Point (ZMP) based balance control strategies
- Design walking pattern generators for humanoid robots
- Apply feedback control for dynamic balance maintenance
- Analyze and mitigate stability challenges in humanoid locomotion

## Key Concepts

- **Bipedal Locomotion**: Two-legged walking motion similar to humans
- **Zero Moment Point (ZMP)**: Point where the net moment of ground reaction forces is zero
- **Center of Mass (CoM)**: Point where the robot's mass is concentrated
- **Support Polygon**: Area defined by contact points with the ground
- **Dynamic Balance**: Balance maintained through active control during motion
- **Static Balance**: Balance maintained when stationary
- **Walking Gaits**: Patterns of leg movement for locomotion
- **Capture Point**: Point where the robot can come to rest

## Technical Explanation

Humanoid locomotion and balance control is one of the most challenging problems in robotics, requiring the coordination of multiple systems to achieve stable, efficient movement. The fundamental challenge lies in the fact that bipedal walking is inherently unstable - unlike wheeled robots that have continuous support, humanoid robots have alternating periods of single and double support, creating dynamic stability challenges.

**Zero Moment Point (ZMP) Theory** is the cornerstone of most humanoid balance control strategies. The ZMP is defined as the point on the ground where the net moment of the ground reaction forces is zero. For a robot to maintain balance, the ZMP must remain within the support polygon defined by the feet in contact with the ground.

Mathematically, the ZMP position can be calculated as:

```
ZMP_x = (Σ(F_iz * x_i) - Σ(M_ix)) / Σ(F_iz)
ZMP_y = (Σ(F_iz * y_i) - Σ(M_iy)) / Σ(F_iz)
```

Where:
- F_iz: Normal force component at contact point i
- (x_i, y_i): Position of contact point i
- M_ix, M_iy: Moment components at contact point i

For the center of mass (CoM) of a humanoid robot, the relationship between CoM position (x, y, z) and ZMP (x_zmp, y_zmp) under the linear inverted pendulum model is:

```
ẍ = g/h * (x - x_zmp)
ÿ = g/h * (y - y_zmp)
```

Where:
- g: gravitational acceleration
- h: height of CoM above ground
- (x, y): CoM coordinates

**Walking Pattern Generation** involves creating trajectories for the feet, CoM, and other body parts that result in stable locomotion. Common approaches include:

1. **Preview Control**: Using future ZMP references to generate stable walking patterns
2. **Linear Inverted Pendulum Model (LIPM)**: Simplifying the robot to a point mass at CoM height
3. **Cart-Table Model**: Extension of LIPM with variable height
4. **Footstep Planning**: Determining where and when to place feet

**Balance Control Strategies** include:

1. **Ankle Strategy**: Using ankle joint torques to maintain balance (for small perturbations)
2. **Hip Strategy**: Using hip joint torques for larger perturbations
3. **Stepping Strategy**: Taking a step to expand the support polygon
4. **Arm Swinging**: Using arm movements to counteract balance disturbances

**Dynamic Balance Control** involves feedback control systems that continuously adjust the robot's posture and movement to maintain stability. Common control approaches include:

- **PID Control**: Proportional-Integral-Derivative control of joint positions
- **Model Predictive Control (MPC)**: Predictive control considering future states
- **Linear Quadratic Regulator (LQR)**: Optimal control minimizing a quadratic cost
- **Adaptive Control**: Adjusting control parameters based on changing conditions

**Walking Gaits** for humanoid robots typically follow a cycle:
1. **Single Support Phase**: One foot in contact with ground
2. **Double Support Phase**: Both feet in contact (brief transition)
3. **Swing Phase**: Non-stance leg moves forward
4. **Heel Strike**: Swing foot contacts ground
5. **Toe Off**: Stance foot lifts off

The timing and coordination of these phases determine the walking speed, stability, and energy efficiency of the robot.

## Diagrams written as text descriptions

**Diagram 1: ZMP and Support Polygon**
```
    Robot View from Above
    ┌─────────────────┐
    │     CoM         │ ← Center of Mass
    │      ●          │
    └─────────────────┘
         │
         ▼
   Ground Level
   ┌─────────────────┐
   │  Support        │
   │  Polygon        │ ← Area where feet contact ground
   │  [=======]      │   (shaded area)
   │  [=======]      │
   └─────────────────┘
         ↑
       ZMP (must stay within polygon for stability)

   If ZMP moves outside polygon → Robot falls
```

**Diagram 2: Walking Phase Cycle**
```
Time →
│
│    Single    │  Double   │   Single    │  Double   │
│   Support    │  Support  │   Support   │  Support  │
│    Phase     │   Phase   │    Phase    │   Phase   │
│              │           │             │           │
│   [Stance]   │  [Both]   │  [Swing]    │  [Both]   │
│    Foot      │   Feet    │   Foot      │   Feet    │
│     ●        │   ● ●     │     ○       │   ● ●     │
│              │           │             │           │
│              │           │             │           │
│              │           │             │           │
└─────────────────────────────────────────────────────
   Left Stance   Double   Right Stance   Double
                Support               Support
```

**Diagram 3: Inverted Pendulum Model**
```
CoM (Center of Mass)
   ●
   │ \
   │  \
   │   \ h (CoM height)
   │    \
   │     \
   ▼      \
Ground ----*---- ← ZMP (Zero Moment Point)
         / | \
        /  |  \
       /   |   \
      /    |    \
     /     |     \
    ○      ●      ○
   Left   ZMP   Right
  Foot   Pos.   Foot
```

## Code Examples

Here's an example of ZMP-based balance control for a humanoid robot:

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class ZMPBasedBalancer:
    def __init__(self, robot_height=0.8, gravity=9.81):
        """
        Initialize ZMP-based balance controller
        robot_height: height of CoM above ground (m)
        gravity: gravitational acceleration (m/s²)
        """
        self.height = robot_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / robot_height)  # Natural frequency of inverted pendulum

        # State variables
        self.com_x = 0.0
        self.com_y = 0.0
        self.com_dx = 0.0
        self.com_dy = 0.0

        # Support polygon (simplified as rectangle)
        self.support_polygon = {
            'min_x': -0.1,  # 10cm from center
            'max_x': 0.1,
            'min_y': -0.05, # 5cm from center
            'max_y': 0.05
        }

        # Control gains
        self.kp = 10.0  # Proportional gain
        self.kd = 2.0 * np.sqrt(self.kp)  # Derivative gain (critically damped)

    def update_state(self, com_pos, com_vel):
        """
        Update CoM position and velocity
        com_pos: [x, y] position of center of mass
        com_vel: [dx, dy] velocity of center of mass
        """
        self.com_x, self.com_y = com_pos
        self.com_dx, self.com_dy = com_vel

    def calculate_zmp(self, com_pos, com_acc):
        """
        Calculate ZMP from CoM position and acceleration
        com_pos: [x, y] CoM position
        com_acc: [ax, ay] CoM acceleration
        """
        x_com, y_com = com_pos
        ax_com, ay_com = com_acc

        # ZMP calculation based on inverted pendulum model
        zmp_x = x_com - (self.height / self.gravity) * ax_com
        zmp_y = y_com - (self.height / self.gravity) * ay_com

        return np.array([zmp_x, zmp_y])

    def is_stable(self, zmp):
        """
        Check if ZMP is within support polygon
        """
        zmp_x, zmp_y = zmp

        return (self.support_polygon['min_x'] <= zmp_x <= self.support_polygon['max_x'] and
                self.support_polygon['min_y'] <= zmp_y <= self.support_polygon['max_y'])

    def balance_control(self, desired_zmp, current_zmp):
        """
        Generate corrective CoM trajectory to maintain balance
        """
        # Calculate error
        error = desired_zmp - current_zmp

        # Simple PD control to determine desired CoM acceleration
        desired_com_acc_x = self.kp * error[0] + self.kd * (-self.com_dx)  # Use negative velocity as feedback
        desired_com_acc_y = self.kp * error[1] + self.kd * (-self.com_dy)

        # Convert to ZMP acceleration command
        zmp_acc_x = self.gravity / self.height * error[0]
        zmp_acc_y = self.gravity / self.height * error[1]

        return np.array([desired_com_acc_x, desired_com_acc_y])

    def generate_ankle_strategy(self, zmp_error):
        """
        Generate ankle torques for small balance corrections
        """
        # Simple ankle strategy: use ankle torques to move ZMP
        ankle_gain = 50.0  # Nm/rad
        max_ankle_angle = 0.1  # 5.7 degrees

        # Calculate required ankle angle to move ZMP
        ankle_roll = np.clip(zmp_error[1] * ankle_gain, -max_ankle_angle, max_ankle_angle)
        ankle_pitch = np.clip(zmp_error[0] * ankle_gain, -max_ankle_angle, max_ankle_angle)

        return np.array([ankle_roll, ankle_pitch])

class WalkingPatternGenerator:
    def __init__(self, step_length=0.3, step_width=0.2, step_height=0.05,
                 single_support_time=0.7, double_support_time=0.1):
        """
        Generate walking patterns for humanoid robot
        """
        self.step_length = step_length  # Forward step distance (m)
        self.step_width = step_width    # Lateral foot distance (m)
        self.step_height = step_height  # Maximum foot height (m)
        self.T_ss = single_support_time  # Single support time (s)
        self.T_ds = double_support_time  # Double support time (s)
        self.T_step = 2 * self.T_ss + 2 * self.T_ds  # Total step time (s)

        # Initialize foot positions
        self.left_foot_pos = np.array([0.0, self.step_width/2, 0.0])
        self.right_foot_pos = np.array([0.0, -self.step_width/2, 0.0])
        self.current_support_foot = 'left'  # Start with left foot support

    def generate_foot_trajectory(self, start_pos, end_pos, height, t, total_time):
        """
        Generate smooth foot trajectory from start to end position
        """
        # Use 5th order polynomial for smooth motion
        # s(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        # with boundary conditions: s(0)=0, s'(0)=0, s''(0)=0, s(1)=1, s'(1)=0, s''(1)=0

        if total_time <= 0:
            return end_pos

        t_norm = t / total_time
        t_norm = np.clip(t_norm, 0, 1)

        # 5th order polynomial coefficients for smooth interpolation
        s = 10*t_norm**3 - 15*t_norm**4 + 6*t_norm**5

        # Calculate horizontal position
        pos_xy = start_pos[:2] + s * (end_pos[:2] - start_pos[:2])

        # Calculate vertical position (parabolic trajectory)
        if t_norm < 0.5:
            # Going up
            s_z = 4 * t_norm**2  # Parabolic rise
        else:
            # Going down
            s_z = 1 - 4 * (t_norm - 1)**2  # Parabolic fall

        pos_z = start_pos[2] + s_z * height

        return np.array([pos_xy[0], pos_xy[1], pos_z])

    def get_current_foot_positions(self, time):
        """
        Get current foot positions based on walking cycle
        """
        # Calculate phase in walking cycle
        cycle_time = time % self.T_step
        phase = cycle_time / self.T_step

        # Determine current phase of walking cycle
        if phase < self.T_ss / self.T_step:
            # Left foot support, right foot swing forward
            left_pos = self.left_foot_pos
            right_pos = self.generate_foot_trajectory(
                self.right_foot_pos,
                [self.right_foot_pos[0] + self.step_length, self.right_foot_pos[1], 0],
                self.step_height,
                cycle_time,
                self.T_ss
            )
        elif phase < (self.T_ss + self.T_ds) / self.T_step:
            # Double support phase
            left_pos = self.left_foot_pos
            right_pos = self.right_foot_pos
        elif phase < (2*self.T_ss + self.T_ds) / self.T_step:
            # Right foot support, left foot swing forward
            right_pos = self.right_foot_pos
            left_pos = self.generate_foot_trajectory(
                self.left_foot_pos,
                [self.left_foot_pos[0] + self.step_length, self.left_foot_pos[1], 0],
                self.step_height,
                cycle_time - self.T_ss - self.T_ds,
                self.T_ss
            )
        else:
            # Double support phase
            left_pos = self.left_foot_pos
            right_pos = self.right_foot_pos

        return left_pos, right_pos

    def update_support_foot(self, time):
        """
        Update which foot is in support based on walking cycle
        """
        cycle_time = time % self.T_step
        phase = cycle_time / self.T_step

        if phase < self.T_ss / self.T_step:
            self.current_support_foot = 'left'
        elif phase < (self.T_ss + self.T_ds) / self.T_step:
            # Double support - keep previous support foot
            pass
        elif phase < (2*self.T_ss + self.T_ds) / self.T_step:
            self.current_support_foot = 'right'
        else:
            # Double support - keep previous support foot
            pass

class HumanoidWalkingController:
    def __init__(self):
        self.balancer = ZMPBasedBalancer()
        self.pattern_gen = WalkingPatternGenerator()
        self.time = 0.0

    def step(self, dt):
        """
        Perform one control step
        """
        # Update walking pattern generator
        self.pattern_gen.update_support_foot(self.time)
        left_foot, right_foot = self.pattern_gen.get_current_foot_positions(self.time)

        # Calculate support polygon based on foot positions
        self.update_support_polygon(left_foot, right_foot)

        # Get current CoM state (in real system, this would come from sensors)
        current_com_pos = self.estimate_com_position(left_foot, right_foot)
        current_com_vel = self.estimate_com_velocity()

        # Update balancer state
        self.balancer.update_state(current_com_pos[:2], current_com_vel[:2])

        # Calculate current ZMP
        current_zmp = self.calculate_current_zmp(current_com_pos, current_com_vel)

        # Determine desired ZMP (should be in middle of support polygon for stability)
        desired_zmp = self.get_desired_zmp()

        # Check stability
        is_stable = self.balancer.is_stable(current_zmp)

        if not is_stable:
            # Generate corrective actions
            corrective_acc = self.balancer.balance_control(desired_zmp, current_zmp)
            ankle_angles = self.balancer.generate_ankle_strategy(desired_zmp - current_zmp)

            print(f"Balance correction needed! ZMP error: {desired_zmp - current_zmp}")

        # Update time
        self.time += dt

    def update_support_polygon(self, left_foot, right_foot):
        """
        Update support polygon based on foot positions
        """
        # Calculate support polygon from foot positions
        min_x = min(left_foot[0], right_foot[0])
        max_x = max(left_foot[0], right_foot[0])
        min_y = min(left_foot[1], right_foot[1])
        max_y = max(left_foot[1], right_foot[1])

        # Add small margin for safety
        margin = 0.02
        self.balancer.support_polygon = {
            'min_x': min_x - margin,
            'max_x': max_x + margin,
            'min_y': min_y - margin,
            'max_y': max_y + margin
        }

    def estimate_com_position(self, left_foot, right_foot):
        """
        Estimate CoM position (simplified)
        """
        # For now, assume CoM is at fixed height above midpoint of feet
        foot_midpoint = (left_foot + right_foot) / 2
        com_height = 0.8  # Fixed CoM height
        return np.array([foot_midpoint[0], foot_midpoint[1], com_height])

    def estimate_com_velocity(self):
        """
        Estimate CoM velocity (simplified)
        """
        # For now, return zero velocity
        # In real system, this would come from IMU or state estimation
        return np.array([0.1, 0.0, 0.0])  # Small forward velocity

    def calculate_current_zmp(self, com_pos, com_vel):
        """
        Calculate current ZMP based on CoM state
        """
        # This would involve dynamics calculations
        # For now, return a simplified estimate
        return np.array([com_pos[0] - 0.01, com_pos[1]])  # Small offset

    def get_desired_zmp(self):
        """
        Get desired ZMP (center of support polygon for stability)
        """
        poly = self.balancer.support_polygon
        return np.array([
            (poly['min_x'] + poly['max_x']) / 2,
            (poly['min_y'] + poly['max_y']) / 2
        ])

# Example usage and simulation
def simulate_walking():
    controller = HumanoidWalkingController()

    # Simulation parameters
    dt = 0.01  # 100 Hz control loop
    simulation_time = 10.0  # 10 seconds

    # Store data for plotting
    times = []
    com_x_data = []
    com_y_data = []
    zmp_x_data = []
    zmp_y_data = []

    for t in np.arange(0, simulation_time, dt):
        controller.step(dt)

        # Store data (in real system, you'd get this from the controller)
        times.append(t)
        com_x_data.append(controller.estimate_com_position(
            *controller.pattern_gen.get_current_foot_positions(t))[0])
        com_y_data.append(controller.estimate_com_position(
            *controller.pattern_gen.get_current_foot_positions(t))[1])

        # Simplified ZMP data
        zmp = controller.get_desired_zmp()
        zmp_x_data.append(zmp[0])
        zmp_y_data.append(zmp[1])

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(times, com_x_data, label='CoM X')
    plt.plot(times, zmp_x_data, label='ZMP X', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('CoM and ZMP X Position')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(times, com_y_data, label='CoM Y')
    plt.plot(times, zmp_y_data, label='ZMP Y', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('CoM and ZMP Y Position')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(com_x_data, com_y_data, label='CoM Trajectory')
    plt.plot(zmp_x_data, zmp_y_data, label='ZMP Trajectory', linestyle='--')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('CoM vs ZMP Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(2, 2, 4)
    # Show support polygon over time
    plt.plot(times, [0.1]*len(times), label='Support Polygon X limits', color='red')
    plt.plot(times, [-0.1]*len(times), color='red')
    plt.fill_between(times, -0.1, 0.1, alpha=0.3, color='red', label='Support Region')
    plt.plot(times, zmp_x_data, label='ZMP X Position')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('ZMP vs Support Polygon')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    simulate_walking()
```

Example of a more advanced balance control system with Capture Point:

```python
import numpy as np
from scipy.integrate import odeint

class AdvancedBalanceController:
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

        # State: [x, y, dx, dy] where (x,y) is CoM position and (dx,dy) is velocity
        self.state = np.zeros(4)

        # Control parameters
        self.control_horizon = 2.0  # seconds to plan ahead
        self.dt = 0.01  # control timestep

    def capture_point(self, com_pos, com_vel):
        """
        Calculate capture point: where the robot needs to step to come to rest
        Capture Point = CoM position + CoM velocity / omega
        """
        cp_x = com_pos[0] + com_vel[0] / self.omega
        cp_y = com_pos[1] + com_vel[1] / self.omega
        return np.array([cp_x, cp_y])

    def linear_inverted_pendulum_dynamics(self, state, t, zmp_ref):
        """
        Dynamics of linear inverted pendulum model
        dx/dt = v
        dv/dt = omega^2 * (com - zmp)
        """
        x, y, vx, vy = state
        zmp_x, zmp_y = zmp_ref

        dstate_dt = [
            vx,  # dx/dt
            vy,  # dy/dt
            self.omega**2 * (x - zmp_x),  # dvx/dt
            self.omega**2 * (y - zmp_y)   # dvy/dt
        ]
        return dstate_dt

    def plan_step_location(self, current_com, current_vel, support_polygon):
        """
        Plan where to step based on capture point
        """
        cp = self.capture_point(current_com, current_vel)

        # Check if capture point is outside support polygon
        if (cp[0] < support_polygon['min_x'] or cp[0] > support_polygon['max_x'] or
            cp[1] < support_polygon['min_y'] or cp[1] > support_polygon['max_y']):

            # Need to step toward capture point
            # For now, step directly to capture point (or closest safe point)
            step_x = max(support_polygon['min_x'], min(support_polygon['max_x'], cp[0]))
            step_y = max(support_polygon['min_y'], min(support_polygon['max_y'], cp[1]))

            return np.array([step_x, step_y])

        return None  # No step needed

    def model_predictive_balance_control(self, current_state, reference_zmp_trajectory,
                                       prediction_horizon=1.0):
        """
        Model Predictive Control for balance
        """
        n_steps = int(prediction_horizon / self.dt)

        # Predict future states based on current control
        predicted_states = []
        current_pred_state = current_state.copy()

        for i in range(n_steps):
            # Simple Euler integration
            derivatives = self.linear_inverted_pendulum_dynamics(
                current_pred_state, i*self.dt, reference_zmp_trajectory[i % len(reference_zmp_trajectory)])

            current_pred_state += np.array(derivatives) * self.dt
            predicted_states.append(current_pred_state.copy())

        # Calculate cost based on deviation from reference
        total_cost = 0
        for i, state in enumerate(predicted_states):
            # Cost function: penalize ZMP deviation and CoM velocity
            com_pos = state[:2]
            com_vel = state[2:]
            ref_zmp = reference_zmp_trajectory[i % len(reference_zmp_trajectory)]

            # Calculate what ZMP would be for this state
            current_zmp = com_pos - com_vel / (self.omega**2)

            zmp_error = np.linalg.norm(current_zmp - ref_zmp)
            vel_penalty = 0.1 * np.linalg.norm(com_vel)

            total_cost += zmp_error + vel_penalty

        return total_cost

    def generate_balance_pattern(self, start_com, start_vel, duration, dt=0.01):
        """
        Generate a balance pattern to maintain stability
        """
        times = np.arange(0, duration, dt)

        # For balance, we want to keep ZMP near center of support polygon
        # This creates a reference trajectory for ZMP
        zmp_refs = []
        com_trajectory = []
        vel_trajectory = []

        current_state = np.array([start_com[0], start_com[1], start_vel[0], start_vel[1]])

        for t in times:
            # Desired ZMP (keeping it at center for balance)
            desired_zmp = np.array([0.0, 0.0])  # Center of support polygon
            zmp_refs.append(desired_zmp)

            # Integrate dynamics
            derivatives = self.linear_inverted_pendulum_dynamics(
                current_state, t, desired_zmp)

            current_state += np.array(derivatives) * dt

            com_trajectory.append(current_state[:2].copy())
            vel_trajectory.append(current_state[2:].copy())

        return {
            'times': times,
            'zmp_refs': np.array(zmp_refs),
            'com_trajectory': np.array(com_trajectory),
            'vel_trajectory': np.array(vel_trajectory)
        }

class WalkingGaitController:
    def __init__(self, step_length=0.3, step_width=0.2, step_height=0.05):
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height

        # Walking state
        self.phase = 0  # 0-1, where 0 is start of step cycle
        self.cycle_time = 0.0
        self.gait_period = 1.0  # seconds per step cycle

        # Foot positions
        self.left_foot = np.array([0.0, self.step_width/2, 0.0])
        self.right_foot = np.array([-0.1, -self.step_width/2, 0.0])  # Start with right foot back

        # Support polygon changes as feet move
        self.support_polygon = self.calculate_support_polygon()

    def calculate_support_polygon(self):
        """
        Calculate support polygon based on foot positions
        """
        min_x = min(self.left_foot[0], self.right_foot[0])
        max_x = max(self.left_foot[0], self.right_foot[0])
        min_y = min(self.left_foot[1], self.right_foot[1])
        max_y = max(self.left_foot[1], self.right_foot[1])

        # Add small margin
        margin = 0.05
        return {
            'min_x': min_x - margin,
            'max_x': max_x + margin,
            'min_y': min_y - margin,
            'max_y': max_y + margin
        }

    def update_gait(self, dt):
        """
        Update gait phase and foot positions
        """
        self.cycle_time += dt
        self.phase = (self.cycle_time % self.gait_period) / self.gait_period

        # Update foot positions based on phase
        self.update_foot_positions()

        # Update support polygon
        self.support_polygon = self.calculate_support_polygon()

    def update_foot_positions(self):
        """
        Update foot positions based on gait phase
        """
        # Simplified walking gait
        if self.phase < 0.5:
            # Left foot swings forward, right foot supports
            # Left foot trajectory
            swing_progress = self.phase * 2  # 0 to 1 during swing
            self.left_foot[0] = -0.1 + self.step_length * swing_progress  # Move forward
            self.left_foot[2] = self.step_height * np.sin(np.pi * swing_progress)  # Vertical motion

            # Right foot stays in place (support phase)
        else:
            # Right foot swings forward, left foot supports
            # Right foot trajectory
            swing_progress = (self.phase - 0.5) * 2  # 0 to 1 during swing
            self.right_foot[0] = 0.0 + self.step_length * swing_progress  # Move forward
            self.right_foot[2] = self.step_height * np.sin(np.pi * swing_progress)  # Vertical motion

            # Left foot stays in place (support phase)

def main():
    # Create controllers
    balance_ctrl = AdvancedBalanceController()
    gait_ctrl = WalkingGaitController()

    # Simulation parameters
    dt = 0.01
    simulation_time = 5.0

    print("Starting humanoid balance and walking simulation...")

    for t in np.arange(0, simulation_time, dt):
        # Update gait
        gait_ctrl.update_gait(dt)

        # Get current CoM estimate (simplified)
        # In real system, this would come from state estimation
        current_com = np.array([0.0, 0.0])
        current_vel = np.array([0.0, 0.0])

        # Calculate capture point
        cp = balance_ctrl.capture_point(current_com, current_vel)

        # Check if step is needed
        step_location = balance_ctrl.plan_step_location(
            current_com, current_vel, gait_ctrl.support_polygon)

        if step_location is not None:
            print(f"Step needed at: {step_location}, Capture point: {cp}")

        # Generate balance pattern
        balance_pattern = balance_ctrl.generate_balance_pattern(
            current_com, current_vel, 1.0, dt)

        # Print status periodically
        if int(t / 0.5) != int((t - dt) / 0.5):
            print(f"Time: {t:.2f}s, Phase: {gait_ctrl.phase:.2f}, "
                  f"Support polygon: {gait_ctrl.support_polygon}")

    print("Simulation completed.")

if __name__ == '__main__':
    main()
```

## Exercises

1. **ZMP Control**: Implement a ZMP-based balance controller and test it with perturbations.

2. **Walking Gait**: Create a walking pattern generator that can handle different walking speeds and step lengths.

3. **Capture Point**: Implement capture point-based stepping strategy for dynamic balance recovery.

4. **Gait Optimization**: Optimize walking parameters (step length, width, timing) for energy efficiency.

5. **Disturbance Rejection**: Test the balance system with external disturbances and measure recovery performance.

## Summary

Locomotion and balance control are critical capabilities for humanoid robots, requiring sophisticated control algorithms to maintain stability during dynamic movement. The Zero Moment Point (ZMP) theory provides a solid foundation for balance control, while walking pattern generation enables coordinated leg movements. Successful implementation requires integration of multiple control strategies including ankle, hip, and stepping responses. As we continue our exploration of Physical AI, mastering these balance and locomotion principles will be essential for creating humanoid robots that can navigate real-world environments safely and effectively.