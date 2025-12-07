---
sidebar_position: 8
---

# Chapter 8: Visualizing in Unity

## Introduction

Unity has emerged as a powerful platform for creating high-fidelity visualizations and simulations for robotics applications. While Gazebo provides excellent physics simulation capabilities, Unity offers superior graphics rendering, user interaction features, and cross-platform deployment options. This chapter explores how to leverage Unity for visualizing and simulating humanoid robots, creating immersive environments for human-robot interaction, and building user interfaces for robot control and monitoring.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the advantages and use cases for Unity in robotics visualization
- Import and configure robot models in Unity for simulation
- Create interactive scenes for robot visualization and control
- Implement basic human-robot interaction interfaces in Unity
- Connect Unity simulations with ROS 2 for real-time data exchange

## Key Concepts

- **Unity Robotics**: Integration of robotics simulation and visualization in Unity
- **Robot Model Import**: Techniques for bringing URDF/Xacro models into Unity
- **Human-Robot Interaction (HRI)**: Interfaces for human-robot communication and control
- **ROS# (ROS Sharp)**: Unity package for ROS communication
- **Physics Simulation**: Unity's built-in physics engine for robotics applications
- **XR Integration**: Virtual and augmented reality applications for robotics

## Technical Explanation

Unity provides a powerful real-time 3D development platform that has found significant applications in robotics for visualization, simulation, and human-robot interaction. Unlike traditional robotics simulators focused primarily on physics accuracy, Unity excels in creating visually compelling, interactive, and immersive environments.

The Unity robotics ecosystem includes several key components:

1. **Unity Robotics Hub**: A collection of packages and tools specifically designed for robotics applications in Unity, including ROS# for communication and various simulation tools.

2. **ROS# (ROS Sharp)**: A Unity package that enables communication between Unity and ROS 2. It provides publishers, subscribers, services, and actions that mirror the ROS 2 communication patterns within the Unity environment.

3. **Robot Model Import**: Unity can import robot models from URDF files through conversion tools or by importing COLLADA/OBJ files exported from CAD software. The Unity Robotics package includes utilities for converting URDF to Unity-compatible formats.

4. **Physics Engine**: Unity's built-in physics engine (NVIDIA PhysX) provides collision detection, rigid body dynamics, and joint constraints suitable for basic robotics simulation, though it may not be as accurate as specialized robotics physics engines.

5. **XR Capabilities**: Unity's support for virtual and augmented reality platforms makes it ideal for immersive robot teleoperation and visualization experiences.

To import robot models into Unity:
- Export the robot model from CAD software as COLLADA (.dae) or OBJ format
- Import into Unity and set up materials and textures
- Create colliders for physics interactions
- Set up joints using Unity's built-in joint components
- Add scripts for controlling the robot's behavior

Unity's advantages for robotics visualization include:
- High-quality graphics rendering with realistic lighting and materials
- Flexible user interface system for creating control panels
- Cross-platform deployment to desktop, mobile, and XR devices
- Rich ecosystem of assets and tools
- Scripting capabilities in C# for custom behaviors

However, Unity is typically not used for detailed physics simulation in robotics due to the specialized requirements of robotic systems. Instead, it's often used for visualization, user interfaces, and human-robot interaction.

## Diagrams written as text descriptions

**Diagram 1: Unity Robotics Architecture**
```
Unity Environment
├── Robot Models (3D Assets)
├── Physics Engine (PhysX)
├── Visual Components
│   ├── Cameras
│   ├── Lighting
│   └── Materials
├── ROS# Communication Layer
│   ├── Publishers
│   ├── Subscribers
│   ├── Services
│   └── Actions
├── UI System
│   ├── Control Panels
│   ├── Status Displays
│   └── Data Visualization
└── XR Integration (Optional)
    ├── VR Headsets
    ├── AR Devices
    └── Haptic Feedback
```

**Diagram 2: Robot Model Import Pipeline**
```
CAD Model → URDF → COLLADA/OBJ → Unity Import → Robot Setup
     ↓           ↓         ↓            ↓           ↓
   Solid    Robot      3D Mesh    Materials    Joints &
   Model   Structure   Geometry    & Textures   Scripts
```

**Diagram 3: ROS# Communication Flow**
```
ROS 2 Network ↔ ROS# Package ↔ Unity Scripts ↔ Robot Behavior
     ↑              ↑              ↑            ↑
State Topics   Publishers/    Game Objects   Physics/
Commands       Subscribers    UI Elements    Animation
```

## Code Examples

Here's an example of a Unity script that interfaces with ROS 2 using ROS#:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Geometry;

public class UnityRobotController : MonoBehaviour
{
    private RosSocket rosSocket;
    private string robotName = "my_robot";

    [Header("ROS Connection")]
    public string rosBridgeServerUrl = "ws://192.168.1.100:9090";

    [Header("Robot Components")]
    public Transform baseLink;
    public Transform[] jointLinks; // Array of joint transforms

    [Header("ROS Topics")]
    public string jointStateTopic = "/joint_states";
    public string cmdVelTopic = "/cmd_vel";

    private float[] jointPositions;
    private string[] jointNames;

    void Start()
    {
        // Initialize ROS connection
        WebSocketNativeClient webSocket = new WebSocketNativeClient(WebSocketProtocol.WebSocketSharp, rosBridgeServerUrl);
        rosSocket = new RosSocket(webSocket);

        // Subscribe to joint states
        rosSocket.Subscribe<sensor_msgs.JointState>(jointStateTopic, JointStateCallback);

        // Initialize joint arrays
        jointPositions = new float[jointLinks.Length];
        jointNames = new string[jointLinks.Length];

        // Set up joint names (these should match your URDF)
        for (int i = 0; i < jointLinks.Length; i++)
        {
            jointNames[i] = "joint_" + i; // Replace with actual joint names
        }
    }

    void JointStateCallback(sensor_msgs.JointState jointState)
    {
        // Update joint positions from ROS message
        for (int i = 0; i < jointNames.Length; i++)
        {
            for (int j = 0; j < jointState.name.Length; j++)
            {
                if (jointState.name[j] == jointNames[i])
                {
                    jointPositions[i] = (float)jointState.position[j];
                    break;
                }
            }
        }

        // Apply joint positions to Unity transforms
        UpdateRobotJoints();
    }

    void UpdateRobotJoints()
    {
        // Update each joint's rotation based on received positions
        for (int i = 0; i < jointLinks.Length; i++)
        {
            // Apply rotation (adjust axis based on joint type)
            jointLinks[i].localRotation = Quaternion.Euler(0, jointPositions[i] * Mathf.Rad2Deg, 0);
        }
    }

    // Method to send velocity commands
    public void SendVelocityCommand(float linearX, float angularZ)
    {
        Twist twist = new Twist();
        twist.linear = new Vector3(linearX, 0, 0);
        twist.angular = new Vector3(0, 0, angularZ);

        rosSocket.Publish(cmdVelTopic, twist);
    }

    void OnDestroy()
    {
        if (rosSocket != null)
        {
            rosSocket.Close();
        }
    }
}
```

Example of a Unity UI script for robot status display:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Sensor;

public class RobotStatusDisplay : MonoBehaviour
{
    [Header("UI Elements")]
    public Text batteryText;
    public Text positionText;
    public Text statusText;
    public Slider batterySlider;

    [Header("ROS Connection")]
    public string batteryTopic = "/battery_state";
    public string odomTopic = "/odom";

    private RosSocket rosSocket;

    void Start()
    {
        // Initialize ROS connection
        WebSocketNativeClient webSocket = new WebSocketNativeClient(
            WebSocketProtocol.WebSocketSharp, "ws://192.168.1.100:9090");
        rosSocket = new RosSocket(webSocket);

        // Subscribe to robot status topics
        rosSocket.Subscribe<sensor_msgs.BatteryState>(batteryTopic, BatteryCallback);
        rosSocket.Subscribe<nav_msgs.Odometry>(odomTopic, OdometryCallback);
    }

    void BatteryCallback(sensor_msgs.BatteryState batteryState)
    {
        // Update battery UI
        float batteryLevel = (float)batteryState.percentage;
        batterySlider.value = batteryLevel;
        batteryText.text = $"Battery: {batteryLevel:F1}%";

        // Color code based on battery level
        if (batteryLevel < 20)
        {
            batteryText.color = Color.red;
            batterySlider.GetComponent<Image>().color = Color.red;
        }
        else if (batteryLevel < 50)
        {
            batteryText.color = Color.yellow;
        }
        else
        {
            batteryText.color = Color.green;
        }
    }

    void OdometryCallback(nav_msgs.Odometry odom)
    {
        // Update position display
        var pos = odom.pose.pose.position;
        positionText.text = $"Position: ({pos.x:F2}, {pos.y:F2}, {pos.z:F2})";
    }

    public void EmergencyStop()
    {
        // Send emergency stop command
        RosSharp.Messages.Geometry.Twist zeroTwist = new RosSharp.Messages.Geometry.Twist();
        zeroTwist.linear = new Vector3(0, 0, 0);
        zeroTwist.angular = new Vector3(0, 0, 0);

        rosSocket.Publish("/cmd_vel", zeroTwist);
        statusText.text = "EMERGENCY STOP";
        statusText.color = Color.red;
    }
}
```

Example of a simple robot model setup in Unity:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class UnityRobotModel : MonoBehaviour
{
    [Header("Robot Configuration")]
    public float maxVelocity = 1.0f;
    public float maxAngularVelocity = 1.0f;

    [Header("Joint Configuration")]
    public ConfigurableJoint[] joints;
    public float[] jointLimits;

    private Rigidbody rb;
    private Vector3 targetVelocity;
    private Vector3 targetAngularVelocity;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
    }

    void FixedUpdate()
    {
        // Apply movement based on target velocities
        rb.velocity = new Vector3(targetVelocity.x, rb.velocity.y, targetVelocity.z);
        rb.angularVelocity = new Vector3(0, targetAngularVelocity.y, 0);
    }

    public void SetTargetVelocity(Vector3 linear, Vector3 angular)
    {
        targetVelocity = linear.normalized * Mathf.Clamp(linear.magnitude, 0, maxVelocity);
        targetAngularVelocity = angular.normalized * Mathf.Clamp(angular.magnitude, 0, maxAngularVelocity);
    }

    public void SetJointPositions(float[] positions)
    {
        for (int i = 0; i < joints.Length && i < positions.Length; i++)
        {
            if (joints[i] != null)
            {
                // Apply joint position (this is a simplified example)
                // In practice, you'd need to properly configure joint drives
                joints[i].targetPosition = new Vector3(positions[i], 0, 0);
            }
        }
    }

    // Get current robot state for publishing to ROS
    public Vector3 GetPosition()
    {
        return transform.position;
    }

    public Quaternion GetRotation()
    {
        return transform.rotation;
    }
}
```

## Exercises

1. **Model Import**: Import a simple robot model (e.g., a 2-wheeled robot) into Unity and set up basic physics and visualization.

2. **ROS# Integration**: Create a Unity scene that subscribes to a ROS topic and visualizes the data in a 3D environment.

3. **UI Development**: Design a Unity user interface for monitoring and controlling a robot, including battery status, position, and command inputs.

4. **Interaction Design**: Implement a Unity scene where users can interact with a robot model using mouse or touch input, with corresponding ROS messages sent to the real robot.

## Summary

Unity provides a powerful platform for creating high-quality visualizations and interactive interfaces for robotics applications. While it may not replace specialized physics simulators like Gazebo for detailed robot simulation, it excels in creating visually compelling, user-friendly interfaces for human-robot interaction, monitoring, and teleoperation. The integration with ROS through ROS# enables real-time communication between Unity applications and robotic systems, making it an invaluable tool for developing sophisticated user interfaces and visualization systems for Physical AI and humanoid robotics applications.