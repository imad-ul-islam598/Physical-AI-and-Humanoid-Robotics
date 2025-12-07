---
sidebar_position: 4
---

# Chapter 4: ROS 2 Basics

## Introduction

Robot Operating System 2 (ROS 2) serves as the nervous system for modern robotics applications, providing a flexible framework for developing robot software. Unlike traditional operating systems, ROS 2 is a middleware that offers libraries, tools, and conventions for creating robot applications. This chapter introduces the fundamental concepts of ROS 2, including its core architectural elements: nodes, topics, services, and actions, which form the backbone of communication in robotic systems.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the architecture and purpose of ROS 2
- Identify and explain the core communication concepts: nodes, topics, services, and actions
- Implement basic ROS 2 nodes using Python
- Create publishers and subscribers for data exchange
- Distinguish between different communication patterns in ROS 2

## Key Concepts

- **Node**: A process that performs computation in ROS 2
- **Topic**: A named channel for passing messages between nodes
- **Publisher**: A node that sends messages on a topic
- **Subscriber**: A node that receives messages from a topic
- **Service**: A synchronous request/response communication pattern
- **Action**: An asynchronous communication pattern for long-running tasks
- **rclpy**: Python client library for ROS 2

## Technical Explanation

ROS 2 (Robot Operating System 2) is not an operating system but rather a middleware framework designed for robot software development. It provides a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a heterogeneous cluster of computers.

The core architecture of ROS 2 is based on a distributed system where multiple processes (nodes) communicate with each other through messages. This architecture enables modularity, allowing different components of a robot system to be developed and tested independently.

**Nodes** are the fundamental building blocks of ROS 2 applications. Each node is a process that performs specific computations and communicates with other nodes. Nodes can be written in different programming languages (C++, Python, etc.) and can run on different machines, as long as they are connected to the same ROS 2 network.

**Topics** provide a publish-subscribe communication pattern. A topic is a named channel through which nodes exchange messages. Publishers send messages to a topic, and subscribers receive messages from that topic. This decouples the sender and receiver, allowing for flexible system design.

**Services** implement a request-response communication pattern. A client sends a request to a service server, which processes the request and returns a response. This is synchronous communication suitable for tasks that have a clear beginning and end.

**Actions** are designed for long-running tasks that require feedback and the ability to cancel. Actions provide a way to send a goal to an action server, receive feedback during execution, and get a result when the goal is completed or canceled.

ROS 2 uses Data Distribution Service (DDS) as its underlying communication middleware, which provides reliable message delivery, quality of service settings, and support for real-time systems.

## Diagrams written as text descriptions

**Diagram 1: ROS 2 Node Communication Architecture**
```
        ROS 2 Network
    ┌─────────────────┐
    │                 │
    │  ┌─────────┐    │
    │  │Node A   │    │
    │  │Pub/Topic│    │
    │  └─────────┘    │
    │       │          │
    │       ▼          │
    │  ┌─────────┐     │
    │  │ Topic   │     │
    │  │(data)   │     │
    │  └─────────┘     │
    │       │          │
    │       ▼          │
    │  ┌─────────┐     │
    │  │Node B   │     │
    │  │Sub/Topic│     │
    │  └─────────┘     │
    │                 │
    └─────────────────┘
```

**Diagram 2: Service Communication Pattern**
```
Client Node           Service Server
     │                      │
     │    Request           │
     │─────────────────────►│
     │                      │
     │                      │─┐
     │                      │ │ (Processing)
     │                      │◄┘
     │                      │
     │◄─────────────────────│
     │   Response           │
```

**Diagram 3: Action Communication Pattern**
```
Action Client           Action Server
     │                      │
     │    Goal              │
     │─────────────────────►│
     │                      │─┐
     │                      │ │ (Execution)
     │◄─────────────────────│ │ (Feedback)
     │   Feedback           │ │
     │─────────────────────►│ │
     │                      │ │
     │    Result            │◄┘
     │◄─────────────────────│
```

## Code Examples

Here's an example of a simple ROS 2 publisher node:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

And here's a corresponding subscriber node:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Example of a service client and server:

Service definition file (add_two_ints.srv):
```
int64 a
int64 b
---
int64 sum
```

Service server:
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercises

1. **Node Implementation**: Create a ROS 2 node that publishes the current time every second to a topic called "current_time".

2. **Subscriber Extension**: Create a subscriber that listens to the "current_time" topic and logs the received time values.

3. **Service Creation**: Design and implement a service that converts temperatures between Celsius and Fahrenheit.

4. **Communication Pattern Analysis**: For each of the following scenarios, determine whether to use topics, services, or actions:
   - Publishing sensor data from a camera
   - Requesting a robot to move to a specific location
   - Controlling a robot arm to pick up an object
   - Receiving periodic updates about robot battery level

## Summary

ROS 2 provides the essential communication infrastructure for modern robotics applications through its core concepts of nodes, topics, services, and actions. Understanding these communication patterns is crucial for developing effective Physical AI systems that can coordinate multiple sensors, actuators, and computational modules. The publish-subscribe model enables decoupled, flexible system design, while services and actions provide synchronous and asynchronous request-response capabilities for specific tasks. As we progress through this textbook, we'll build upon these foundational concepts to create more sophisticated robotic systems.