---
sidebar_position: 19
---

# Chapter 19: Autonomous Humanoid Pipeline

## Introduction

The autonomous humanoid pipeline represents the integration of all the components explored throughout this textbook into a cohesive system capable of receiving voice commands, planning actions, navigating environments, identifying objects, and manipulating them autonomously. This capstone chapter demonstrates how to combine perception, planning, control, and interaction systems to create a complete Physical AI pipeline that enables humanoid robots to perform complex tasks in response to natural language commands.

## Learning Objectives

By the end of this chapter, you will be able to:
- Integrate all subsystems into a complete autonomous humanoid pipeline
- Design system architectures that coordinate perception, planning, and control
- Implement voice command processing with end-to-end task execution
- Handle system-level challenges like failure recovery and state management
- Evaluate the performance of integrated humanoid systems

## Key Concepts

- **System Integration**: Combining individual components into a cohesive system
- **State Management**: Tracking system state across different operational modes
- **Failure Recovery**: Handling and recovering from system failures gracefully
- **Human-Robot Interaction**: Natural interfaces for commanding robot systems
- **End-to-End Pipeline**: Complete workflow from command to execution
- **System Architecture**: High-level design of integrated robotic systems
- **Real-time Coordination**: Managing multiple concurrent processes

## Technical Explanation

The autonomous humanoid pipeline is a complex system that orchestrates multiple subsystems to achieve high-level goals. The pipeline typically follows this sequence:

1. **Command Reception**: Receiving and interpreting high-level commands from users
2. **Task Planning**: Decomposing high-level commands into executable subtasks
3. **Perception**: Sensing and understanding the environment
4. **Navigation Planning**: Planning paths to navigate to relevant locations
5. **Object Identification**: Detecting and recognizing objects of interest
6. **Manipulation Planning**: Planning how to interact with identified objects
7. **Execution**: Executing planned actions while monitoring for failures
8. **Feedback**: Providing status updates and requesting clarification when needed

**System Architecture** for the autonomous pipeline typically uses a service-oriented or component-based approach:

- **Command Interface**: Handles voice commands and natural language processing
- **Task Orchestrator**: Coordinates the overall task execution flow
- **Perception System**: Processes sensor data to understand the environment
- **Planning System**: Generates navigation and manipulation plans
- **Control System**: Executes low-level motor commands
- **State Manager**: Tracks system state and handles transitions
- **Recovery System**: Detects and handles failures or unexpected situations

**State Management** is crucial for autonomous systems, as they must track:
- Current operational mode (idle, listening, planning, executing, etc.)
- Environmental state (object locations, obstacles, etc.)
- Robot state (battery, joint positions, etc.)
- Task state (progress, subtasks completed, etc.)

**Failure Recovery** strategies include:
- **Graceful Degradation**: Continuing operation with reduced capabilities
- **Retry Mechanisms**: Attempting failed actions multiple times
- **Alternative Plans**: Switching to backup plans when primary plans fail
- **Human Intervention**: Requesting human assistance when stuck

**Real-time Coordination** requires managing multiple concurrent processes:
- Sensor data processing at high frequency
- Control commands at appropriate rates
- Planning updates as environment changes
- Communication with external systems

The pipeline must handle the inherent uncertainty in real-world environments through:
- **Probabilistic Reasoning**: Accounting for sensor and actuator uncertainty
- **Adaptive Planning**: Updating plans as new information becomes available
- **Robust Control**: Maintaining performance despite disturbances
- **Continuous Monitoring**: Detecting when the system deviates from expected behavior

## Diagrams written as text descriptions

**Diagram 1: Autonomous Humanoid Pipeline Architecture**
```
User Command
     │
     ▼
Voice Recognition → NLP → Task Decomposition
     │                    │
     ▼                    ▼
Command Interface → Task Orchestrator
                    │
         ┌──────────┴──────────┐
         │                     │
    Perception           Planning
    System ────────────→ System
         │                     │
         ▼                     ▼
    Environment        Navigation/
    Understanding      Manipulation
         │              Planning
         └────────┬────────┘
                   │
                   ▼
              Control System
                   │
                   ▼
              Robot Execution
                   │
         ┌────────┴────────┐
         │                 │
    Success?         Failure?
         │                 │
    Yes ┘            ┌─────┘ No
         │            │
         ▼            ▼
    Feedback    Recovery/
    to User     Retry Logic
```

**Diagram 2: State Machine for Autonomous Operation**
```
          ┌─────────────┐
          │   IDLE      │ ←───┐
          └──────┬──────┘     │
                 │              │
                 ▼              │
        ┌─────────────────┐     │
        │  LISTENING FOR  │     │
        │   COMMANDS      │     │
        └────────┬────────┘     │
                 │              │
                 ▼              │
        ┌─────────────────┐     │
        │   PROCESSING    │     │
        │   COMMAND       │     │
        └────────┬────────┘     │
                 │              │
                 ▼              │
        ┌─────────────────┐     │
        │   EXECUTING     │     │
        │   TASK          │ ────┘
        └────────┬────────┘
                 │
         ┌───────┴───────┐
         │               │
      Success?       Failure?
         │               │
    ┌────┘          ┌────┘
    │               │
    ▼               ▼
┌─────────┐   ┌───────────┐
│ FEEDBACK│   │ RECOVERY  │
│ TO USER │   │  ACTION   │
└─────────┘   └───────────┘
```

**Diagram 3: Real-time Coordination Loop**
```
┌─────────────────────────────────────────────────────────┐
│                    Main Control Loop                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Perceptual  │  │ Planning    │  │ Execution   │    │
│  │ Processing  │  │ Updates     │  │ Control     │    │
│  │ (10-30 Hz)  │  │ (1-5 Hz)    │  │ (100-500Hz)│    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                   │               │         │
│         ▼                   ▼               ▼         │
│  Environment      Plan Updates      Motor Commands    │
│  Understanding    & Adjustments     to Robot          │
│         │                   │               │         │
└─────────┼───────────────────┼───────────────┼─────────┘
            │                   │               │
            └───────────────────┼───────────────┘
                                │
                    System State & Coordination
```

## Code Examples

Here's an example of a complete autonomous humanoid pipeline system:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, JointState
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
import time
import threading
import queue
import json
from enum import Enum
from typing import Dict, List, Optional, Any

class SystemState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    NAVIGATING = "navigating"
    MANIPULATING = "manipulating"
    WAITING_FOR_FEEDBACK = "waiting_for_feedback"
    RECOVERING = "recovering"
    ERROR = "error"

class AutonomousHumanoidPipeline(Node):
    def __init__(self):
        super().__init__('autonomous_humanoid_pipeline')

        # Initialize system state
        self.current_state = SystemState.IDLE
        self.state_lock = threading.Lock()

        # Initialize subsystems
        self.voice_processor = VoiceCommandProcessor(self)
        self.task_planner = TaskPlanner(self)
        self.perception_system = PerceptionSystem(self)
        self.navigation_system = NavigationSystem(self)
        self.manipulation_system = ManipulationSystem(self)
        self.state_manager = StateManager()

        # Publishers and subscribers
        self.status_pub = self.create_publisher(String, '/system_status', 10)
        self.command_sub = self.create_subscription(
            String, '/high_level_command', self.command_callback, 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # Action servers for complex tasks
        self.execute_task_server = ActionServer(
            self,
            ExecuteTask,
            'execute_task',
            self.execute_task_callback,
            callback_group=ReentrantCallbackGroup()
        )

        # System queues
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Timer for state monitoring
        self.state_monitor_timer = self.create_timer(0.1, self.monitor_system_state)

        self.get_logger().info('Autonomous Humanoid Pipeline initialized')

    def command_callback(self, msg):
        """Handle high-level commands from user"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Add command to processing queue
        self.command_queue.put(command)

        # Transition to processing state
        self.set_state(SystemState.PROCESSING)

    def joint_state_callback(self, msg):
        """Update joint state information"""
        self.state_manager.update_joint_state(msg)

    def monitor_system_state(self):
        """Monitor system state and trigger appropriate actions"""
        with self.state_lock:
            current_state = self.current_state

        # Based on current state, trigger appropriate monitoring
        if current_state == SystemState.PROCESSING:
            self.process_commands()
        elif current_state == SystemState.NAVIGATING:
            self.monitor_navigation()
        elif current_state == SystemState.MANIPULATING:
            self.monitor_manipulation()

    def process_commands(self):
        """Process commands from the queue"""
        try:
            while not self.command_queue.empty():
                command = self.command_queue.get_nowait()

                # Parse and execute command
                success = self.execute_command(command)

                if success:
                    self.set_state(SystemState.IDLE)
                    self.publish_status(f"Command completed: {command}")
                else:
                    self.set_state(SystemState.ERROR)
                    self.publish_status(f"Command failed: {command}")

        except queue.Empty:
            pass

    def execute_command(self, command: str) -> bool:
        """Execute a high-level command end-to-end"""
        try:
            self.get_logger().info(f'Executing command: {command}')

            # Step 1: Parse command using voice processor
            parsed_command = self.voice_processor.parse_command(command)
            if not parsed_command:
                self.get_logger().error('Failed to parse command')
                return False

            # Step 2: Plan the task
            self.set_state(SystemState.PROCESSING)
            task_plan = self.task_planner.create_plan(parsed_command)
            if not task_plan:
                self.get_logger().error('Failed to create task plan')
                return False

            # Step 3: Execute the plan step by step
            for i, task in enumerate(task_plan):
                self.get_logger().info(f'Executing task {i+1}/{len(task_plan)}: {task["type"]}')

                success = self.execute_task_step(task)
                if not success:
                    self.get_logger().error(f'Task step failed: {task["type"]}')
                    # Try recovery
                    recovery_success = self.attempt_recovery(task)
                    if not recovery_success:
                        return False

            return True

        except Exception as e:
            self.get_logger().error(f'Error executing command: {str(e)}')
            return False

    def execute_task_step(self, task: Dict[str, Any]) -> bool:
        """Execute a single task step"""
        task_type = task['type']

        if task_type == 'NAVIGATE':
            self.set_state(SystemState.NAVIGATING)
            return self.navigation_system.navigate_to(task['parameters'])
        elif task_type == 'PERCEIVE':
            return self.perception_system.process_environment(task['parameters'])
        elif task_type == 'MANIPULATE':
            self.set_state(SystemState.MANIPULATING)
            return self.manipulation_system.execute_manipulation(task['parameters'])
        elif task_type == 'SPEAK':
            return self.voice_processor.speak(task['parameters']['text'])
        else:
            self.get_logger().error(f'Unknown task type: {task_type}')
            return False

    def attempt_recovery(self, failed_task: Dict[str, Any]) -> bool:
        """Attempt to recover from a failed task"""
        self.set_state(SystemState.RECOVERING)

        # Log the failure
        self.get_logger().warn(f'Attempting recovery for failed task: {failed_task["type"]}')

        # Recovery strategies based on task type
        if failed_task['type'] == 'NAVIGATE':
            # Try alternative navigation approach
            return self.navigation_system.recovery_strategy(failed_task)
        elif failed_task['type'] == 'MANIPULATE':
            # Try alternative grasp or approach
            return self.manipulation_system.recovery_strategy(failed_task)
        elif failed_task['type'] == 'PERCEIVE':
            # Try different perception parameters
            return self.perception_system.recovery_strategy(failed_task)
        else:
            # Default recovery - ask for human assistance
            self.voice_processor.speak("I encountered a problem and need assistance.")
            return False

    def monitor_navigation(self):
        """Monitor navigation progress"""
        if self.navigation_system.is_navigation_complete():
            self.set_state(SystemState.IDLE)

    def monitor_manipulation(self):
        """Monitor manipulation progress"""
        if self.manipulation_system.is_manipulation_complete():
            self.set_state(SystemState.IDLE)

    def set_state(self, new_state: SystemState):
        """Safely update system state"""
        with self.state_lock:
            old_state = self.current_state
            self.current_state = new_state
            self.get_logger().info(f'State transition: {old_state.value} → {new_state.value}')

    def publish_status(self, status: str):
        """Publish system status"""
        status_msg = String()
        status_msg.data = f"{self.current_state.value}: {status}"
        self.status_pub.publish(status_msg)

    def execute_task_callback(self, goal_handle):
        """Handle task execution action"""
        self.get_logger().info('Executing task via action server')

        task_description = goal_handle.request.task_description

        # Execute the task
        success = self.execute_command(task_description)

        if success:
            goal_handle.succeed()
            result = ExecuteTask.Result()
            result.success = True
            result.message = "Task completed successfully"
            return result
        else:
            goal_handle.abort()
            result = ExecuteTask.Result()
            result.success = False
            result.message = "Task failed"
            return result

class VoiceCommandProcessor:
    def __init__(self, pipeline_node):
        self.pipeline_node = pipeline_node
        self.command_map = {
            'go to': 'NAVIGATE',
            'move to': 'NAVIGATE',
            'pick up': 'MANIPULATE',
            'grasp': 'MANIPULATE',
            'get': 'MANIPULATE',
            'place': 'MANIPULATE',
            'put': 'MANIPULATE',
            'find': 'PERCEIVE',
            'look for': 'PERCEIVE',
            'search for': 'PERCEIVE'
        }

    def parse_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Parse natural language command into structured format"""
        command_lower = command.lower()

        for keyword, task_type in self.command_map.items():
            if keyword in command_lower:
                # Extract object/location based on command type
                if task_type == 'NAVIGATE':
                    # Extract destination from command like "go to the kitchen"
                    destination = self.extract_destination(command_lower, keyword)
                    return {
                        'type': task_type,
                        'destination': destination
                    }
                elif task_type == 'MANIPULATE':
                    # Extract object from command like "pick up the red cup"
                    obj = self.extract_object(command_lower, keyword)
                    return {
                        'type': task_type,
                        'object': obj
                    }
                elif task_type == 'PERCEIVE':
                    # Extract search target
                    target = self.extract_target(command_lower, keyword)
                    return {
                        'type': task_type,
                        'target': target
                    }

        return None

    def extract_destination(self, command: str, keyword: str) -> str:
        """Extract destination from navigation command"""
        # Simple extraction - in practice, use NLP
        parts = command.split(keyword)
        if len(parts) > 1:
            remaining = parts[1].strip()
            # Remove common words like "the", "a", "an"
            destination = remaining.replace('the ', '').replace('a ', '').replace('an ', '')
            return destination.strip()
        return "unknown"

    def extract_object(self, command: str, keyword: str) -> str:
        """Extract object from manipulation command"""
        parts = command.split(keyword)
        if len(parts) > 1:
            remaining = parts[1].strip()
            return remaining.replace('the ', '').replace('a ', '').replace('an ', '').strip()
        return "unknown"

    def extract_target(self, command: str, keyword: str) -> str:
        """Extract search target from perception command"""
        parts = command.split(keyword)
        if len(parts) > 1:
            remaining = parts[1].strip()
            return remaining.replace('the ', '').replace('a ', '').replace('an ', '').strip()
        return "unknown"

    def speak(self, text: str) -> bool:
        """Make the robot speak"""
        # In practice, publish to TTS system
        print(f"Robot says: {text}")
        return True

class TaskPlanner:
    def __init__(self, pipeline_node):
        self.pipeline_node = pipeline_node
        self.location_map = {
            'kitchen': {'x': 2.0, 'y': 1.0, 'z': 0.0},
            'living room': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'bedroom': {'x': -1.0, 'y': 1.5, 'z': 0.0},
            'office': {'x': 1.5, 'y': -1.0, 'z': 0.0}
        }

    def create_plan(self, parsed_command: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Create a task plan from parsed command"""
        command_type = parsed_command['type']

        if command_type == 'NAVIGATE':
            destination = parsed_command['destination']
            if destination in self.location_map:
                location = self.location_map[destination]
                return [
                    {
                        'type': 'NAVIGATE',
                        'parameters': {
                            'x': location['x'],
                            'y': location['y'],
                            'z': location['z']
                        }
                    }
                ]
            else:
                # If unknown location, ask for clarification or search
                return [
                    {
                        'type': 'SPEAK',
                        'parameters': {
                            'text': f"I don't know where {destination} is. Can you guide me there?"
                        }
                    }
                ]

        elif command_type == 'MANIPULATE':
            obj = parsed_command['object']
            return [
                {
                    'type': 'PERCEIVE',
                    'parameters': {
                        'target': obj
                    }
                },
                {
                    'type': 'NAVIGATE',
                    'parameters': {
                        'x': 0.5,  # Approach object location
                        'y': 0.0,
                        'z': 0.0
                    }
                },
                {
                    'type': 'MANIPULATE',
                    'parameters': {
                        'object': obj,
                        'action': 'grasp'
                    }
                }
            ]

        elif command_type == 'PERCEIVE':
            target = parsed_command['target']
            return [
                {
                    'type': 'PERCEIVE',
                    'parameters': {
                        'target': target
                    }
                },
                {
                    'type': 'SPEAK',
                    'parameters': {
                        'text': f"I {'found' if self.found_target(target) else 'did not find'} the {target}"
                    }
                }
            ]

        return None

    def found_target(self, target: str) -> bool:
        """Simulate target detection"""
        # In practice, this would use perception system
        import random
        return random.random() > 0.3  # 70% chance of finding target

class PerceptionSystem:
    def __init__(self, pipeline_node):
        self.pipeline_node = pipeline_node
        self.detected_objects = {}
        self.environment_map = {}

    def process_environment(self, parameters: Dict[str, Any]):
        """Process environment to detect objects"""
        target = parameters.get('target', 'any')

        # Simulate object detection
        # In practice, this would use camera, LIDAR, etc.
        detected = self.simulate_detection(target)

        if detected:
            self.pipeline_node.get_logger().info(f'Detected target: {target}')
            return True
        else:
            self.pipeline_node.get_logger().info(f'Did not detect target: {target}')
            return False

    def simulate_detection(self, target: str) -> bool:
        """Simulate object detection"""
        # Simulate detection with some probability
        import random
        return random.random() > 0.2  # 80% detection rate

    def recovery_strategy(self, failed_task: Dict[str, Any]) -> bool:
        """Recovery strategy for perception failures"""
        # Try with different parameters or sensors
        self.pipeline_node.get_logger().info('Trying perception recovery with different parameters')
        return self.process_environment(failed_task['parameters'])

class NavigationSystem:
    def __init__(self, pipeline_node):
        self.pipeline_node = pipeline_node
        self.current_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.is_moving = False

    def navigate_to(self, parameters: Dict[str, Any]) -> bool:
        """Navigate to specified location"""
        target_x = parameters.get('x', 0.0)
        target_y = parameters.get('y', 0.0)
        target_z = parameters.get('z', 0.0)

        self.pipeline_node.get_logger().info(f'Navigating to ({target_x}, {target_y}, {target_z})')

        # Simulate navigation
        # In practice, this would use navigation stack
        self.is_moving = True

        # Simulate movement
        import time
        time.sleep(2)  # Simulate navigation time

        # Update position
        self.current_position = {'x': target_x, 'y': target_y, 'z': target_z}
        self.is_moving = False

        self.pipeline_node.get_logger().info('Navigation completed')
        return True

    def is_navigation_complete(self) -> bool:
        """Check if navigation is complete"""
        return not self.is_moving

    def recovery_strategy(self, failed_task: Dict[str, Any]) -> bool:
        """Recovery strategy for navigation failures"""
        # Try alternative route or ask for help
        self.pipeline_node.get_logger().info('Trying alternative navigation route')
        return self.navigate_to(failed_task['parameters'])

class ManipulationSystem:
    def __init__(self, pipeline_node):
        self.pipeline_node = pipeline_node
        self.is_manipulating = False

    def execute_manipulation(self, parameters: Dict[str, Any]) -> bool:
        """Execute manipulation task"""
        obj = parameters.get('object', 'unknown')
        action = parameters.get('action', 'grasp')

        self.pipeline_node.get_logger().info(f'Performing {action} on {obj}')

        # Simulate manipulation
        self.is_manipulating = True

        # Simulate manipulation time
        import time
        time.sleep(1.5)

        self.is_manipulating = False

        self.pipeline_node.get_logger().info(f'Manipulation of {obj} completed')
        return True

    def is_manipulation_complete(self) -> bool:
        """Check if manipulation is complete"""
        return not self.is_manipulating

    def recovery_strategy(self, failed_task: Dict[str, Any]) -> bool:
        """Recovery strategy for manipulation failures"""
        # Try different grasp or approach
        self.pipeline_node.get_logger().info('Trying alternative manipulation approach')
        return self.execute_manipulation(failed_task['parameters'])

class StateManager:
    def __init__(self):
        self.joint_states = {}
        self.battery_level = 100.0
        self.current_location = {'x': 0.0, 'y': 0.0, 'z': 0.0}

    def update_joint_state(self, joint_state_msg):
        """Update joint state information"""
        for i, name in enumerate(joint_state_msg.name):
            if i < len(joint_state_msg.position):
                self.joint_states[name] = joint_state_msg.position[i]

    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        return {
            'joint_states': self.joint_states.copy(),
            'battery_level': self.battery_level,
            'location': self.current_location.copy()
        }

# Action message definition (would be in separate .action file in practice)
class ExecuteTask:
    class Goal:
        def __init__(self):
            self.task_description = ""

    class Result:
        def __init__(self):
            self.success = False
            self.message = ""

def main(args=None):
    rclpy.init(args=args)
    pipeline = AutonomousHumanoidPipeline()

    # Example commands to test the system
    test_commands = [
        "Go to the kitchen",
        "Find the red cup",
        "Pick up the book",
        "Move to the living room"
    ]

    # For testing, we'll publish a command after a delay
    def publish_test_command():
        time.sleep(2)  # Wait for system to initialize
        cmd_msg = String()
        cmd_msg.data = "Go to the kitchen"
        pipeline.command_sub.publish(cmd_msg)
        print("Published test command: Go to the kitchen")

    # Start test command publisher in separate thread
    test_thread = threading.Thread(target=publish_test_command, daemon=True)
    test_thread.start()

    try:
        rclpy.spin(pipeline)
    except KeyboardInterrupt:
        pipeline.get_logger().info('Shutting down autonomous pipeline...')
    finally:
        pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Example of a more sophisticated system monitor and evaluation framework:

```python
import time
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
import threading

@dataclass
class SystemMetrics:
    """Data class for system metrics"""
    timestamp: float
    success_rate: float
    execution_time: float
    failure_count: int
    recovery_attempts: int
    battery_level: float
    cpu_usage: float
    memory_usage: float

class SystemMonitor:
    def __init__(self, pipeline_node):
        self.pipeline_node = pipeline_node
        self.metrics_history: List[SystemMetrics] = []
        self.task_history: List[Dict] = []
        self.performance_log = []
        self.monitoring = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_system, daemon=True)
        self.monitor_thread.start()

    def monitor_system(self):
        """Continuously monitor system performance"""
        while self.monitoring:
            try:
                # Collect current metrics
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)

                # Log performance data
                self.log_performance(metrics)

                # Check for performance degradation
                self.check_performance_degradation(metrics)

                # Sleep for monitoring interval
                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                self.pipeline_node.get_logger().error(f'Error in system monitoring: {str(e)}')
                time.sleep(1.0)

    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # In practice, get real metrics from system
        import random
        return SystemMetrics(
            timestamp=time.time(),
            success_rate=random.uniform(0.7, 1.0),  # Simulated success rate
            execution_time=random.uniform(5.0, 15.0),  # Simulated execution time
            failure_count=0,  # Would track actual failures
            recovery_attempts=0,  # Would track actual recovery attempts
            battery_level=85.0,  # Simulated battery level
            cpu_usage=random.uniform(20.0, 60.0),  # Simulated CPU usage
            memory_usage=random.uniform(30.0, 70.0)  # Simulated memory usage
        )

    def log_performance(self, metrics: SystemMetrics):
        """Log performance metrics"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(metrics.timestamp).isoformat(),
            'success_rate': metrics.success_rate,
            'execution_time': metrics.execution_time,
            'battery_level': metrics.battery_level,
            'cpu_usage': metrics.cpu_usage,
            'memory_usage': metrics.memory_usage
        }
        self.performance_log.append(log_entry)

    def check_performance_degradation(self, metrics: SystemMetrics):
        """Check for performance degradation and trigger alerts if needed"""
        # Check if success rate is dropping
        if len(self.metrics_history) > 10:
            recent_success_rates = [m.success_rate for m in self.metrics_history[-10:]]
            avg_recent = sum(recent_success_rates) / len(recent_success_rates)
            if avg_recent < 0.8:  # Alert if success rate drops below 80%
                self.pipeline_node.get_logger().warn(
                    f'Performance degradation detected: recent success rate = {avg_recent:.2f}'
                )

    def get_system_health(self) -> Dict[str, float]:
        """Get overall system health metrics"""
        if not self.metrics_history:
            return {'health_score': 0.0}

        # Calculate health score based on various metrics
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history

        avg_success_rate = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)

        # Normalize metrics to 0-1 scale
        success_score = avg_success_rate
        time_score = max(0.0, 1.0 - (avg_execution_time / 30.0))  # Assume 30s is maximum acceptable

        # Weighted health score
        health_score = 0.7 * success_score + 0.3 * time_score

        return {
            'health_score': health_score,
            'success_rate': avg_success_rate,
            'avg_execution_time': avg_execution_time
        }

    def generate_performance_report(self) -> str:
        """Generate a performance report"""
        if not self.metrics_history:
            return "No performance data available"

        recent_metrics = self.metrics_history[-50:] if len(self.metrics_history) >= 50 else self.metrics_history

        avg_success_rate = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)

        report = f"""
Autonomous Humanoid System Performance Report
============================================

Time Period: Last {len(recent_metrics)} monitoring intervals

Performance Metrics:
- Success Rate: {avg_success_rate:.2%}
- Average Execution Time: {avg_execution_time:.2f}s
- Average CPU Usage: {avg_cpu:.1f}%
- Average Memory Usage: {avg_memory:.1f}%

System Health Score: {(0.7 * avg_success_rate + 0.3 * min(1.0, 30.0/avg_execution_time)):.2f}/1.0

Recommendations:
"""
        if avg_success_rate < 0.8:
            report += "- Success rate is below target (80%). Consider improving perception or planning.\n"
        if avg_execution_time > 15.0:
            report += "- Execution time is high. Consider optimizing task planning.\n"
        if avg_cpu > 80.0:
            report += "- CPU usage is high. Consider optimizing computational tasks.\n"

        return report

class TaskEvaluator:
    def __init__(self, pipeline_node):
        self.pipeline_node = pipeline_node
        self.completed_tasks = []
        self.failed_tasks = []

    def evaluate_task(self, task_description: str, execution_result: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the execution of a task"""
        evaluation = {
            'task_success': execution_result.get('success', False),
            'execution_time': execution_result.get('execution_time', 0.0),
            'energy_consumption': execution_result.get('energy_used', 0.0),
            'accuracy': execution_result.get('accuracy', 0.0),
            'safety_score': execution_result.get('safety_rating', 1.0)
        }

        # Calculate overall task score
        weights = {
            'success': 0.4,
            'time': 0.2,
            'energy': 0.15,
            'accuracy': 0.15,
            'safety': 0.1
        }

        score = 0.0
        if evaluation['task_success']:
            score += weights['success']

        # Time efficiency (lower is better, max 10 seconds)
        time_score = max(0.0, 1.0 - min(1.0, evaluation['execution_time'] / 10.0))
        score += weights['time'] * time_score

        # Energy efficiency (lower is better, max 50 units)
        energy_score = max(0.0, 1.0 - min(1.0, evaluation['energy_consumption'] / 50.0))
        score += weights['energy'] * energy_score

        # Accuracy (higher is better)
        score += weights['accuracy'] * evaluation['accuracy']

        # Safety (higher is better)
        score += weights['safety'] * evaluation['safety_score']

        evaluation['overall_score'] = score
        return evaluation

    def add_task_result(self, task_description: str, result: Dict[str, Any]):
        """Add a task result to the evaluation system"""
        evaluation = self.evaluate_task(task_description, result)

        if result.get('success', False):
            self.completed_tasks.append({
                'task': task_description,
                'evaluation': evaluation,
                'timestamp': time.time()
            })
        else:
            self.failed_tasks.append({
                'task': task_description,
                'evaluation': evaluation,
                'timestamp': time.time()
            })

    def get_system_performance(self) -> Dict[str, float]:
        """Get overall system performance metrics"""
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        if total_tasks == 0:
            return {
                'success_rate': 0.0,
                'avg_execution_time': 0.0,
                'avg_task_score': 0.0
            }

        success_rate = len(self.completed_tasks) / total_tasks
        avg_execution_time = sum(
            t['evaluation']['execution_time'] for t in self.completed_tasks
        ) / len(self.completed_tasks) if self.completed_tasks else 0.0

        avg_task_score = sum(
            t['evaluation']['overall_score'] for t in self.completed_tasks
        ) / len(self.completed_tasks) if self.completed_tasks else 0.0

        return {
            'success_rate': success_rate,
            'avg_execution_time': avg_execution_time,
            'avg_task_score': avg_task_score,
            'total_completed': len(self.completed_tasks),
            'total_failed': len(self.failed_tasks)
        }

def create_evaluation_framework():
    """Create a comprehensive evaluation framework"""
    print("Setting up evaluation framework...")

    # Example evaluation scenarios
    scenarios = [
        {
            'name': 'Basic Navigation',
            'commands': ['Go to the kitchen', 'Go to the living room'],
            'expected_outcomes': ['navigation_success'] * 2
        },
        {
            'name': 'Object Manipulation',
            'commands': ['Pick up the red cup', 'Place the book on the table'],
            'expected_outcomes': ['manipulation_success'] * 2
        },
        {
            'name': 'Complex Task',
            'commands': ['Go to the kitchen and find the apple'],
            'expected_outcomes': ['navigation_success', 'perception_success']
        }
    ]

    print("Evaluation framework ready")
    print(f"Defined {len(scenarios)} test scenarios")

    return scenarios

def main_evaluation():
    """Main evaluation function"""
    print("Autonomous Humanoid Pipeline - System Evaluation")
    print("=" * 50)

    # Create evaluation framework
    scenarios = create_evaluation_framework()

    # Initialize monitoring and evaluation systems
    monitor = SystemMonitor(None)  # Pass actual pipeline node in real implementation
    evaluator = TaskEvaluator(None)

    print("\nStarting system evaluation...")

    # Simulate task execution and evaluation
    for scenario in scenarios:
        print(f"\nExecuting scenario: {scenario['name']}")

        for command in scenario['commands']:
            print(f"  Executing: {command}")

            # Simulate task execution result
            import random
            success = random.random() > 0.2  # 80% success rate for simulation

            result = {
                'success': success,
                'execution_time': random.uniform(5.0, 15.0),
                'energy_used': random.uniform(10.0, 30.0),
                'accuracy': random.uniform(0.8, 1.0),
                'safety_rating': random.uniform(0.9, 1.0)
            }

            evaluator.add_task_result(command, result)
            print(f"    Result: {'Success' if success else 'Failed'} (Score: {result.get('accuracy', 0):.2f})")

    # Generate performance report
    performance = evaluator.get_system_performance()
    print(f"\nOverall Performance:")
    print(f"  Success Rate: {performance['success_rate']:.1%}")
    print(f"  Avg Execution Time: {performance['avg_execution_time']:.2f}s")
    print(f"  Avg Task Score: {performance['avg_task_score']:.2f}")

    print(f"\nSystem completed {performance['total_completed']} tasks, failed {performance['total_failed']} tasks")

    # Generate detailed report
    report = monitor.generate_performance_report()
    print(f"\nDetailed Performance Report:")
    print(report)

if __name__ == '__main__':
    main_evaluation()
```

## Exercises

1. **System Integration**: Integrate all the subsystems covered in previous chapters into a working pipeline.

2. **State Management**: Implement a robust state management system that handles all operational modes.

3. **Failure Recovery**: Design and implement comprehensive failure recovery strategies.

4. **Performance Evaluation**: Create metrics and evaluation frameworks to assess system performance.

5. **Real-time Coordination**: Implement real-time coordination between perception, planning, and control.

## Summary

The autonomous humanoid pipeline represents the culmination of all the concepts explored throughout this textbook, demonstrating how perception, planning, control, and interaction systems can be integrated into a cohesive whole. Success in this domain requires careful attention to system architecture, state management, failure recovery, and real-time coordination. The pipeline must handle the inherent uncertainty of real-world environments while maintaining safety and achieving user goals. As we conclude our exploration of Physical AI and Humanoid Robotics, the principles and techniques covered in this textbook provide the foundation for developing sophisticated autonomous humanoid systems capable of performing complex tasks in human environments. The future of Physical AI lies in creating systems that can seamlessly integrate into human spaces, understand natural commands, and assist with everyday tasks while maintaining the highest standards of safety and reliability.