---
sidebar_position: 14
---

# Chapter 14: Cognitive Planning with LLMs

## Introduction

Large Language Models (LLMs) have emerged as powerful tools for cognitive planning in Physical AI systems, enabling robots to understand natural language commands and decompose them into executable action sequences. This chapter explores how LLMs can serve as high-level cognitive planners, bridging the gap between human intentions expressed in natural language and low-level robot control commands. We'll examine the integration of LLMs with robotic systems, task decomposition strategies, and the challenges of grounding abstract language in concrete physical actions.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand how LLMs can be integrated with robotic systems for cognitive planning
- Implement task decomposition using LLMs for complex robot behaviors
- Address the grounding problem of mapping abstract language to physical actions
- Design effective prompts for robot task planning with LLMs
- Evaluate the reliability and safety of LLM-based cognitive planning

## Key Concepts

- **Large Language Models (LLMs)**: Neural networks trained on vast text corpora for language understanding and generation
- **Cognitive Planning**: High-level planning that involves reasoning about tasks and goals
- **Task Decomposition**: Breaking complex tasks into simpler, executable subtasks
- **Language Grounding**: Connecting abstract language concepts to physical entities and actions
- **Prompt Engineering**: Designing effective inputs for LLMs to produce desired outputs
- **Plan Execution**: Converting LLM-generated plans into robot actions

## Technical Explanation

Cognitive planning with LLMs represents a paradigm shift in robotics, moving from purely algorithmic planning to language-based reasoning. LLMs excel at understanding the compositional and contextual nature of human instructions, making them ideal for interpreting high-level commands and generating appropriate action sequences.

The cognitive planning pipeline with LLMs involves several key components:

1. **Perception Integration**: LLMs need to be aware of the current state of the world and the robot's capabilities. This includes information about objects, their properties, locations, and the robot's current state.

2. **Natural Language Understanding**: The LLM interprets the human command, identifying the goal, constraints, and any relevant context from the natural language input.

3. **Task Decomposition**: The LLM breaks down complex tasks into smaller, manageable subtasks that can be executed by the robot. This involves reasoning about dependencies, prerequisites, and optimal ordering.

4. **Action Mapping**: Each subtask is mapped to specific robot actions or behaviors, considering the robot's kinematic and dynamic constraints.

5. **Plan Refinement**: The LLM may refine the plan based on feedback from the robot's execution or new information from the environment.

The grounding problem is particularly challenging in this context. While LLMs have vast knowledge about the world, they must connect abstract concepts to concrete physical entities in the robot's environment. For example, when a human says "the red cup," the LLM must help the robot identify which physical object corresponds to this description.

LLMs can be integrated with robotic systems in several ways:

- **Direct Integration**: Using the LLM as a planning service that generates action sequences
- **Chain-of-Thought Reasoning**: Using intermediate reasoning steps to improve planning accuracy
- **Few-Shot Learning**: Providing examples of successful plans to guide the LLM
- **Tool Integration**: Combining LLM reasoning with specialized tools for perception, navigation, etc.

Task decomposition strategies include:

- **Hierarchical Decomposition**: Breaking tasks into high-level goals, then into subgoals, and finally into primitive actions
- **Temporal Decomposition**: Sequencing actions based on temporal relationships and dependencies
- **Functional Decomposition**: Grouping actions based on their function or purpose

The effectiveness of LLM-based planning depends heavily on prompt engineering. Well-crafted prompts provide the LLM with the necessary context, constraints, and examples to generate appropriate plans. This includes information about the robot's capabilities, the environment, and the expected format of the response.

## Diagrams written as text descriptions

**Diagram 1: LLM Cognitive Planning Architecture**
```
Human Command (Natural Language)
         │
         ▼
LLM Planner (Task Decomposition)
         │
         ▼
Structured Plan (Subtasks + Dependencies)
         │
         ▼
Action Mapping (Robot Commands)
         │
         ▼
Robot Execution System
         │
         ▼
Environment Feedback → Perception System → LLM (for plan adjustment)
```

**Diagram 2: Task Decomposition Process**
```
High-Level Command: "Set the table for dinner"
         │
         ▼
Level 1: [Get plates, Get utensils, Get glasses, Place items]
         │
         ▼
Level 2: [Go to cabinet → Get plates → Place on table,
          Go to drawer → Get forks → Place on table,
          Go to cabinet → Get glasses → Place on table]
         │
         ▼
Level 3: [Navigation commands, Manipulation commands, Placement actions]
```

**Diagram 3: LLM-Robot Integration Loop**
```
Environment State
     │
     ▼
LLM Context (Objects, Locations, Capabilities)
     │
     ▼
Human Command → LLM Planner → Action Sequence
     │                              │
     │                              ▼
     └─── Feedback ←─── Robot Execution ←───┘
```

## Code Examples

Here's an example of integrating an LLM with a robotic system for cognitive planning:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import openai
import json
import time
from typing import Dict, List, Any, Optional

class LLMBasedPlanner(Node):
    def __init__(self):
        super().__init__('llm_planner')

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key='your-api-key-here')

        # Publishers for different robot commands
        self.navigation_pub = self.create_publisher(String, '/navigation_goal', 10)
        self.manipulation_pub = self.create_publisher(String, '/manipulation_goal', 10)
        self.speech_pub = self.create_publisher(String, '/tts_input', 10)

        # Subscribers for robot state
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.robot_pose_sub = self.create_subscription(
            Pose, '/robot_pose', self.robot_pose_callback, 10)

        # Robot state
        self.current_joints = None
        self.current_pose = None
        self.object_locations = {}  # Will be populated by perception system

        # Service for receiving high-level commands
        self.command_sub = self.create_subscription(
            String, '/high_level_command', self.command_callback, 10)

        self.get_logger().info('LLM-Based Planner initialized')

    def joint_state_callback(self, msg):
        """Update current joint states"""
        self.current_joints = msg

    def robot_pose_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg

    def command_callback(self, msg):
        """Process high-level command using LLM"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Get robot state and environment context
        context = self.get_robot_context()

        # Generate plan using LLM
        plan = self.generate_plan_with_llm(command, context)

        if plan:
            self.execute_plan(plan)
        else:
            self.get_logger().error('Failed to generate plan')

    def get_robot_context(self) -> Dict[str, Any]:
        """Get current robot state and environment context"""
        context = {
            'robot_state': {
                'position': {
                    'x': self.current_pose.position.x if self.current_pose else 0.0,
                    'y': self.current_pose.position.y if self.current_pose else 0.0,
                    'z': self.current_pose.position.z if self.current_pose else 0.0
                } if self.current_pose else None,
                'joints': {
                    'names': list(self.current_joints.name) if self.current_joints else [],
                    'positions': list(self.current_joints.position) if self.current_joints else []
                } if self.current_joints else None
            },
            'environment': {
                'object_locations': self.object_locations,
                'robot_capabilities': {
                    'max_reach': 1.0,  # meters
                    'max_payload': 2.0,  # kg
                    'navigation_enabled': True,
                    'manipulation_enabled': True
                }
            }
        }
        return context

    def generate_plan_with_llm(self, command: str, context: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Generate a plan using LLM based on command and context"""
        try:
            # Create a detailed prompt for the LLM
            prompt = f"""
            You are a cognitive planner for a humanoid robot. Your task is to decompose high-level commands into executable subtasks.

            Current robot state:
            {json.dumps(context, indent=2)}

            Command: {command}

            Please decompose this command into a sequence of executable subtasks. Each subtask should be one of:
            - NAVIGATE: Move to a specific location
            - PICK_UP: Pick up an object
            - PLACE: Place an object at a location
            - GREET: Perform a greeting action
            - SPEAK: Say something to the user

            Return the plan as a JSON array of subtasks, where each subtask has:
            - type: The subtask type
            - parameters: Relevant parameters for the subtask

            Example response format:
            [
                {{
                    "type": "NAVIGATE",
                    "parameters": {{
                        "target_location": "kitchen",
                        "x": 2.5,
                        "y": 1.0
                    }}
                }},
                {{
                    "type": "PICK_UP",
                    "parameters": {{
                        "object": "red cup",
                        "object_id": "cup_001"
                    }}
                }}
            ]

            Plan:
            """

            response = self.client.chat.completions.create(
                model="gpt-4-turbo",  # or another appropriate model
                messages=[
                    {"role": "system", "content": "You are a cognitive planner for a humanoid robot. Generate executable plans for robot tasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent outputs
                response_format={"type": "json_object"}  # Request JSON response
            )

            # Extract the plan from the response
            plan_text = response.choices[0].message.content
            self.get_logger().info(f'LLM response: {plan_text}')

            # Parse the JSON response
            plan = json.loads(plan_text)
            return plan

        except Exception as e:
            self.get_logger().error(f'Error generating plan with LLM: {str(e)}')
            return None

    def execute_plan(self, plan: List[Dict[str, Any]]):
        """Execute the plan generated by the LLM"""
        self.get_logger().info(f'Executing plan with {len(plan)} subtasks')

        for i, subtask in enumerate(plan):
            self.get_logger().info(f'Executing subtask {i+1}/{len(plan)}: {subtask["type"]}')

            if subtask['type'] == 'NAVIGATE':
                self.execute_navigation(subtask['parameters'])
            elif subtask['type'] == 'PICK_UP':
                self.execute_pickup(subtask['parameters'])
            elif subtask['type'] == 'PLACE':
                self.execute_placement(subtask['parameters'])
            elif subtask['type'] == 'GREET':
                self.execute_greeting(subtask['parameters'])
            elif subtask['type'] == 'SPEAK':
                self.execute_speech(subtask['parameters'])
            else:
                self.get_logger().error(f'Unknown subtask type: {subtask["type"]}')
                continue

            # Wait for task completion (in practice, this would be more sophisticated)
            time.sleep(1)

        self.get_logger().info('Plan execution completed')

    def execute_navigation(self, params: Dict[str, Any]):
        """Execute navigation subtask"""
        nav_msg = String()
        nav_msg.data = json.dumps({
            'target_location': params.get('target_location', 'unknown'),
            'x': params.get('x', 0.0),
            'y': params.get('y', 0.0)
        })
        self.navigation_pub.publish(nav_msg)
        self.get_logger().info(f'Navigating to {params.get("target_location", "unknown")}')

    def execute_pickup(self, params: Dict[str, Any]):
        """Execute pickup subtask"""
        pickup_msg = String()
        pickup_msg.data = json.dumps({
            'object': params.get('object', 'unknown'),
            'object_id': params.get('object_id', 'unknown')
        })
        self.manipulation_pub.publish(pickup_msg)
        self.get_logger().info(f'Picking up {params.get("object", "unknown")}')

    def execute_placement(self, params: Dict[str, Any]):
        """Execute placement subtask"""
        place_msg = String()
        place_msg.data = json.dumps({
            'object': params.get('object', 'unknown'),
            'location': params.get('location', 'default')
        })
        self.manipulation_pub.publish(place_msg)
        self.get_logger().info(f'Placing {params.get("object", "unknown")} at {params.get("location", "default")}')

    def execute_greeting(self, params: Dict[str, Any]):
        """Execute greeting subtask"""
        speech_msg = String()
        greeting = params.get('greeting', 'Hello! How can I help you?')
        speech_msg.data = greeting
        self.speech_pub.publish(speech_msg)
        self.get_logger().info(f'Greeting: {greeting}')

    def execute_speech(self, params: Dict[str, Any]):
        """Execute speech subtask"""
        speech_msg = String()
        text = params.get('text', 'I have completed the task.')
        speech_msg.data = text
        self.speech_pub.publish(speech_msg)
        self.get_logger().info(f'Speaking: {text}')

def main(args=None):
    rclpy.init(args=args)
    planner = LLMBasedPlanner()

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.get_logger().info('Shutting down LLM Planner...')
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Example of a more sophisticated task decomposition system with error handling:

```python
import asyncio
from typing import Dict, List, Tuple, Optional, Callable
import copy

class RobustLLMPlanner:
    def __init__(self, llm_client):
        self.client = llm_client
        self.max_retries = 3
        self.task_history = []

    async def generate_robust_plan(self, command: str, context: Dict,
                                 validate_fn: Callable = None) -> Optional[List[Dict]]:
        """
        Generate a plan with multiple validation and refinement steps
        """
        # Step 1: Generate initial plan
        plan = await self.generate_initial_plan(command, context)
        if not plan:
            return None

        # Step 2: Validate plan
        if validate_fn:
            is_valid, feedback = validate_fn(plan, context)
            if not is_valid:
                # Step 3: Refine plan based on feedback
                plan = await self.refine_plan(command, context, plan, feedback)

        # Step 4: Final validation
        if validate_fn:
            is_valid, _ = validate_fn(plan, context)
            if not is_valid:
                self.get_logger().error('Plan failed final validation')
                return None

        return plan

    async def generate_initial_plan(self, command: str, context: Dict) -> Optional[List[Dict]]:
        """Generate initial plan with multiple attempts"""
        for attempt in range(self.max_retries):
            try:
                prompt = self.create_planning_prompt(command, context)
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": self.get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )

                plan_text = response.choices[0].message.content
                plan = json.loads(plan_text)

                # Validate plan structure
                if self.validate_plan_structure(plan):
                    return plan
                else:
                    self.get_logger().warning(f'Plan structure invalid on attempt {attempt + 1}')

            except Exception as e:
                self.get_logger().error(f'Error on attempt {attempt + 1}: {str(e)}')
                if attempt == self.max_retries - 1:
                    return None
                await asyncio.sleep(1)  # Brief delay before retry

        return None

    def create_planning_prompt(self, command: str, context: Dict) -> str:
        """Create a detailed prompt for the LLM"""
        return f"""
        You are a cognitive planner for a humanoid robot. Decompose the following command into executable subtasks.

        Robot Capabilities:
        - Navigation: Can move to specific coordinates
        - Manipulation: Can pick up and place objects
        - Perception: Can detect objects and their locations
        - Speech: Can speak to users

        Current Context:
        {json.dumps(context, indent=2)}

        Command: {command}

        Generate a plan as a JSON array of subtasks. Each subtask must have:
        - type: NAVIGATE, PICK_UP, PLACE, SPEAK, or GREET
        - parameters: Required parameters for the subtask
        - preconditions: Conditions that must be true before executing
        - expected_effects: Effects that should result from execution

        Ensure the plan is executable given the robot's capabilities and current state.
        """

    def get_system_prompt(self) -> str:
        """Get system prompt for consistent behavior"""
        return """
        You are a precise cognitive planner for a humanoid robot. Always return valid JSON.
        Ensure plans are executable with the robot's capabilities.
        Include all necessary parameters for each subtask.
        Consider preconditions and expected effects for each action.
        """

    def validate_plan_structure(self, plan: List[Dict]) -> bool:
        """Validate the structure of the generated plan"""
        if not isinstance(plan, list):
            return False

        required_fields = ['type', 'parameters']

        for task in plan:
            if not isinstance(task, dict):
                return False

            for field in required_fields:
                if field not in task:
                    return False

            if task['type'] not in ['NAVIGATE', 'PICK_UP', 'PLACE', 'SPEAK', 'GREET']:
                return False

        return True

    async def refine_plan(self, command: str, context: Dict,
                         current_plan: List[Dict], feedback: str) -> Optional[List[Dict]]:
        """Refine plan based on validation feedback"""
        try:
            refinement_prompt = f"""
            You are a cognitive planner for a humanoid robot. Refine the following plan based on feedback.

            Original Command: {command}

            Current Context:
            {json.dumps(context, indent=2)}

            Current Plan:
            {json.dumps(current_plan, indent=2)}

            Feedback: {feedback}

            Generate an improved plan that addresses the feedback while still accomplishing the original command.
            Return as valid JSON.
            """

            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            plan_text = response.choices[0].message.content
            refined_plan = json.loads(plan_text)

            return refined_plan

        except Exception as e:
            self.get_logger().error(f'Error refining plan: {str(e)}')
            return current_plan  # Return original plan if refinement fails

    def validate_plan_with_robot_capabilities(self, plan: List[Dict], context: Dict) -> Tuple[bool, str]:
        """Validate plan against robot capabilities"""
        robot_caps = context.get('environment', {}).get('robot_capabilities', {})

        for i, task in enumerate(plan):
            task_type = task['type']

            if task_type == 'PICK_UP':
                # Check if robot can manipulate
                if not robot_caps.get('manipulation_enabled', False):
                    return False, f'Task {i}: Robot manipulation not enabled'

                # Check payload limit
                obj_weight = task['parameters'].get('weight', 0.1)  # Default weight
                if obj_weight > robot_caps.get('max_payload', 2.0):
                    return False, f'Task {i}: Object too heavy (max {robot_caps["max_payload"]}kg)'

            elif task_type == 'NAVIGATE':
                # Check if navigation is enabled
                if not robot_caps.get('navigation_enabled', False):
                    return False, f'Task {i}: Robot navigation not enabled'

            elif task_type in ['PLACE', 'PICK_UP']:
                # Check reach constraints
                target_height = task['parameters'].get('height', 0.8)  # Default height
                if target_height > robot_caps.get('max_reach', 1.0):
                    return False, f'Task {i}: Target location out of reach (max {robot_caps["max_reach"]}m)'

        return True, "Plan is valid"

# Example usage with validation
def create_robot_validation_function(robot_node):
    """Create a validation function that checks against actual robot state"""
    def validate_plan(plan, context):
        # This would integrate with the actual robot's state and capabilities
        return robot_node.validate_plan_with_actual_state(plan, context)

    return validate_plan
```

Example of a prompt engineering system for better LLM performance:

```python
class PromptEngineer:
    def __init__(self):
        self.templates = {
            'navigation': """Plan navigation subtasks for: {command}
            Consider: current location {current_pos}, target {target}, obstacles {obstacles}
            Output: [{{"type": "NAVIGATE", "params": {{"x": float, "y": float, "theta": float}}}}]""",

            'manipulation': """Plan manipulation subtasks for: {command}
            Consider: object {object}, current pose {pose}, robot capabilities {caps}
            Output: [{{"type": "PICK_UP", "params": {{"object_id": str, "grasp_type": str}}}},
                    {{"type": "PLACE", "params": {{"location": str, "orientation": list}}}}]""",

            'complex_task': """Decompose complex task: {command}
            Environment: {env_context}
            Robot state: {robot_state}
            Break into: [high_level_goals] -> [mid_level_tasks] -> [primitive_actions]
            Ensure each step is achievable and ordered correctly."""
        }

        self.examples = {
            'navigation': [
                {
                    'input': 'Go to the kitchen',
                    'output': '[{"type": "NAVIGATE", "params": {"x": 2.5, "y": 1.0, "theta": 0.0}}]'
                },
                {
                    'input': 'Move to the table near the window',
                    'output': '[{"type": "NAVIGATE", "params": {"x": 3.2, "y": 2.1, "theta": 1.57}}]'
                }
            ],
            'manipulation': [
                {
                    'input': 'Pick up the red cup',
                    'output': '[{"type": "PICK_UP", "params": {"object_id": "cup_001", "grasp_type": "top"}}]'
                }
            ]
        }

    def create_contextual_prompt(self, task_type: str, command: str, context: Dict) -> str:
        """Create a contextual prompt based on task type and context"""
        template = self.templates.get(task_type, self.templates['complex_task'])

        # Add few-shot examples if available
        examples_text = ""
        if task_type in self.examples:
            examples = self.examples[task_type][:2]  # Use first 2 examples
            examples_text = "\nExamples:\n"
            for ex in examples:
                examples_text += f"Input: {ex['input']}\nOutput: {ex['output']}\n"

        # Format the prompt with context
        prompt = template.format(
            command=command,
            **context
        )

        return examples_text + prompt + "\n\nOutput:"

    def optimize_for_reliability(self, prompt: str) -> str:
        """Add reliability-focused instructions to the prompt"""
        reliability_additions = """

        IMPORTANT:
        - Only include actions that are physically possible
        - Verify all parameters are within robot capabilities
        - Include error handling considerations
        - Ensure logical sequence of actions
        - Use precise, measurable parameters
        - Output valid JSON format only
        """

        return prompt + reliability_additions
```

## Exercises

1. **Plan Generation**: Implement an LLM-based planner that can decompose complex household tasks into robot actions.

2. **Context Integration**: Enhance the LLM planner to incorporate real-time perception data for grounding language in the physical world.

3. **Error Recovery**: Design a system where the LLM can modify plans when execution fails or unexpected situations arise.

4. **Multi-Modal Planning**: Extend the planner to handle commands that combine language with visual input or demonstrations.

## Summary

Cognitive planning with LLMs represents a powerful approach to bridging natural language understanding with robotic action execution. By leveraging the reasoning capabilities of large language models, robots can interpret high-level human commands and decompose them into executable action sequences. Success in this domain requires careful attention to prompt engineering, context integration, and validation of generated plans against robot capabilities. As we continue our exploration of Physical AI, LLM-based cognitive planning will serve as a crucial component for enabling natural and intuitive human-robot interaction, allowing robots to understand and execute complex tasks expressed in natural language.