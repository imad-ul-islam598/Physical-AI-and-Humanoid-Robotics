---
sidebar_position: 13
---

# Chapter 13: Voice to Action

## Introduction

Voice to Action represents a critical interface in Physical AI systems, enabling natural human-robot interaction through spoken language. This chapter explores the integration of voice recognition technologies, particularly OpenAI's Whisper, with robotic systems to transform spoken commands into executable robot actions. We'll examine the complete pipeline from speech recognition to intent interpretation and action execution, focusing on the unique challenges of processing voice commands in physical environments with humanoid robots.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the architecture of voice-to-action systems for robotics
- Implement speech recognition using Whisper or similar technologies
- Convert voice commands into robot action intents
- Handle noise and environmental challenges in voice processing
- Design robust voice command vocabularies for robot control

## Key Concepts

- **Voice Recognition**: Converting spoken language to text
- **Whisper**: OpenAI's automatic speech recognition model
- **Intent Classification**: Determining the action requested by the user
- **Command Mapping**: Linking recognized intents to robot actions
- **Noise Robustness**: Handling environmental noise in voice processing
- **Voice Command Grammar**: Structured command formats for robot control

## Technical Explanation

Voice to Action systems for Physical AI involve multiple components working together to transform spoken language into robot actions. The pipeline typically consists of:

1. **Audio Capture**: Collecting speech from the environment using microphones. For humanoid robots, this often involves multiple microphones for noise cancellation and directional audio capture.

2. **Preprocessing**: Filtering and conditioning the audio signal to improve recognition quality. This includes noise reduction, echo cancellation, and audio normalization.

3. **Speech Recognition**: Converting the audio signal to text using models like Whisper. Modern speech recognition models can handle multiple languages and various accents.

4. **Natural Language Processing**: Understanding the meaning and intent behind the recognized text. This involves parsing the command and extracting relevant parameters.

5. **Action Mapping**: Converting the interpreted intent into specific robot commands that can be executed by the robot's control system.

6. **Execution**: Sending commands to the robot's control system to perform the requested action.

Whisper, developed by OpenAI, is particularly well-suited for robotics applications due to its robustness across different audio conditions and its ability to handle multiple languages. The model can be fine-tuned for specific domains and vocabulary, making it more accurate for robot command recognition.

The voice command vocabulary for robots typically follows structured patterns to ensure reliable recognition:

- **Action + Object**: "Move to the table", "Pick up the red ball"
- **Action + Location**: "Go to the kitchen", "Turn left"
- **Action + Parameter**: "Move forward 2 meters", "Turn 90 degrees"

Environmental challenges in voice processing for robotics include:

- **Background Noise**: Mechanical sounds from the robot, environmental noise, etc.
- **Distance and Direction**: The speaker's distance and direction relative to the robot's microphones
- **Echo and Reverberation**: Sound reflections in indoor environments
- **Multiple Speakers**: Distinguishing between commands and other speech in the environment

The system architecture for voice-to-action typically involves:

- **Audio Processing Pipeline**: Real-time audio capture, preprocessing, and noise reduction
- **Speech Recognition Service**: Either local or cloud-based recognition
- **Intent Processing Engine**: Natural language understanding and command parsing
- **Action Execution System**: Integration with the robot's control framework

## Diagrams written as text descriptions

**Diagram 1: Voice to Action Pipeline**
```
Human Speech
     │
     ▼
Audio Capture (Microphones)
     │
     ▼
Preprocessing (Noise Reduction, Normalization)
     │
     ▼
Speech Recognition (Whisper/Other ASR)
     │
     ▼
Text Output ("Move to the kitchen")
     │
     ▼
Natural Language Processing (Intent Classification)
     │
     ▼
Intent + Parameters (Action: Navigate, Location: kitchen)
     │
     ▼
Action Mapping (Navigate to kitchen coordinates)
     │
     ▼
Robot Command Execution
     │
     ▼
Robot Action (Movement to kitchen)
```

**Diagram 2: System Architecture**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Microphone    │───►│ Audio Processing │───►│ Speech Recogn.  │
│   Array         │    │ (Noise Cancel,   │    │ (Whisper)       │
│                 │    │ Echo Cancel)     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Robot Control  │◄───│ Intent Process.  │◄───│ NLP + Command   │
│   System        │    │ (Intent Class.)  │    │ Parsing         │
│ (Navigation,    │    │                  │    │                 │
│ Manipulation)   │    └──────────────────┘    └─────────────────┘
└─────────────────┘
```

**Diagram 3: Command Processing Flow**
```
Voice Command: "Please go to the table near the window"
         │
         ▼
Tokenization: [Please, go, to, the, table, near, the, window]
         │
         ▼
Part-of-Speech Tagging: [Adv, Verb, Prep, Det, Noun, Prep, Det, Noun]
         │
         ▼
Named Entity Recognition: Action=go, Object=table, Location=near window
         │
         ▼
Semantic Parsing: {action: "navigate", target: "table", modifier: "near_window"}
         │
         ▼
Command Execution: Execute navigation to table location
```

## Code Examples

Here's an example of a voice-to-action system using Whisper for speech recognition:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import AudioData
import speech_recognition as sr
import openai
import json
import threading
import queue
import time

class VoiceToActionNode(Node):
    def __init__(self):
        super().__init__('voice_to_action')

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Set energy threshold for silence detection
        self.recognizer.energy_threshold = 4000

        # Audio queue for processing
        self.audio_queue = queue.Queue()

        # Publishers for different robot commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.action_pub = self.create_publisher(String, '/robot_action', 10)

        # Voice command vocabulary
        self.command_map = {
            'move forward': 'move_forward',
            'move backward': 'move_backward',
            'turn left': 'turn_left',
            'turn right': 'turn_right',
            'stop': 'stop',
            'go to the kitchen': 'navigate_kitchen',
            'go to the table': 'navigate_table',
            'pick up the object': 'pick_up',
            'wave': 'wave',
            'dance': 'dance'
        }

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.audio_thread.start()

        # Timer for continuous listening
        self.listen_timer = self.create_timer(0.1, self.check_audio)

        self.get_logger().info('Voice to Action node initialized')

    def check_audio(self):
        """Check for audio in the queue and process it"""
        try:
            while not self.audio_queue.empty():
                audio_data = self.audio_queue.get_nowait()
                self.process_audio_data(audio_data)
        except queue.Empty:
            pass

    def process_audio(self):
        """Continuously listen for audio and put it in the queue"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        while rclpy.ok():
            try:
                with self.microphone as source:
                    self.get_logger().info('Listening for voice commands...')
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                    # Put audio in queue for processing
                    self.audio_queue.put(audio)

            except sr.WaitTimeoutError:
                # No audio detected, continue listening
                continue
            except sr.UnknownValueError:
                self.get_logger().info('Could not understand audio')
            except sr.RequestError as e:
                self.get_logger().error(f'Recognition error: {str(e)}')

    def process_audio_data(self, audio):
        """Process audio data and convert to robot actions"""
        try:
            # Use Whisper for speech recognition (or other ASR)
            # For this example, we'll use Google's speech recognition
            # In practice, you'd use Whisper's API or local model
            text = self.recognizer.recognize_google(audio)
            self.get_logger().info(f'Recognized: {text}')

            # Process the recognized text to determine action
            action = self.parse_voice_command(text.lower())

            if action:
                self.execute_action(action)
            else:
                self.get_logger().info(f'Unknown command: {text}')

        except sr.UnknownValueError:
            self.get_logger().info('Could not understand audio')
        except sr.RequestError as e:
            self.get_logger().error(f'Recognition error: {str(e)}')

    def parse_voice_command(self, text):
        """Parse voice command and return corresponding action"""
        # Simple command matching (in practice, use NLP for more complex parsing)
        for command_phrase, action in self.command_map.items():
            if command_phrase in text:
                return action

        # Handle commands with parameters
        if 'move' in text and 'meter' in text:
            # Extract distance from command like "move forward 2 meters"
            words = text.split()
            for i, word in enumerate(words):
                if word.isdigit() and i > 0:
                    distance = float(word)
                    if 'forward' in text:
                        return f'move_forward_{distance}'
                    elif 'backward' in text:
                        return f'move_backward_{distance}'

        return None

    def execute_action(self, action):
        """Execute the determined action"""
        self.get_logger().info(f'Executing action: {action}')

        if action == 'move_forward':
            self.move_robot(0.5, 0.0)  # Move forward at 0.5 m/s
        elif action == 'move_backward':
            self.move_robot(-0.5, 0.0)  # Move backward at 0.5 m/s
        elif action == 'turn_left':
            self.move_robot(0.0, 0.5)  # Turn left at 0.5 rad/s
        elif action == 'turn_right':
            self.move_robot(0.0, -0.5)  # Turn right at 0.5 rad/s
        elif action == 'stop':
            self.move_robot(0.0, 0.0)  # Stop
        elif action.startswith('move_forward_'):
            # Handle parameterized movement
            try:
                distance = float(action.split('_')[2])
                self.move_distance(0.5, distance)  # Move forward at 0.5 m/s for 'distance' meters
            except ValueError:
                self.get_logger().error(f'Invalid distance in action: {action}')
        elif action.startswith('move_backward_'):
            # Handle parameterized movement
            try:
                distance = float(action.split('_')[2])
                self.move_distance(-0.5, distance)  # Move backward at 0.5 m/s for 'distance' meters
            except ValueError:
                self.get_logger().error(f'Invalid distance in action: {action}')
        elif action in ['navigate_kitchen', 'navigate_table']:
            # Publish navigation command
            nav_msg = String()
            nav_msg.data = action
            self.action_pub.publish(nav_msg)
        else:
            # For other actions, publish to action topic
            action_msg = String()
            action_msg.data = action
            self.action_pub.publish(action_msg)

    def move_robot(self, linear_x, angular_z):
        """Send velocity command to robot"""
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd)

    def move_distance(self, speed, distance):
        """Move robot a specific distance"""
        # Calculate time needed (simple approach)
        if speed != 0:
            time_needed = abs(distance) / abs(speed)

            # Send command
            self.move_robot(speed, 0.0)

            # Wait for the required time (in practice, use feedback from robot)
            time.sleep(time_needed)

            # Stop
            self.move_robot(0.0, 0.0)

def main(args=None):
    rclpy.init(args=args)
    voice_node = VoiceToActionNode()

    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        voice_node.get_logger().info('Shutting down Voice to Action node...')
    finally:
        voice_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Example of using OpenAI's Whisper API for speech recognition:

```python
import openai
import os
import requests
import io
import wave
import pyaudio
import numpy as np
from datetime import datetime

class WhisperVoiceProcessor:
    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key
        else:
            # Try to get API key from environment
            openai.api_key = os.getenv('OPENAI_API_KEY')

        # Audio parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 5

        self.audio = pyaudio.PyAudio()

    def record_audio(self):
        """Record audio from microphone"""
        self.get_logger().info('Recording audio...')

        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        # Save to WAV file
        filename = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        return filename

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio using Whisper API"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return None

    def process_voice_command(self, audio_file_path):
        """Process voice command from audio file"""
        # Transcribe the audio
        transcript = self.transcribe_audio(audio_file_path)

        if transcript:
            print(f"Transcribed: {transcript}")
            # Clean up temporary file
            os.remove(audio_file_path)
            return transcript.strip().lower()
        else:
            # Clean up temporary file even if transcription failed
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            return None

    def cleanup(self):
        """Clean up audio resources"""
        self.audio.terminate()
```

Example of a more sophisticated intent classification system:

```python
import re
from typing import Dict, List, Tuple, Optional

class IntentClassifier:
    def __init__(self):
        # Define command patterns and their corresponding actions
        self.patterns = {
            'move_forward': [
                r'move forward(?: by)? (\d+(?:\.\d+)?) meters?',
                r'go forward(?: by)? (\d+(?:\.\d+)?) meters?',
                r'forward (\d+(?:\.\d+)?) meters?'
            ],
            'move_backward': [
                r'move backward(?: by)? (\d+(?:\.\d+)?) meters?',
                r'go backward(?: by)? (\d+(?:\.\d+)?) meters?',
                r'backward (\d+(?:\.\d+)?) meters?'
            ],
            'turn': [
                r'turn (left|right)(?: by)? (\d+(?:\.\d+)?) degrees?',
                r'rotate (left|right)(?: by)? (\d+(?:\.\d+)?) degrees?',
                r'pivot (left|right)'
            ],
            'navigate': [
                r'go to the (kitchen|bedroom|living room|office|bathroom)',
                r'move to the (kitchen|bedroom|living room|office|bathroom)',
                r'go to (kitchen|bedroom|living room|office|bathroom)'
            ],
            'grasp': [
                r'pick up the (.+)',
                r'grab the (.+)',
                r'take the (.+)',
                r'get the (.+)'
            ],
            'stop': [
                r'stop',
                r'halt',
                r'freeze'
            ],
            'greet': [
                r'hello',
                r'hi',
                r'hey',
                r'greetings'
            ]
        }

    def classify_intent(self, text: str) -> Optional[Tuple[str, Dict]]:
        """
        Classify the intent of the given text and extract parameters.
        Returns a tuple of (intent_name, parameters) or None if no match.
        """
        text = text.strip().lower()

        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    # Extract parameters
                    params = {}
                    if match.groups():
                        if intent == 'turn':
                            # Handle turn command: direction and angle
                            params['direction'] = match.group(1)
                            if len(match.groups()) > 1:
                                params['angle'] = float(match.group(2))
                            else:
                                params['angle'] = 90.0  # default turn angle
                        elif intent in ['move_forward', 'move_backward']:
                            # Handle movement command: distance
                            params['distance'] = float(match.group(1))
                        elif intent == 'navigate':
                            # Handle navigation command: location
                            params['location'] = match.group(1)
                        elif intent == 'grasp':
                            # Handle grasp command: object
                            params['object'] = match.group(1)

                    return intent, params

        return None

class VoiceCommandProcessor:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.command_history = []

    def process_command(self, voice_text: str) -> Optional[Dict]:
        """
        Process voice command and return structured action.
        """
        result = self.intent_classifier.classify_intent(voice_text)

        if result:
            intent, params = result
            action = {
                'intent': intent,
                'parameters': params,
                'original_text': voice_text,
                'timestamp': datetime.now().isoformat()
            }

            self.command_history.append(action)
            return action
        else:
            # If no pattern matches, try to determine if it's a general request
            if any(word in voice_text.lower() for word in ['please', 'could you', 'can you', 'help']):
                # This might be a complex request that requires more sophisticated NLP
                return {
                    'intent': 'unknown_complex',
                    'parameters': {'text': voice_text},
                    'original_text': voice_text,
                    'timestamp': datetime.now().isoformat()
                }

            return None

# Example usage
def main():
    processor = VoiceCommandProcessor()

    # Test commands
    test_commands = [
        "Move forward 2.5 meters",
        "Turn left by 45 degrees",
        "Go to the kitchen",
        "Pick up the red ball",
        "Stop immediately",
        "Hello robot"
    ]

    for cmd in test_commands:
        result = processor.process_command(cmd)
        if result:
            print(f"Command: '{cmd}' -> Intent: {result['intent']}, Params: {result['parameters']}")
        else:
            print(f"Command: '{cmd}' -> No match")

if __name__ == '__main__':
    main()
```

## Exercises

1. **Voice Recognition**: Implement a voice recognition system using Whisper that can handle background noise from a moving robot.

2. **Intent Classification**: Create a more sophisticated intent classification system that can handle complex voice commands with multiple parameters.

3. **Command Vocabulary**: Design a voice command vocabulary specifically for humanoid robot control and implement a parser for it.

4. **Robustness Testing**: Test the voice-to-action system under various noise conditions and evaluate its performance.

## Summary

Voice to Action systems provide a natural and intuitive interface for human-robot interaction, enabling users to control Physical AI systems through spoken language. The integration of modern speech recognition technologies like Whisper with robotic control systems creates opportunities for more accessible and user-friendly robot interfaces. Success in this domain requires careful attention to audio processing, intent classification, and the unique challenges of operating in noisy physical environments. As we continue our exploration of Physical AI, voice interfaces will serve as a crucial component for enabling natural human-robot collaboration.