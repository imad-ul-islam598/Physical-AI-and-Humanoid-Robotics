# Physical AI and Humanoid Robotics Textbook

This repository contains the complete source for the "Physical AI and Humanoid Robotics" textbook, created as part of the Spec-Driven Hackathon. The textbook follows Docusaurus MDX format and covers all aspects of Physical AI and humanoid robotics development.

## Book Structure

The textbook is organized into 7 parts with 19 chapters:

### Part 1: Foundations of Physical AI
- Chapter 1: What is Physical AI
- Chapter 2: Humanoid Robotics Overview
- Chapter 3: Understanding the Physical World

### Part 2: ROS 2, the Robotic Nervous System
- Chapter 4: ROS 2 Basics
- Chapter 5: ROS 2 Packages
- Chapter 6: Robot Description Files

### Part 3: Digital Twins with Gazebo and Unity
- Chapter 7: Gazebo Simulation Fundamentals
- Chapter 8: Visualizing in Unity
- Chapter 9: Simulated Sensors

### Part 4: NVIDIA Isaac Platform
- Chapter 10: Isaac Sim Overview
- Chapter 11: Isaac ROS
- Chapter 12: Navigation 2

### Part 5: Vision Language Action (VLA)
- Chapter 13: Voice to Action
- Chapter 14: Cognitive Planning with LLMs
- Chapter 15: Perception and Object Interaction

### Part 6: Humanoid Robot Engineering
- Chapter 16: Humanoid Kinematics
- Chapter 17: Locomotion and Balance
- Chapter 18: Manipulation

### Part 7: Capstone Project
- Chapter 19: Autonomous Humanoid Pipeline

## Project Structure

```
Physical AI and Humanoid Robotics/
├── .specify/                 # Spec-Driven Development artifacts
│   ├── memory/              # Project constitution
│   ├── scripts/             # Automation scripts
│   └── templates/           # Template files
├── frontend/                # Docusaurus documentation site
│   ├── docs/               # Textbook chapters
│   ├── src/                # Custom components
│   ├── static/             # Static assets
│   ├── blog/               # Blog content
│   ├── sidebars.js         # Navigation configuration
│   └── docusaurus.config.js # Site configuration
├── CLAUDE.md               # Claude Code rules and instructions
└── README.md               # This file
```

## Getting Started

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run start
```

The textbook will be available at http://localhost:3000/

## Development

The textbook follows the Docusaurus MDX format with the following structure for each chapter:

```markdown
---
sidebar_position: X
---

# Chapter X: Title

## Introduction
...

## Learning Objectives
...

## Key Concepts
...

## Technical Explanation
...

## Diagrams written as text descriptions
...

## Code Examples
...

## Exercises
...

## Summary
...
```

## Contributing

This textbook was developed following the Spec-Driven Development methodology with the following principles:

- Clear, structured, and technically accurate content
- Beginner-friendly explanations with deeper technical content
- Academic tone with short sentences and clean formatting
- Proper Docusaurus MDX formatting
- Python, ROS 2, and C++ code examples where needed
- Diagrams described in text
- Exercises and summaries in each chapter

## License

This textbook is provided as part of the Spec-Driven Hackathon for educational purposes.