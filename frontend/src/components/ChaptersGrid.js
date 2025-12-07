import React from 'react';
import ChapterCard from './ChapterCard';
import styles from './ChaptersGrid.module.css';

const ChaptersData = [
  {
    part: 1,
    chapters: [
      {
        title: "What is Physical AI",
        path: "/docs/intro",
        description:
          "Introduction to Physical AI and its fundamental concepts.",
        whyRead:
          "Understand the difference between traditional AI and Physical AI, and why embodiment is crucial for creating truly intelligent systems.",
        keyConcepts: [
          "Embodied Intelligence",
          "Sensorimotor Learning",
          "Perception-Action Loop",
          "Morphological Computation",
        ],
      },
      {
        title: "Humanoid Robotics Overview",
        path: "/docs/part-1-foundations/chapter-2-humanoid-robotics-overview",
        description:
          "Explore the field of humanoid robotics and its applications.",
        whyRead:
          "Learn about the current state of humanoid robotics and why these systems represent the next frontier in AI.",
        keyConcepts: [
          "Humanoid Design",
          "Kinematics",
          "Biomechanics",
          "Human-Robot Interaction",
        ],
      },
      {
        title: "Understanding the Physical World",
        path: "/docs/part-1-foundations/chapter-3-understanding-physical-world",
        description:
          "How robots perceive and understand physical environments.",
        whyRead:
          "Master the fundamentals of how robots interpret the physical world through sensors and perception algorithms.",
        keyConcepts: [
          "Physics Simulation",
          "Sensor Fusion",
          "Environment Modeling",
          "Uncertainty Management",
        ],
      },
    ],
  },
  {
    part: 2,
    chapters: [
      {
        title: "ROS 2 Basics",
        path: "/docs/part-2-ros/chapter-4-ros2-basics",
        description: "Introduction to Robot Operating System 2.",
        whyRead:
          "ROS 2 is the de facto standard for robotics development. Master the platform used by researchers and industry worldwide.",
        keyConcepts: [
          "Nodes and Topics",
          "Services",
          "Actions",
          "Launch Files",
        ],
      },
      {
        title: "ROS 2 Packages",
        path: "/docs/part-2-ros/chapter-5-ros2-packages",
        description: "Creating and managing ROS 2 packages.",
        whyRead:
          "Learn to organize your robotics code in reusable packages that follow best practices.",
        keyConcepts: [
          "Package Structure",
          "Dependencies",
          "Building Systems",
          "Reusability",
        ],
      },
      {
        title: "Robot Description Files",
        path: "/docs/part-2-ros/chapter-6-robot-description-files",
        description: "URDF, XACRO, and describing robot kinematics.",
        whyRead:
          "Understand how robots are described digitally, essential for simulation and control.",
        keyConcepts: ["URDF", "XACRO", "Kinematics", "Robot Modeling"],
      },
    ],
  },
  {
    part: 3,
    chapters: [
      {
        title: "Gazebo Simulation Fundamentals",
        path: "/docs/part-3-simulation/chapter-7-gazebo-simulation-fundamentals",
        description: "Creating and simulating robots in Gazebo.",
        whyRead:
          "Simulation is crucial for robotics development. Master Gazebo to test algorithms safely and efficiently.",
        keyConcepts: [
          "Physics Simulation",
          "World Modeling",
          "Sensor Simulation",
          "Simulation-Reality Gap",
        ],
      },
      {
        title: "Visualizing in Unity",
        path: "/docs/part-3-simulation/chapter-8-visualizing-in-unity",
        description: "Using Unity for robot visualization and simulation.",
        whyRead:
          "Learn to create high-quality visualizations for robot systems using Unity, a powerful game engine adapted for robotics.",
        keyConcepts: [
          "3D Visualization",
          "Unity Engine",
          "Realistic Rendering",
          "User Interfaces",
        ],
      },
      {
        title: "Simulated Sensors",
        path: "/docs/part-3-simulation/chapter-9-simulated-sensors",
        description: "Implementing realistic sensors in simulation.",
        whyRead:
          "Realistic sensors in simulation bridge the gap between synthetic and real-world testing.",
        keyConcepts: [
          "Camera Simulation",
          "LiDAR Simulation",
          "IMU Simulation",
          "Sensor Accuracy",
        ],
      },
    ],
  },
  {
    part: 4,
    chapters: [
      {
        title: "Isaac Sim Overview",
        path: "/docs/part-4-isaac/chapter-10-isaac-sim-overview",
        description: "Introduction to NVIDIA Isaac Sim platform.",
        whyRead:
          "Isaac Sim is NVIDIA's cutting-edge simulation platform for AI-powered robots. Learn to leverage GPU-accelerated simulation.",
        keyConcepts: [
          "GPU Acceleration",
          "PhysX Physics",
          "AI Training Environments",
          "Synthetic Data Generation",
        ],
      },
      {
        title: "Isaac ROS",
        path: "/docs/part-4-isaac/chapter-11-isaac-ros",
        description: "Using Isaac ROS packages for perception and navigation.",
        whyRead:
          "Combine the power of ROS 2 with NVIDIA's AI capabilities for advanced robot perception and navigation.",
        keyConcepts: [
          "CUDA Acceleration",
          "Computer Vision",
          "Deep Learning",
          "Perception Pipelines",
        ],
      },
      {
        title: "Navigation2",
        path: "/docs/part-4-isaac/chapter-12-navigation2",
        description: "Advanced navigation for robots.",
        whyRead:
          "Master the latest navigation stack that enables robots to autonomously navigate complex environments.",
        keyConcepts: [
          "Path Planning",
          "Localization",
          "Mapping",
          "Dynamic Obstacle Avoidance",
        ],
      },
    ],
  },
  {
    part: 5,
    chapters: [
      {
        title: "Voice to Action",
        path: "/docs/part-5-vla/chapter-13-voice-to-action",
        description: "Converting natural language to robot actions.",
        whyRead:
          "Enable intuitive human-robot interaction through natural language commands.",
        keyConcepts: [
          "Natural Language Processing",
          "Command Recognition",
          "Semantic Parsing",
          "Action Mapping",
        ],
      },
      {
        title: "Cognitive Planning with LLMs",
        path: "/docs/part-5-vla/chapter-14-cognitive-planning-with-llms",
        description: "Using Large Language Models for robot task planning.",
        whyRead:
          "Leverage LLMs for high-level reasoning and task planning in complex environments.",
        keyConcepts: [
          "Large Language Models",
          "Task Planning",
          "Reasoning",
          "Human-in-the-Loop",
        ],
      },
      {
        title: "Perception and Object Interaction",
        path: "/docs/part-5-vla/chapter-15-perception-and-object-interaction",
        description: "Recognizing and interacting with objects.",
        whyRead:
          "Master the ability for robots to identify and manipulate objects in unstructured environments.",
        keyConcepts: [
          "Object Recognition",
          "Manipulation Planning",
          "Grasp Synthesis",
          "Scene Understanding",
        ],
      },
    ],
  },
  {
    part: 6,
    chapters: [
      {
        title: "Humanoid Kinematics",
        path: "/docs/part-6-humanoid-engineering/chapter-16-humanoid-kinematics",
        description: "Forward and inverse kinematics for humanoid robots.",
        whyRead:
          "Understand the mathematical foundation for how humanoid robots move and position their limbs.",
        keyConcepts: [
          "Forward Kinematics",
          "Inverse Kinematics",
          "Dynamics",
          "Jacobian Matrices",
        ],
      },
      {
        title: "Locomotion and Balance",
        path: "/docs/part-6-humanoid-engineering/chapter-17-locomotion-and-balance",
        description: "Walking and balancing algorithms for humanoid robots.",
        whyRead:
          "Master the complex algorithms required for stable bipedal locomotion.",
        keyConcepts: [
          "Zero Moment Point",
          "Balance Control",
          "Walking Patterns",
          "Stability",
        ],
      },
      {
        title: "Manipulation",
        path: "/docs/part-6-humanoid-engineering/chapter-18-manipulation",
        description: "Arm and hand manipulation techniques.",
        whyRead:
          "Learn how humanoid robots can manipulate objects with the same dexterity as humans.",
        keyConcepts: [
          "Grasp Planning",
          "Manipulation Control",
          "Force Control",
          "Tactile Sensing",
        ],
      },
    ],
  },
  {
    part: 7,
    chapters: [
      {
        title: "Autonomous Humanoid Pipeline",
        path: "/docs/part-7-capstone/chapter-19-autonomous-humanoid-pipeline",
        description: "Complete pipeline for autonomous humanoid operation.",
        whyRead:
          "Integrate all the concepts learned throughout the book into a complete autonomous humanoid system.",
        keyConcepts: [
          "System Integration",
          "Multi-Modal Perception",
          "Behavior Trees",
          "Autonomous Operation",
        ],
      },
      {
        title: "Autonomous Humanoid Pipeline",
        path: "/docs/part-7-capstone/chapter-19-autonomous-humanoid-pipeline",
        description: "Complete pipeline for autonomous humanoid operation.",
        whyRead:
          "Integrate all the concepts learned throughout the book into a complete autonomous humanoid system.",
        keyConcepts: [
          "System Integration",
          "Multi-Modal Perception",
          "Behavior Trees",
          "Autonomous Operation",
        ],
      },
      {
        title: "Autonomous Humanoid Pipeline",
        path: "/docs/part-7-capstone/chapter-19-autonomous-humanoid-pipeline",
        description: "Complete pipeline for autonomous humanoid operation.",
        whyRead:
          "Integrate all the concepts learned throughout the book into a complete autonomous humanoid system.",
        keyConcepts: [
          "System Integration",
          "Multi-Modal Perception",
          "Behavior Trees",
          "Autonomous Operation",
        ],
      },
    ],
  },
];

const ChaptersGrid = () => {
  return (
    <section className={styles.chaptersGrid}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2 className={styles.gridTitle}>Textbook Chapters</h2>
            <p className={styles.gridSubtitle}>Explore the complete guide to Physical AI and Humanoid Robotics</p>
          </div>
        </div>
        
        {ChaptersData.map((partData) => (
          <div key={partData.part} className="row">
            <div className="col col--12">
              <h3 className={styles.partTitle}>Part {partData.part}: {getPartTitle(partData.part)}</h3>
            </div>
            <div className="row">
              {partData.chapters.map((chapter, index) => (
                <ChapterCard key={index} chapter={chapter} part={partData.part} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

// Helper function to get part titles
const getPartTitle = (partNumber) => {
  const partTitles = {
    1: 'Foundations of Physical AI',
    2: 'ROS 2, the Robotic Nervous System',
    3: 'Digital Twins with Gazebo and Unity',
    4: 'NVIDIA Isaac Platform',
    5: 'Vision Language Action (VLA)',
    6: 'Humanoid Robot Engineering',
    7: 'Capstone Project'
  };
  return partTitles[partNumber] || `Part ${partNumber}`;
};

export default ChaptersGrid;