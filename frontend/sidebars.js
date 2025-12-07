// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Part 1: Foundations of Physical AI',
      items: [
        'intro', // Chapter 1: What is Physical AI
        'part-1-foundations/chapter-2-humanoid-robotics-overview',
        'part-1-foundations/chapter-3-understanding-physical-world'
      ],
    },
    {
      type: 'category',
      label: 'Part 2: ROS 2, the Robotic Nervous System',
      items: [
        'part-2-ros/chapter-4-ros2-basics',
        'part-2-ros/chapter-5-ros2-packages',
        'part-2-ros/chapter-6-robot-description-files'
      ],
    },
    {
      type: 'category',
      label: 'Part 3: Digital Twins with Gazebo and Unity',
      items: [
        'part-3-simulation/chapter-7-gazebo-simulation-fundamentals',
        'part-3-simulation/chapter-8-visualizing-in-unity',
        'part-3-simulation/chapter-9-simulated-sensors'
      ],
    },
    {
      type: 'category',
      label: 'Part 4: NVIDIA Isaac Platform',
      items: [
        'part-4-isaac/chapter-10-isaac-sim-overview',
        'part-4-isaac/chapter-11-isaac-ros',
        'part-4-isaac/chapter-12-navigation2'
      ],
    },
    {
      type: 'category',
      label: 'Part 5: Vision Language Action (VLA)',
      items: [
        'part-5-vla/chapter-13-voice-to-action',
        'part-5-vla/chapter-14-cognitive-planning-with-llms',
        'part-5-vla/chapter-15-perception-and-object-interaction'
      ],
    },
    {
      type: 'category',
      label: 'Part 6: Humanoid Robot Engineering',
      items: [
        'part-6-humanoid-engineering/chapter-16-humanoid-kinematics',
        'part-6-humanoid-engineering/chapter-17-locomotion-and-balance',
        'part-6-humanoid-engineering/chapter-18-manipulation'
      ],
    },
    {
      type: 'category',
      label: 'Part 7: Capstone Project',
      items: [
        'part-7-capstone/chapter-19-autonomous-humanoid-pipeline'
      ],
    },
  ],
};

export default sidebars;
