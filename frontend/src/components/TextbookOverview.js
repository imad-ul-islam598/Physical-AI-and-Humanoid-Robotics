import React from 'react';
import clsx from 'clsx';
import styles from './TextbookOverview.module.css';

const TextbookOverview = () => {
  return (
    <section className={styles.textbookOverview}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2>Why Read This Textbook?</h2>
            <p className={styles.subtitle}>
              Physical AI and Humanoid Robotics represents the next frontier in artificial intelligence, where machines don't just think but also interact with and understand the physical world.
            </p>
          </div>
        </div>
        
        <div className="row">
          <div className="col col--4">
            <div className={styles.featureCard}>
              <h3>Embodied Intelligence</h3>
              <p>Learn how intelligence emerges through the interaction between an agent's body, its sensors and actuators, and the environment. Understand why embodiment is crucial for creating truly intelligent systems.</p>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.featureCard}>
              <h3>Practical Applications</h3>
              <p>Explore real-world applications of Physical AI in humanoid robotics, autonomous systems, and human-robot interaction. See how these technologies are transforming industries and daily life.</p>
            </div>
          </div>
          <div className="col col--4">
            <div className={styles.featureCard}>
              <h3>Cutting-edge Technology</h3>
              <p>Master state-of-the-art tools and platforms like ROS 2, Gazebo, Unity, and NVIDIA Isaac. Get hands-on experience with the same technologies used in industry and research.</p>
            </div>
          </div>
        </div>
        
        <div className="row">
          <div className="col col--12">
            <div className={styles.whoShouldRead}>
              <h3>Who Should Read This Textbook?</h3>
              <ul>
                <li>Computer science and engineering students looking to specialize in robotics and AI</li>
                <li>Researchers and professionals in robotics, AI, and autonomous systems</li>
                <li>Developers and engineers working with humanoid robots or physical AI systems</li>
                <li>Anyone interested in the future of human-robot interaction and embodied intelligence</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default TextbookOverview;