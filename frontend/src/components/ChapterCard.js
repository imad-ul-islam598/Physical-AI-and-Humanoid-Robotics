import React from 'react';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import styles from './ChapterCard.module.css';

const ChapterCard = ({ chapter, part }) => {
  return (
    <div className={clsx('col col--4', styles.chapterCard)}>
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <h3 className={styles.cardTitle}>
            <Link to={chapter.path}>
              {chapter.title}
            </Link>
          </h3>
          <span className={styles.partTag}>Part {part}</span>
        </div>
        <div className={styles.cardBody}>
          <p className={styles.description}>{chapter.description}</p>
          <div className={styles.whyRead}>
            <h4>Why Read This?</h4>
            <p>{chapter.whyRead}</p>
          </div>
          <div className={styles.learningObjectives}>
            <h4>Key Concepts</h4>
            <ul>
              {chapter.keyConcepts.map((concept, index) => (
                <li key={index}>{concept}</li>
              ))}
            </ul>
          </div>
        </div>
        <div className={styles.cardFooter}>
          <Link className="button button--primary" to={chapter.path}>
            Read Chapter
          </Link>
        </div>
      </div>
    </div>
  );
};

export default ChapterCard;