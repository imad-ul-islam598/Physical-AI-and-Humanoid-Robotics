import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import TextbookOverview from '@site/src/components/TextbookOverview';
import ChaptersGrid from '@site/src/components/ChaptersGrid';

import Heading from '@theme/Heading';
import styles from './index.module.css';
import ChatBot from '../components/Chatbot';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Start Reading - 15min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="A comprehensive textbook on embodied artificial intelligence and humanoid robotics">
      <HomepageHeader />
      <main>
        <ChatBot/>
        <TextbookOverview />
        <ChaptersGrid />
      </main>
    </Layout>
  );
}
