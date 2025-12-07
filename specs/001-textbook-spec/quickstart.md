# Quickstart Guide: Physical AI and Humanoid Robotics Textbook

## Project Setup

### Prerequisites
- Node.js 18+ installed
- npm or yarn package manager
- Basic knowledge of Markdown and Git
- Python 3.8+ for code example testing (optional)

### Initial Setup
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Start the development server:
   ```bash
   npm run start
   ```
   The textbook will be available at http://localhost:3000/

## Content Creation Workflow

### Creating a New Chapter
1. Navigate to the appropriate part directory in `frontend/docs/`
2. Create a new MDX file following the naming convention: `chapter-X-topic.md`
3. Ensure the file includes all required sections:
   ```markdown
   ---
   sidebar_position: X
   ---

   # Chapter X: [Title]

   ## Introduction
   [Content here]

   ## Learning Objectives
   - [Objective 1]
   - [Objective 2]

   ## Key Concepts
   - [Concept 1]
   - [Concept 2]

   ## Technical Explanation
   [Detailed explanation]

   ## Diagrams written as text descriptions
   [Diagram descriptions]

   ## Code Examples
   [Code examples with explanations]

   ## Exercises
   [Exercise questions]

   ## Summary
   [Chapter summary]
   ```

### Adding Code Examples
- Use proper syntax highlighting: `python`, `cpp`, `javascript`, etc.
- Include comments explaining complex concepts
- Ensure all examples are technically accurate
- Test examples when possible

### Adding Diagrams
- Describe diagrams in text format
- Use ASCII art or text-based representations when helpful
- Explain the purpose and key elements of each diagram

### Creating Exercises
- Include exercises of varying difficulty
- Provide hints for complex problems
- Consider conceptual, practical, and implementation exercises

## Navigation and Structure

### Sidebar Configuration
- Update `sidebars.js` to include new chapters
- Organize chapters by part structure
- Maintain proper ordering for sequential learning

### Cross-References
- Link between related chapters when appropriate
- Use relative links for internal navigation
- Maintain consistency in terminology across chapters

## Content Standards

### Writing Style
- Maintain academic tone throughout
- Use short, clear sentences
- Provide beginner-friendly explanations before technical depth
- Ensure technical accuracy in all content

### Target Audience
- Write for students aged 16+ with Python basics
- Avoid assuming advanced robotics knowledge
- Build concepts progressively
- Include practical applications

## Testing and Validation

### Local Testing
- Use `npm run start` to preview changes
- Verify all links and navigation work correctly
- Check that code examples render properly
- Ensure exercises and summaries are complete

### Content Verification
- Validate all technical claims
- Ensure code examples are runnable
- Verify diagrams are properly described
- Confirm exercises have appropriate difficulty