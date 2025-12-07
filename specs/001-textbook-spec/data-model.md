# Data Model: Physical AI and Humanoid Robotics Textbook

## Content Entities

### Textbook Chapter
- **name**: String - Title of the chapter
- **part**: String - Which part the chapter belongs to (1-7)
- **chapter_number**: Integer - Sequential number (1-19)
- **learning_objectives**: List of String - What students should learn
- **key_concepts**: List of String - Important concepts covered
- **technical_explanation**: String - Detailed technical content
- **diagrams**: List of String - Text descriptions of diagrams
- **code_examples**: List of CodeExample - Practical examples
- **exercises**: List of Exercise - Practice problems
- **summary**: String - Chapter summary
- **dependencies**: List of String - Prerequisite knowledge/chapters

### CodeExample
- **language**: String - Programming language (Python, C++, etc.)
- **code**: String - The actual code with proper formatting
- **description**: String - What the code demonstrates
- **purpose**: String - How it relates to chapter content
- **complexity**: Enum (beginner, intermediate, advanced)

### Exercise
- **type**: Enum (conceptual, practical, implementation)
- **difficulty**: Enum (beginner, intermediate, advanced)
- **question**: String - The exercise question
- **expected_outcome**: String - What the exercise should teach
- **hints**: List of String - Guidance for solving
- **solution**: String - Suggested solution (for educator use)

### Part
- **name**: String - Name of the part (e.g., "Foundations of Physical AI")
- **chapters**: List of Chapter - Chapters belonging to this part
- **overview**: String - Summary of the part's content
- **prerequisites**: List of String - Knowledge needed before this part

### Code Language
- **name**: String - Language name (Python, C++, JavaScript, etc.)
- **version**: String - Recommended version
- **purpose**: String - What the language is used for in the textbook
- **examples_count**: Integer - Number of examples using this language

## Content Relationships

### Chapter Dependencies
- Each Chapter may depend on other Chapters (prerequisites)
- Part contains multiple Chapters
- Chapter contains multiple CodeExamples and Exercises
- CodeExample belongs to exactly one Chapter
- Exercise belongs to exactly one Chapter

### Content Progression
- Chapters are sequentially ordered within each Part
- Parts are ordered to build upon previous knowledge
- Code examples increase in complexity throughout the textbook
- Exercises build upon concepts introduced in earlier chapters

## Validation Rules

### Chapter Requirements
- Each Chapter MUST have all required sections: Introduction, Learning Objectives, Key Concepts, Technical Explanation, Diagrams, Code Examples, Exercises, and Summary
- Chapter title MUST be descriptive and accurate
- Learning objectives MUST be specific and measurable
- Code examples MUST be technically accurate and runnable
- Exercises MUST have appropriate difficulty for target audience

### Content Consistency
- All terminology MUST be consistent across chapters
- Academic tone MUST be maintained throughout
- Target audience requirements MUST be met (age 16+, Python basics)
- No hallucinated technology MUST be present
- All examples MUST be verifiable and accurate

### Structural Requirements
- Each Part MUST contain the correct number of Chapters as defined in constitution
- Chapter numbering MUST be sequential within each Part
- All cross-references MUST be accurate
- Navigation links MUST work correctly in Docusaurus