"""
Sample prompts for different content generation tasks.
These can be customized based on specific course needs.
"""

# Summary Generation Prompts
SUMMARY_PROMPTS = {
    "general": """Generate a comprehensive summary of the following course material.
    
Include:
- Main concepts and key takeaways
- Important definitions and terminology
- Relationships between concepts
- Practical applications or examples

Material:
{context}

Provide a clear, structured summary that helps students understand and retain the information.""",

    "exam_prep": """Create an exam preparation summary of the following material.

Focus on:
- Critical concepts likely to be tested
- Key formulas, algorithms, or processes
- Common pitfalls and misconceptions
- Important examples and use cases

Material:
{context}

Format the summary to help students prepare effectively for assessments.""",

    "quick_review": """Generate a concise quick-review summary of this material.

Include only:
- Essential concepts (bullet points)
- Key terms with brief definitions
- Critical facts to remember

Material:
{context}

Keep it brief and focused for quick revision."""
}

# Question Generation Prompts
QUESTION_PROMPTS = {
    "mcq_basic": """Generate {num} multiple choice questions testing basic understanding.

Guidelines:
- Test fundamental concepts and definitions
- Create clear, unambiguous questions
- Provide 4 options with one correct answer
- Include brief explanations
- Avoid trick questions

Material:
{context}""",

    "mcq_application": """Generate {num} application-based multiple choice questions.

Guidelines:
- Test ability to apply concepts to new scenarios
- Include realistic problem-solving situations
- Create plausible distractors
- Provide detailed explanations

Material:
{context}""",

    "conceptual": """Generate {num} conceptual questions that test deep understanding.

Focus on:
- "Why" and "How" questions
- Comparing and contrasting concepts
- Explaining relationships
- Predicting outcomes

Material:
{context}"""
}

# RAG Answer Prompts
RAG_ANSWER_PROMPTS = {
    "detailed": """You are a knowledgeable teaching assistant for MISM students at CMU.

Context from course materials:
{context}

Student Question: {query}

Provide a detailed, accurate answer that:
1. Directly addresses the question
2. Uses information from the context
3. Explains underlying concepts
4. Includes relevant examples
5. Suggests related topics to explore

If the context doesn't contain sufficient information, clearly state what's missing.""",

    "concise": """Answer the following question based on the course material context.

Context:
{context}

Question: {query}

Provide a clear, concise answer (2-3 sentences) that directly addresses the question.""",

    "socratic": """You are a Socratic teaching assistant helping students learn through guided discovery.

Context:
{context}

Student Question: {query}

Instead of directly answering, help the student think through the problem by:
1. Asking clarifying questions
2. Pointing to relevant concepts in the material
3. Guiding them toward the answer
4. Encouraging critical thinking"""
}

# Evaluation Prompts
EVALUATION_PROMPTS = {
    "answer_quality": """Evaluate the quality of this answer to a student's question.

Question: {query}
Answer: {answer}
Reference Material: {context}

Rate the answer on:
1. Accuracy (0-10): Is the information correct?
2. Completeness (0-10): Does it fully address the question?
3. Clarity (0-10): Is it easy to understand?
4. Relevance (0-10): Is it on-topic?

Provide scores and brief justification."""
}


def get_prompt(category: str, prompt_type: str, **kwargs) -> str:
    """
    Get a formatted prompt.
    
    Args:
        category: Category of prompt (summary, question, rag_answer, evaluation)
        prompt_type: Type within the category
        **kwargs: Variables to format into the prompt
        
    Returns:
        Formatted prompt string
    """
    prompts = {
        "summary": SUMMARY_PROMPTS,
        "question": QUESTION_PROMPTS,
        "rag_answer": RAG_ANSWER_PROMPTS,
        "evaluation": EVALUATION_PROMPTS
    }
    
    if category not in prompts:
        raise ValueError(f"Unknown category: {category}")
    
    if prompt_type not in prompts[category]:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return prompts[category][prompt_type].format(**kwargs)
