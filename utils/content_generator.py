"""
AI-powered content generation for summaries and practice questions.
Uses OpenAI GPT models.
"""
from typing import List, Dict, Optional
import logging
import json

from openai import OpenAI

logger = logging.getLogger(__name__)


class ContentGenerator:
    """Generate educational content using LLMs."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview", temperature: float = 0.2):
        """
        Initialize the content generator.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
            temperature: Temperature for generation (0-1)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
    
    def generate_summary(self, context: str, topic: Optional[str] = None) -> str:
        """
        Generate a concise summary of the given context.
        
        Args:
            context: Text content to summarize
            topic: Optional specific topic to focus on
            
        Returns:
            Generated summary
        """
        prompt = self._build_summary_prompt(context, topic)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert educational assistant helping MISM students at CMU understand their course materials. CRITICAL: You must ONLY use information from the provided document context. DO NOT add any external knowledge, examples, or information not explicitly mentioned in the given text. If the document doesn't contain information about something, state that clearly. Generate summaries based STRICTLY on the provided content."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        
        summary = response.choices[0].message.content
        logger.info("Generated summary")
        return summary

    def generate_notes(self, context: str, topic: Optional[str] = None) -> str:
        """
        Generate detailed study notes from the context.
        
        Args:
            context: Text content to turn into notes
            topic: Optional topic focus
        """
        focus_line = f"Focus on the topic: {topic}.\n" if topic else ""
        prompt = f"""Create detailed study notes for CMU MISM students using ONLY the content below.

{focus_line}Requirements:
- Structured headings and subpoints explained in detail to help understand topics.
- Slide by slide or section by section explanations
- Key terms and definitions
- Examples or clarifications when present in the source
- Clear, instructional tone
- No external knowledge beyond the provided text

Document Content:
{context}

Produce the notes in markdown-style sections with bullets and sub-bullets."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert study-note writer. Produce comprehensive, well-structured notes grounded strictly in the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        
        notes = response.choices[0].message.content
        logger.info("Generated detailed notes")
        return notes
    
    def generate_mcq_questions(self, context: str, num_questions: int = 5) -> List[Dict]:
        """
        Generate multiple choice questions from the context.
        
        Args:
            context: Text content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of MCQ dictionaries
        """
        prompt = self._build_mcq_prompt(context, num_questions)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert educational assistant creating high-quality multiple choice questions for MISM students. Generate questions that test understanding, application, and analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            questions = result.get('questions', [])
            logger.info(f"Generated {len(questions)} MCQ questions")
            return questions
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing MCQ response: {str(e)}")
            return []
    
    def generate_true_false_questions(self, context: str, num_questions: int = 5) -> List[Dict]:
        """
        Generate true/false questions from the context.
        
        Args:
            context: Text content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of true/false question dictionaries
        """
        prompt = self._build_true_false_prompt(context, num_questions)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert educational assistant creating true/false questions for MISM students. Generate clear statements that test factual knowledge."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            questions = result.get('questions', [])
            logger.info(f"Generated {len(questions)} true/false questions")
            return questions
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing true/false response: {str(e)}")
            return []
    
    def generate_fill_blank_questions(self, context: str, num_questions: int = 5) -> List[Dict]:
        """
        Generate fill-in-the-blank questions from the context.
        
        Args:
            context: Text content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of fill-in-the-blank question dictionaries
        """
        prompt = self._build_fill_blank_prompt(context, num_questions)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert educational assistant creating fill-in-the-blank questions for MISM students. Generate questions with clear blanks and unambiguous answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            questions = result.get('questions', [])
            logger.info(f"Generated {len(questions)} fill-in-the-blank questions")
            return questions
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing fill-in-the-blank response: {str(e)}")
            return []
    
    def generate_short_answer_questions(self, context: str, num_questions: int = 5) -> List[Dict]:
        """
        Generate short answer questions from the context.
        
        Args:
            context: Text content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of short answer question dictionaries
        """
        prompt = self._build_short_answer_prompt(context, num_questions)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert educational assistant creating short answer questions for MISM students. Generate questions that require brief, focused responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            questions = result.get('questions', [])
            logger.info(f"Generated {len(questions)} short answer questions")
            return questions
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing short answer response: {str(e)}")
            return []
    
    def generate_match_following_questions(self, context: str, num_pairs: int = 5) -> Dict:
        """
        Generate match-the-following questions from the context.
        
        Args:
            context: Text content to generate questions from
            num_pairs: Number of matching pairs to generate
            
        Returns:
            Dictionary with matching pairs
        """
        prompt = self._build_match_following_prompt(context, num_pairs)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert educational assistant creating match-the-following questions for MISM students. Generate clear pairs that test understanding of relationships."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            logger.info("Generated match-the-following question")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing match-the-following response: {str(e)}")
            return {}
    
    def _build_summary_prompt(self, context: str, topic: Optional[str] = None) -> str:
        """Build prompt for summary generation."""
        if topic:
            return f"""Generate a summary focusing on the topic: {topic}

IMPORTANT INSTRUCTIONS:
- Use ONLY the information provided in the document below
- DO NOT add any external knowledge or information
- DO NOT include examples that are not in the document
- If the document doesn't contain specific information about the topic, clearly state that
- Quote or reference the document directly when possible
- Stay strictly within the boundaries of what the document says

Document Content:
{context}

Generate a summary that includes ONLY what is explicitly stated in the document above."""
        else:
            return f"""Generate a summary of the following document.

IMPORTANT INSTRUCTIONS:
- Use ONLY the information provided in the document below
- DO NOT add any external knowledge or information
- DO NOT include examples that are not in the document
- Summarize only what is explicitly stated
- Quote or reference the document directly when possible
- Stay strictly within the boundaries of what the document says

Document Content:
{context}

Generate a summary that includes ONLY what is explicitly stated in the document above."""
    
    def _build_mcq_prompt(self, context: str, num_questions: int) -> str:
        """Build prompt for MCQ generation."""
        return f"""Based on the following course material, generate {num_questions} multiple choice questions that test student understanding.

For each question:
- Create a clear, unambiguous question
- Provide 4 answer options (A, B, C, D)
- Mark the correct answer
- Include a brief explanation of why the answer is correct
- Ensure questions test different aspects: recall, understanding, and application

Course Material:
{context}

Return your response as a JSON object with this structure:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": {{
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
      }},
      "correct_answer": "A",
      "explanation": "Explanation of why this answer is correct"
    }}
  ]
}}"""
    
    def _build_true_false_prompt(self, context: str, num_questions: int) -> str:
        """Build prompt for true/false question generation."""
        return f"""Based on the following course material, generate {num_questions} true/false questions.

For each question:
- Create a clear statement that is definitively true or false
- Mark whether it is true or false
- Include a brief explanation

Course Material:
{context}

Return your response as a JSON object with this structure:
{{
  "questions": [
    {{
      "statement": "Statement text here",
      "answer": true,
      "explanation": "Explanation of why this is true/false"
    }}
  ]
}}"""
    
    def _build_fill_blank_prompt(self, context: str, num_questions: int) -> str:
        """Build prompt for fill-in-the-blank generation."""
        return f"""Based on the following course material, generate {num_questions} fill-in-the-blank questions.

For each question:
- Create a sentence with one blank (use _____ for the blank)
- Provide the correct answer
- Optionally provide hints

Course Material:
{context}

Return your response as a JSON object with this structure:
{{
  "questions": [
    {{
      "question": "The process of _____ involves...",
      "answer": "correct answer",
      "hint": "Optional hint"
    }}
  ]
}}"""
    
    def _build_short_answer_prompt(self, context: str, num_questions: int) -> str:
        """Build prompt for short answer generation."""
        return f"""Based on the following course material, generate {num_questions} short answer questions.

For each question:
- Create a question that requires a brief response (2-4 sentences)
- Provide a sample answer
- Include key points that should be covered

Course Material:
{context}

Return your response as a JSON object with this structure:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "sample_answer": "A sample correct answer",
      "key_points": ["Point 1", "Point 2", "Point 3"]
    }}
  ]
}}"""
    
    def _build_match_following_prompt(self, context: str, num_pairs: int) -> str:
        """Build prompt for match-the-following generation."""
        return f"""Based on the following course material, generate a match-the-following question with {num_pairs} pairs.

Create pairs that test understanding of:
- Concepts and definitions
- Terms and descriptions
- Causes and effects
- Problems and solutions

Course Material:
{context}

Return your response as a JSON object with this structure:
{{
  "instruction": "Match the items in Column A with their corresponding items in Column B",
  "column_a": [
    {{"id": "A1", "text": "Item 1"}},
    {{"id": "A2", "text": "Item 2"}}
  ],
  "column_b": [
    {{"id": "B1", "text": "Match 1"}},
    {{"id": "B2", "text": "Match 2"}}
  ],
  "correct_matches": {{
    "A1": "B1",
    "A2": "B2"
  }}
}}"""
    
    def generate_answer_with_context(self, query: str, context: str) -> str:
        """
        Generate an answer to a query using the provided context.
        
        Args:
            query: User's question
            context: Retrieved context from RAG
            
        Returns:
            Generated answer
        """
        prompt = f"""You are an AI study assistant for MISM students at CMU. Answer the following question based ONLY on the provided context from course materials.

If the context doesn't contain enough information to answer the question, say so clearly. Do not make up information.

Context:
{context}

Question: {query}

Provide a clear, accurate answer that:
1. Directly addresses the question
2. Uses information from the context
3. Explains concepts in a way that helps learning
4. Includes relevant examples when available"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful educational AI assistant for MISM students. Always ground your answers in the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        
        answer = response.choices[0].message.content
        logger.info("Generated answer with context")
        return answer
