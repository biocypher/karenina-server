"""
API handlers for rubric management and trait generation.
"""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from karenina.schemas import Rubric, RubricTrait
from karenina.schemas.question_class import Question
from pydantic import BaseModel

from karenina_server.services.generation_service import GenerationService

router = APIRouter()

# Global rubric store (in-memory for now)
# In a production system, this would be stored in a database
current_rubric: Rubric | None = None


class RubricTraitGenerationRequest(BaseModel):
    """Request to generate rubric traits using LLM."""

    questions: list[dict[str, Any]]  # Question data from frontend
    system_prompt: str | None = None
    user_suggestions: list[str] | None = None
    model_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1


class RubricTraitGenerationResponse(BaseModel):
    """Response containing generated rubric traits."""

    traits: list[RubricTrait]
    job_id: str | None = None


@router.post("/generate-rubric-traits", response_model=RubricTraitGenerationResponse)
async def generate_rubric_traits(request: RubricTraitGenerationRequest):
    """
    Generate rubric traits using LLM based on question context.

    This endpoint analyzes the provided questions and generates appropriate
    evaluation traits that can be used to create a rubric.
    """
    try:
        # Convert questions to internal format
        questions = []
        for q_data in request.questions:
            question = Question(
                id=q_data.get("id", str(uuid.uuid4())),
                question=q_data.get("text", "Unknown question"),
                raw_answer=q_data.get("raw_answer", "Unknown answer"),
                tags=q_data.get("tags", [])
            )
            questions.append(question)

        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        # Build system prompt for trait generation
        system_prompt = request.system_prompt or _build_default_rubric_system_prompt()

        # Build user prompt with question context and suggestions
        user_prompt = _build_rubric_generation_prompt(questions, request.user_suggestions)

        # Use generation service to generate traits
        generation_service = GenerationService()

        # For now, we'll generate traits synchronously
        # In a production system, this might use the job queue system
        generated_text = generation_service.generate_rubric_traits(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_provider=request.model_provider,
            model_name=request.model_name,
            temperature=request.temperature,
        )

        # Parse the generated text into RubricTrait objects
        traits = _parse_generated_traits(generated_text)

        return RubricTraitGenerationResponse(traits=traits)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating rubric traits: {str(e)}")


@router.post("/rubric", response_model=dict[str, str])
async def create_or_update_rubric(rubric: Rubric):
    """
    Create or update the current rubric.

    This endpoint stores the rubric that will be used for evaluation.
    """
    global current_rubric

    try:
        # Validate rubric
        if not rubric.title.strip():
            raise HTTPException(status_code=400, detail="Rubric title cannot be empty")

        if not rubric.traits:
            raise HTTPException(status_code=400, detail="Rubric must have at least one trait")

        # Validate trait names are unique
        trait_names = [trait.name for trait in rubric.traits]
        if len(trait_names) != len(set(trait_names)):
            raise HTTPException(status_code=400, detail="Trait names must be unique")

        # Store the rubric
        current_rubric = rubric

        return {"message": "Rubric saved successfully", "title": rubric.title}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving rubric: {str(e)}")


@router.get("/rubric", response_model=Rubric | None)
async def get_current_rubric():
    """
    Get the current rubric.

    Returns the rubric that is currently configured for evaluation,
    or None if no rubric is set.
    """
    return current_rubric


@router.delete("/rubric", response_model=dict[str, str])
async def delete_current_rubric():
    """
    Delete the current rubric.

    This removes the currently configured rubric.
    """
    global current_rubric
    current_rubric = None
    return {"message": "Rubric deleted successfully"}


def _build_default_rubric_system_prompt() -> str:
    """Build the default system prompt for rubric trait generation."""
    return """You are an expert in educational assessment and rubric design. Your task is to analyze question-answer pairs and suggest appropriate evaluation criteria (traits) that can be used to assess the quality of responses.

Generate evaluation traits that are:
1. Specific and measurable
2. Relevant to the question domain
3. Independent of each other
4. Useful for distinguishing between good and poor responses

For each trait, specify:
- Name: Short, descriptive identifier
- Description: Clear explanation of what is being evaluated
- Type: Either "boolean" (true/false) or "score" (1-5 scale)

Focus on qualitative aspects like clarity, completeness, accuracy, relevance, and coherence."""


def _build_rubric_generation_prompt(questions: list[Question], user_suggestions: list[str] | None) -> str:
    """Build the user prompt for rubric trait generation."""
    prompt_parts = ["Please analyze the following questions and suggest appropriate evaluation traits:\n"]

    # Add question context
    for i, question in enumerate(questions[:5], 1):  # Limit to first 5 questions
        prompt_parts.append(f"{i}. {question.question}")

    if len(questions) > 5:
        prompt_parts.append(f"... and {len(questions) - 5} more questions")

    # Add user suggestions if provided
    if user_suggestions:
        prompt_parts.append(f"\nUser suggestions for traits to consider: {', '.join(user_suggestions)}")

    prompt_parts.append("""
Please suggest 3-7 evaluation traits in the following JSON format:
[
  {
    "name": "trait_name",
    "description": "Clear description of what this trait evaluates",
    "kind": "boolean",
    "min_score": null,
    "max_score": null
  },
  {
    "name": "another_trait",
    "description": "Description for score-based trait",
    "kind": "score",
    "min_score": 1,
    "max_score": 5
  }
]""")

    return "\n".join(prompt_parts)


def _parse_generated_traits(generated_text: str) -> list[RubricTrait]:
    """Parse generated text into RubricTrait objects."""
    import json
    import re

    # Try to extract JSON from the generated text
    json_match = re.search(r"\[.*\]", generated_text, re.DOTALL)
    if not json_match:
        # Fallback: create some default traits
        return [
            RubricTrait(name="clarity", description="Is the response clear and easy to understand?", kind="boolean"),
            RubricTrait(
                name="completeness",
                description="How complete is the response on a scale of 1-5?",
                kind="score",
                min_score=1,
                max_score=5,
            ),
        ]

    try:
        trait_data = json.loads(json_match.group())
        traits = []

        for trait_dict in trait_data:
            trait = RubricTrait(**trait_dict)
            traits.append(trait)

        return traits

    except (json.JSONDecodeError, ValueError):
        # Fallback to default traits if parsing fails
        return [
            RubricTrait(name="accuracy", description="Is the response factually accurate?", kind="boolean"),
            RubricTrait(
                name="relevance",
                description="How relevant is the response to the question (1-5)?",
                kind="score",
                min_score=1,
                max_score=5,
            ),
        ]
