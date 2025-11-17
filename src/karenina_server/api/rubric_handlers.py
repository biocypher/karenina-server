"""
API handlers for rubric management.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from karenina.schemas import LLMRubricTrait, Question, Rubric
from pydantic import BaseModel

from karenina_server.services.generation_service import GenerationService
from karenina_server.services.rubric_service import rubric_service

router = APIRouter()


class RubricTraitGenerationConfig(BaseModel):
    """Configuration for rubric trait generation."""

    model_provider: str | None = None
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.1
    interface: str = "langchain"


class RubricTraitGenerationRequest(BaseModel):
    """Request to generate rubric traits using LLM."""

    questions: dict[str, dict[str, Any]]  # Question data from frontend as object
    system_prompt: str | None = None
    user_suggestions: list[str] | None = None
    config: RubricTraitGenerationConfig


class RubricTraitGenerationResponse(BaseModel):
    """Response containing generated LLM rubric traits."""

    traits: list[LLMRubricTrait]
    job_id: str | None = None


@router.post("/generate-rubric-traits", response_model=RubricTraitGenerationResponse)
async def generate_rubric_traits(request: RubricTraitGenerationRequest) -> RubricTraitGenerationResponse:
    """
    Generate rubric traits using LLM based on question context.

    This endpoint analyzes the provided questions and generates appropriate
    evaluation traits that can be used to create a rubric.
    """
    try:
        # Convert questions to internal format
        questions = []
        for q_id, q_data in request.questions.items():
            question = Question(
                id=q_id,
                question=q_data.get("question", "Unknown question"),
                raw_answer=q_data.get("raw_answer", "Unknown answer"),
                tags=q_data.get("tags", []),
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
        # Determine model_provider based on interface
        if request.config.interface == "langchain":
            model_provider = request.config.model_provider or "google_genai"
        else:
            # For openrouter, provider should be empty
            model_provider = ""

        generated_text = generation_service.generate_rubric_traits(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_provider=model_provider,
            model_name=request.config.model_name,
            temperature=request.config.temperature,
            interface=request.config.interface,
        )

        # Parse the generated text into RubricTrait objects
        traits = _parse_generated_traits(generated_text)

        return RubricTraitGenerationResponse(traits=traits)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating rubric traits: {str(e)}") from e


@router.post("/rubric", response_model=dict[str, str])
async def create_or_update_rubric(rubric: Rubric) -> dict[str, str]:
    """
    Create or update the current rubric.

    This endpoint stores the rubric that will be used for evaluation.
    """
    try:
        # Validate rubric has at least one trait (LLM, manual, or metric)
        if not rubric.traits and not rubric.manual_traits and not rubric.metric_traits:
            raise HTTPException(status_code=400, detail="Rubric must have at least one trait (LLM, manual, or metric)")

        # Validate trait names are unique across all trait types
        llm_trait_names = [trait.name for trait in rubric.traits]
        manual_trait_names = [trait.name for trait in rubric.manual_traits]
        metric_trait_names = [trait.name for trait in rubric.metric_traits]
        all_trait_names = llm_trait_names + manual_trait_names + metric_trait_names

        if len(all_trait_names) != len(set(all_trait_names)):
            raise HTTPException(status_code=400, detail="Trait names must be unique across all trait types")

        # Store the rubric using the service
        rubric_service.set_current_rubric(rubric)

        return {"message": "Rubric saved successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving rubric: {str(e)}") from e


@router.get("/rubric", response_model=Rubric | None)
async def get_current_rubric() -> Rubric | None:
    """
    Get the current rubric.

    Returns the rubric that is currently configured for evaluation,
    or None if no rubric is set.
    """
    return rubric_service.get_current_rubric()


@router.delete("/rubric", response_model=dict[str, str])
async def delete_current_rubric() -> dict[str, str]:
    """
    Delete the current rubric.

    This removes the currently configured rubric.
    """
    rubric_service.clear_rubric()
    return {"message": "Rubric deleted successfully"}


@router.get("/rubric/default-system-prompt", response_model=dict[str, str])
async def get_default_system_prompt() -> dict[str, str]:
    """
    Get the default system prompt for rubric trait generation.

    Returns the default system prompt that provides guidance for generating
    rubric traits based on question-answer pairs.
    """
    return {"prompt": _build_default_rubric_system_prompt()}


def _build_default_rubric_system_prompt() -> str:
    """Build the default system prompt for rubric trait generation."""
    return """You are an expert in rubric design. Your task is to analyze question-answer pairs and suggest appropriate evaluation criteria (traits) that can be used to assess the quality of responses.

<important>
Generate traits that evaluate QUALITATIVE aspects of how the answer is presented, NOT the factual accuracy or correctness of the content. The traits should be assessable by someone who doesn't know the actual answer to the question.
You will be provided with a set of example question-answer pairs from the domain being evaluated. These examples will help you understand the context and typical response patterns in this specific domain, allowing you to generate more targeted and relevant evaluation traits.
</important>

<trait_requirements>
- Specific and measurable
- Relevant to the question domain and response style
- Independent of each other (minimal overlap)
- Useful for distinguishing between well-structured and poorly-structured responses
- Focus on HOW information is presented, not WHETHER it's correct
</trait_requirements>

<example_traits>
Consider qualitative aspects such as:
- Response structure and organization
- Clarity of explanations
- Level of detail provided
- Presence and quality of examples
- Use of technical language (when appropriate)
- Conciseness vs. comprehensiveness
- Presence of code snippets or diagrams
- Step-by-step breakdowns
- Acknowledgment of assumptions or limitations
- Tone appropriateness (formal, casual, technical)
- Use of formatting (lists, headers, emphasis)
</example_traits>

<output_format>
You can provide a reasoning trace explaining your choices by encapsulating the traits in a <reasoning> tags.
Once you have state your resaoning you should wrap the traits into a JSON code fence in order to make it easier to parse.

Each trait should be in the following format:
```json
{
    "name": # Short, descriptive identifier (2-4 words)
    "description": # Clear description of what this trait evaluates
    "kind": # Either "boolean" (true/false) or "score" (1-5 scale)
    "min_score": # null,
    "max_score": null
}
```
</output_format>

<example_output>
<reasoning>
I have analyzed the question-answer and ....
</reasoning>
```json
[
    {
        "name": "clarity",
        "description": "Is the response clear and easy to understand?",
        "kind": "boolean"
    },
    {
        "name": "completeness",
        "description": "How complete is the response on a scale of 1-5?",
        "kind": "score",
        "min_score": 1,
        "max_score": 5
    }
    ...
]
```
</example_output>
"""


def _build_rubric_generation_prompt(questions: list[Question], user_suggestions: list[str] | None) -> str:
    """Build the user prompt for rubric trait generation."""
    user_prompt = """<question_answer_pairs>
"""

    # Add question context
    for i, question in enumerate(questions, 1):
        user_prompt += f"{i}. {question.question}: {question.raw_answer}\n"

    user_prompt += """</question_answer_pairs>"""

    # Add user suggestions if provided
    user_prompt += """\n\n<user_traits_suggestions>"""
    if user_suggestions:
        user_prompt += f"{', '.join(user_suggestions)}"
    user_prompt += """\n</user_traits_suggestions>"""

    return user_prompt


def _parse_generated_traits(generated_text: str) -> list[LLMRubricTrait]:
    """Parse generated text into LLMRubricTrait objects."""
    import json
    import re

    # First, try to extract JSON from code fences (```json...```)
    json_fence_match = re.search(r"```json\s*\n(.*?)\n```", generated_text, re.DOTALL)
    if json_fence_match:
        json_content = json_fence_match.group(1).strip()
    else:
        # Fallback: try to extract raw JSON array
        json_match = re.search(r"\[.*\]", generated_text, re.DOTALL)
        json_content = json_match.group().strip() if json_match else None

    if not json_content:
        # Fallback: create some default traits
        return [
            LLMRubricTrait(name="clarity", description="Is the response clear and easy to understand?", kind="boolean"),
            LLMRubricTrait(
                name="completeness",
                description="How complete is the response on a scale of 1-5?",
                kind="score",
                min_score=1,
                max_score=5,
            ),
        ]

    try:
        trait_data = json.loads(json_content)
        traits = []

        for trait_dict in trait_data:
            trait = LLMRubricTrait(**trait_dict)
            traits.append(trait)

        return traits

    except (json.JSONDecodeError, ValueError):
        # Fallback to default traits if parsing fails
        return [
            LLMRubricTrait(name="accuracy", description="Is the response factually accurate?", kind="boolean"),
            LLMRubricTrait(
                name="relevance",
                description="How relevant is the response to the question (1-5)?",
                kind="score",
                min_score=1,
                max_score=5,
            ),
        ]
