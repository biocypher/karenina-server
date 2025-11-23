"""
API handlers for rubric management.
"""

from fastapi import APIRouter, HTTPException
from karenina.schemas import Rubric

from karenina_server.services.rubric_service import rubric_service

router = APIRouter()


@router.post("/rubric", response_model=dict[str, str])
async def create_or_update_rubric(rubric: Rubric) -> dict[str, str]:
    """
    Create or update the current rubric.

    This endpoint stores the rubric that will be used for evaluation.
    """
    try:
        # Validate rubric has at least one trait (LLM, regex, callable, or metric)
        if (
            not rubric.llm_traits
            and not rubric.regex_traits
            and not rubric.callable_traits
            and not rubric.metric_traits
        ):
            raise HTTPException(
                status_code=400, detail="Rubric must have at least one trait (LLM, regex, callable, or metric)"
            )

        # Validate trait names are unique across all trait types
        llm_trait_names = [trait.name for trait in rubric.llm_traits]
        regex_trait_names = [trait.name for trait in (rubric.regex_traits or [])]
        callable_trait_names = [trait.name for trait in (rubric.callable_traits or [])]
        metric_trait_names = [trait.name for trait in (rubric.metric_traits or [])]
        all_trait_names = llm_trait_names + regex_trait_names + callable_trait_names + metric_trait_names

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
