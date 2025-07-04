"""Template generation API handlers."""

from fastapi import HTTPException

try:
    from karenina.llm import ChatRequest, ChatResponse, call_model, delete_session, get_session, list_sessions

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


def register_generation_routes(
    app, TemplateGenerationRequest, TemplateGenerationResponse, TemplateGenerationStatusResponse
):
    """Register template generation-related routes."""

    @app.post("/api/generate-answer-templates", response_model=TemplateGenerationResponse)
    async def generate_answer_templates_endpoint(request: TemplateGenerationRequest):
        """Start answer template generation for a set of questions."""
        # Import LLM_AVAILABLE from server to maintain compatibility with tests
        from .. import server

        if not getattr(server, "LLM_AVAILABLE", LLM_AVAILABLE):
            raise HTTPException(status_code=503, detail="LLM functionality not available")

        try:
            from karenina_server.services.generation_service import generation_service

            job_id = generation_service.start_generation(
                questions_data=request.questions,
                config=request.config,
                custom_system_prompt=request.custom_system_prompt,
            )

            return TemplateGenerationResponse(
                job_id=job_id,
                status="started",
                message=f"Template generation started for {len(request.questions)} questions",
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start generation: {e!s}")

    @app.get("/api/generation-progress/{job_id}")
    async def get_generation_progress(job_id: str):
        """Get the progress of a template generation job."""
        try:
            from karenina_server.services.generation_service import generation_service

            progress = generation_service.get_progress(job_id)
            if not progress:
                raise HTTPException(status_code=404, detail="Job not found")

            # Format response to match frontend expectations
            response = TemplateGenerationStatusResponse(
                job_id=job_id,
                status=progress["status"],
                percentage=progress.get("percentage", 0.0),
                current_question=progress.get("current_question", ""),
                processed_count=progress.get("processed_count", 0),
                total_count=progress.get("total_questions", 0),
                estimated_time_remaining=progress.get("estimated_time_remaining"),
                error=progress.get("error_message"),
            )

            # Add result if completed
            job = generation_service.jobs.get(job_id)
            if job and job.status == "completed" and job.result:
                response.result = job.result

            return response

        except HTTPException:
            raise
        except Exception as e:
            print(f"Error getting generation progress: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/cancel-generation/{job_id}")
    async def cancel_generation_endpoint(job_id: str):
        """Cancel a template generation job."""
        try:
            from karenina_server.services.generation_service import generation_service

            success = generation_service.cancel_job(job_id)
            if not success:
                raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")

            return {"message": "Job cancelled successfully"}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e!s}")
