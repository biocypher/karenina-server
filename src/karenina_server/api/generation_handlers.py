"""Template generation API handlers."""

from typing import Any

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

try:
    import karenina.infrastructure.llm  # noqa: F401 - Test if LLM module is available

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


def register_generation_routes(
    app: Any, TemplateGenerationRequest: Any, TemplateGenerationResponse: Any, TemplateGenerationStatusResponse: Any
) -> None:
    """Register template generation-related routes."""

    @app.post("/api/generate-answer-templates", response_model=TemplateGenerationResponse)  # type: ignore[misc]
    async def generate_answer_templates_endpoint(request: TemplateGenerationRequest) -> TemplateGenerationResponse:
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
                force_regenerate=getattr(request, "force_regenerate", False),
            )

            return TemplateGenerationResponse(
                job_id=job_id,
                status="started",
                message=f"Template generation started for {len(request.questions)} questions",
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start generation: {e!s}") from e

    @app.get("/api/generation-progress/{job_id}")  # type: ignore[misc]
    async def get_generation_progress(job_id: str) -> TemplateGenerationStatusResponse:
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
                duration_seconds=progress.get("duration_seconds"),
                last_task_duration=progress.get("last_task_duration"),
                error=progress.get("error_message"),
                in_progress_questions=progress.get("in_progress_questions", []),
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
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/cancel-generation/{job_id}")  # type: ignore[misc]
    async def cancel_generation_endpoint(job_id: str) -> dict[str, str]:
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
            raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e!s}") from e

    @app.websocket("/ws/generation-progress/{job_id}")  # type: ignore[misc]
    async def websocket_generation_progress(websocket: WebSocket, job_id: str) -> None:
        """WebSocket endpoint for real-time generation progress updates."""
        import asyncio

        from karenina_server.services.generation_service import generation_service

        # Validate job exists
        job = generation_service.jobs.get(job_id)
        if not job:
            await websocket.close(code=1008, reason="Job not found")
            return

        # Accept the connection
        await websocket.accept()

        # Set the event loop for the broadcaster if not already set
        if generation_service.broadcaster._event_loop is None:
            generation_service.broadcaster.set_event_loop(asyncio.get_running_loop())

        # Subscribe to progress updates
        await generation_service.broadcaster.subscribe(job_id, websocket)

        try:
            # Send current state immediately
            progress = generation_service.get_progress(job_id)
            if progress:
                await websocket.send_json(
                    {
                        "type": "snapshot",
                        "job_id": job_id,
                        "status": progress["status"],
                        "percentage": progress["percentage"],
                        "processed": progress["processed_count"],
                        "total": progress["total_questions"],
                        "in_progress_questions": progress["in_progress_questions"],
                        "start_time": progress["start_time"],  # Unix timestamp for client-side live clock
                        "duration_seconds": progress["duration_seconds"],
                        "last_task_duration": progress["last_task_duration"],
                        "current_question": progress["current_question"],
                    }
                )

            # Keep connection alive and wait for disconnect
            while True:
                try:
                    # Receive messages (mainly for keepalive/ping)
                    await websocket.receive_text()
                except WebSocketDisconnect:
                    break

        except Exception:
            pass  # Connection closed
        finally:
            # Unsubscribe and clean up
            await generation_service.broadcaster.unsubscribe(job_id, websocket)
