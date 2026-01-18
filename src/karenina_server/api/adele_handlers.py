"""ADeLe question classification API handlers."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from karenina_server.schemas.adele import (
    AdeleTraitInfoResponse,
    ClassificationResultPayload,
    ClassifyBatchProgressResponse,
    ClassifyBatchRequest,
    ClassifyBatchResultsResponse,
    ClassifySingleQuestionRequest,
    ClassifySingleQuestionResponse,
    ListAdeleTraitsResponse,
    StartClassifyBatchResponse,
)

if TYPE_CHECKING:
    from karenina_server.services.adele_classification_service import (
        AdeleClassificationService,
    )

logger = logging.getLogger(__name__)


def register_adele_routes(app: FastAPI, adele_service: AdeleClassificationService) -> None:
    """Register ADeLe classification routes.

    V2 Routes (RESTful):
        - GET /api/v2/adele/traits - List available ADeLe traits
        - POST /api/v2/adele/classify - Classify a single question
        - POST /api/v2/adele/classify-batch - Start batch classification job
        - GET /api/v2/adele/classify-batch/{job_id} - Get batch job progress
        - GET /api/v2/adele/classify-batch/{job_id}/results - Get batch job results
        - DELETE /api/v2/adele/classify-batch/{job_id} - Cancel a batch job

    WebSocket:
        - WS /ws/adele/classify-progress/{job_id} - Real-time progress updates

    Args:
        app: The FastAPI application instance.
        adele_service: The ADeLe classification service.
    """

    @app.get("/api/v2/adele/traits", response_model=ListAdeleTraitsResponse)
    async def list_adele_traits() -> ListAdeleTraitsResponse:
        """List all available ADeLe traits with descriptions.

        Returns:
            ListAdeleTraitsResponse with trait information.
        """
        try:
            traits_info = adele_service.get_available_traits()
            traits_response = [
                AdeleTraitInfoResponse(
                    name=t.name,
                    code=t.code,
                    description=t.description,
                    classes=t.classes,
                    class_names=t.class_names,
                )
                for t in traits_info
            ]
            return ListAdeleTraitsResponse(
                success=True,
                traits=traits_response,
                count=len(traits_response),
            )
        except Exception as e:
            logger.error(f"Failed to list ADeLe traits: {e}", exc_info=True)
            return ListAdeleTraitsResponse(
                success=False,
                traits=[],
                count=0,
                error=str(e),
            )

    @app.post("/api/v2/adele/classify", response_model=ClassifySingleQuestionResponse)
    async def classify_single_question(
        request: ClassifySingleQuestionRequest,
    ) -> ClassifySingleQuestionResponse:
        """Classify a single question using ADeLe dimensions.

        This is a synchronous endpoint that returns immediately with results.
        For batch classification, use the batch endpoint.

        Args:
            request: Request with question_text, optional question_id, and trait_names.

        Returns:
            ClassifySingleQuestionResponse with classification results.
        """
        try:
            result = adele_service.classify_single(
                question_text=request.question_text,
                trait_names=request.trait_names,
                question_id=request.question_id,
            )

            return ClassifySingleQuestionResponse(
                success=True,
                result=ClassificationResultPayload(
                    question_id=result.question_id,
                    question_text=result.question_text,
                    scores=result.scores,
                    labels=result.labels,
                    model=result.model,
                    classified_at=result.classified_at,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to classify question: {e}", exc_info=True)
            return ClassifySingleQuestionResponse(
                success=False,
                result=None,
                error=str(e),
            )

    @app.post("/api/v2/adele/classify-batch", response_model=StartClassifyBatchResponse)
    async def start_batch_classification(
        request: ClassifyBatchRequest,
    ) -> StartClassifyBatchResponse:
        """Start a batch classification job.

        This endpoint returns immediately with a job_id. Use the progress
        endpoint or WebSocket to track completion.

        Args:
            request: Request with list of questions and optional trait_names.

        Returns:
            StartClassifyBatchResponse with job_id and status.
        """
        try:
            if not request.questions:
                raise HTTPException(status_code=400, detail="Questions list cannot be empty")

            # Validate question format
            for i, q in enumerate(request.questions):
                if "question_id" not in q or "question_text" not in q:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Question at index {i} missing required fields 'question_id' or 'question_text'",
                    )

            job_id = adele_service.start_batch_job(
                questions=request.questions,
                trait_names=request.trait_names,
            )

            return StartClassifyBatchResponse(
                success=True,
                job_id=job_id,
                status="pending",
                message=f"Batch classification started for {len(request.questions)} questions",
                total_questions=len(request.questions),
            )

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Failed to start batch classification: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to start batch: {e!s}") from e

    @app.get("/api/v2/adele/classify-batch/{job_id}", response_model=ClassifyBatchProgressResponse)
    async def get_batch_progress(job_id: str) -> ClassifyBatchProgressResponse:
        """Get progress of a batch classification job.

        Args:
            job_id: The batch job identifier.

        Returns:
            ClassifyBatchProgressResponse with progress details.
        """
        try:
            status = adele_service.get_job_status(job_id)
            if not status:
                raise HTTPException(status_code=404, detail="Job not found")

            return ClassifyBatchProgressResponse(
                success=True,
                job_id=job_id,
                status=status["status"],
                progress=status["percentage"],
                completed=status["completed_count"],
                total=status["total_questions"],
                message=f"Processing: {status['completed_count']}/{status['total_questions']}",
                error=status.get("error"),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get batch progress: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get progress: {e!s}") from e

    @app.get(
        "/api/v2/adele/classify-batch/{job_id}/results",
        response_model=ClassifyBatchResultsResponse,
    )
    async def get_batch_results(job_id: str) -> ClassifyBatchResultsResponse:
        """Get results of a completed batch classification job.

        Args:
            job_id: The batch job identifier.

        Returns:
            ClassifyBatchResultsResponse with classification results.
        """
        try:
            status = adele_service.get_job_status(job_id)
            if not status:
                raise HTTPException(status_code=404, detail="Job not found")

            if status["status"] != "completed":
                return ClassifyBatchResultsResponse(
                    success=False,
                    job_id=job_id,
                    status=status["status"],
                    results=[],
                    error="Job not yet completed",
                )

            results = adele_service.get_job_results(job_id)
            if not results:
                return ClassifyBatchResultsResponse(
                    success=False,
                    job_id=job_id,
                    status=status["status"],
                    results=[],
                    error="No results available",
                )

            # Convert to response format
            results_payload = [
                ClassificationResultPayload(
                    question_id=r.question_id,
                    question_text=r.question_text,
                    scores=r.scores,
                    labels=r.labels,
                    model=r.model,
                    classified_at=r.classified_at,
                )
                for r in results.values()
            ]

            return ClassifyBatchResultsResponse(
                success=True,
                job_id=job_id,
                status="completed",
                results=results_payload,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get batch results: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get results: {e!s}") from e

    @app.delete("/api/v2/adele/classify-batch/{job_id}")
    async def cancel_batch_job(job_id: str) -> dict[str, Any]:
        """Cancel a batch classification job.

        Args:
            job_id: The batch job identifier.

        Returns:
            Success message dict.
        """
        try:
            success = adele_service.cancel_job(job_id)
            if not success:
                raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")

            return {"success": True, "message": "Job cancelled successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel batch job: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e!s}") from e

    @app.websocket("/ws/adele/classify-progress/{job_id}")
    async def websocket_classify_progress(websocket: WebSocket, job_id: str) -> None:
        """WebSocket endpoint for real-time classification progress updates.

        Args:
            websocket: The WebSocket connection.
            job_id: The batch job identifier.
        """
        # Validate job exists
        job = adele_service.jobs.get(job_id)
        if not job:
            await websocket.close(code=1008, reason="Job not found")
            return

        # Accept the connection
        await websocket.accept()

        # Set the event loop for the broadcaster if not already set
        if adele_service.broadcaster._event_loop is None:
            adele_service.broadcaster.set_event_loop(asyncio.get_running_loop())

        # Subscribe to progress updates
        await adele_service.broadcaster.subscribe(job_id, websocket)

        try:
            # Send current state immediately
            status = adele_service.get_job_status(job_id)
            if status:
                await websocket.send_json(
                    {
                        "type": "snapshot",
                        "job_id": job_id,
                        "status": status["status"],
                        "percentage": status["percentage"],
                        "completed": status["completed_count"],
                        "total": status["total_questions"],
                        "current_question_id": status.get("current_question_id"),
                        "duration_seconds": status.get("duration_seconds"),
                    }
                )

            # Keep connection alive and wait for client disconnect
            while True:
                try:
                    await websocket.receive_text()
                except WebSocketDisconnect:
                    break
        finally:
            # Unsubscribe on disconnect
            await adele_service.broadcaster.unsubscribe(job_id, websocket)
