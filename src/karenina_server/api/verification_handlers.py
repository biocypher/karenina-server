"""Benchmark verification API handlers."""

import tempfile
from pathlib import Path
from typing import Any

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse


def register_verification_routes(app: Any, verification_service: Any) -> None:
    """Register verification-related routes."""

    @app.get("/api/finished-templates")  # type: ignore[misc]
    async def get_finished_templates_endpoint() -> dict[str, Any]:
        """Get list of finished templates for verification."""
        try:
            # This is a placeholder - in a real implementation, you'd get this from your data store
            # For now, return empty list since we don't have access to the checkpoint data here
            return {"finished_templates": []}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting finished templates: {e!s}") from e

    @app.post("/api/start-verification")  # type: ignore[misc]
    async def start_verification_endpoint(request: dict[str, Any]) -> dict[str, Any]:
        """Start verification job."""
        try:
            import json

            from karenina.benchmark.models import FinishedTemplate, VerificationConfig
            from karenina.schemas import ManualRubricTrait, MetricRubricTrait, Rubric, RubricTrait
            from karenina.utils.async_utils import AsyncConfig

            # Parse request
            config_data = request.get("config", {})
            question_ids = request.get("question_ids")
            finished_templates_data = request.get("finished_templates", [])
            run_name = request.get("run_name")  # Optional user-defined run name
            async_config_data = request.get("async_config")  # Optional async configuration
            storage_url = request.get("storage_url")  # Optional database URL for auto-save
            benchmark_name = request.get("benchmark_name")  # Optional benchmark name for auto-save

            # DEBUG: Log what backend receives
            print("🔍 Backend: Received verification request")
            print(f"  Rubric enabled in config? {config_data.get('rubric_enabled', False)}")

            # Check if any templates have metric traits
            templates_with_metric_traits = [
                t
                for t in finished_templates_data
                if t.get("question_rubric") and t.get("question_rubric", {}).get("metric_traits")
            ]
            print(
                f"  Templates with metric traits: {len(templates_with_metric_traits)} / {len(finished_templates_data)}"
            )

            if templates_with_metric_traits:
                sample = templates_with_metric_traits[0]
                print(f"  Sample metric trait: {json.dumps(sample['question_rubric']['metric_traits'][0], indent=2)}")

            # Create config
            config = VerificationConfig(**config_data)

            # Create async config if provided, otherwise use environment defaults
            async_config = None
            if async_config_data:
                async_config = AsyncConfig(**async_config_data)

            # Create finished templates (needed for rubric validation)
            finished_templates = [FinishedTemplate(**template_data) for template_data in finished_templates_data]

            # Convert question_rubric dicts to Rubric objects
            for template in finished_templates:
                if template.question_rubric:
                    rubric_dict = template.question_rubric

                    # Parse traits
                    traits = [RubricTrait(**trait_data) for trait_data in rubric_dict.get("traits", [])]

                    # Parse manual_traits
                    manual_traits = [
                        ManualRubricTrait(**trait_data) for trait_data in rubric_dict.get("manual_traits", [])
                    ]

                    # Parse metric_traits
                    metric_traits = [
                        MetricRubricTrait(**trait_data) for trait_data in rubric_dict.get("metric_traits", [])
                    ]

                    # Create Rubric object
                    rubric = Rubric(traits=traits, manual_traits=manual_traits, metric_traits=metric_traits)

                    # Replace dict with Rubric object (direct attribute assignment)
                    template.question_rubric = rubric

            # DEBUG: Log parsed templates
            templates_with_metric_traits_parsed = [
                t
                for t in finished_templates
                if t.question_rubric and hasattr(t.question_rubric, "metric_traits") and t.question_rubric.metric_traits
            ]
            print(f"  Parsed templates with metric traits: {len(templates_with_metric_traits_parsed)}")
            if templates_with_metric_traits_parsed:
                sample = templates_with_metric_traits_parsed[0]
                print(f"  Sample parsed metric trait name: {sample.question_rubric.metric_traits[0].name}")
                print(f"  Sample evaluation_mode: {sample.question_rubric.metric_traits[0].evaluation_mode}")

            # Validate rubric availability if rubric evaluation is enabled
            if getattr(config, "rubric_enabled", False):
                from ..services.rubric_service import rubric_service

                # Check for any available rubrics (global OR question-specific)
                has_any_rubric = rubric_service.has_any_rubric(finished_templates)

                if not has_any_rubric:
                    raise HTTPException(
                        status_code=400,
                        detail="Rubric evaluation is enabled but no rubrics are configured. Please create a global rubric or include question-specific rubrics in your templates.",
                    )

            # Start verification
            job_id = verification_service.start_verification(
                finished_templates=finished_templates,
                config=config,
                question_ids=question_ids,
                run_name=run_name,
                async_config=async_config,
                storage_url=storage_url,  # Pass storage URL for auto-save
                benchmark_name=benchmark_name,  # Pass benchmark name for auto-save
            )

            # Get the job to return the actual run name (auto-generated if not provided)
            job_status = verification_service.get_job_status(job_id)
            actual_run_name = job_status.get("run_name", run_name) if job_status else run_name

            return {
                "job_id": job_id,
                "run_name": actual_run_name,
                "status": "started",
                "message": f"Verification '{actual_run_name}' started for {len(finished_templates)} templates",
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start verification: {e!s}") from e

    @app.get("/api/verification-progress/{job_id}")  # type: ignore[misc]
    async def get_verification_progress(job_id: str) -> dict[str, Any]:
        """Get verification progress."""
        try:
            progress = verification_service.get_progress(job_id)
            if not progress:
                raise HTTPException(status_code=404, detail="Job not found")

            return progress  # type: ignore[no-any-return]

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting verification progress: {e!s}") from e

    @app.websocket("/ws/verification-progress/{job_id}")  # type: ignore[misc]
    async def websocket_verification_progress(websocket: WebSocket, job_id: str) -> None:
        """WebSocket endpoint for real-time verification progress updates."""
        import asyncio

        # Validate job exists
        job = verification_service.jobs.get(job_id)
        if not job:
            await websocket.close(code=1008, reason="Job not found")
            return

        # Accept the connection
        await websocket.accept()

        # Set the event loop for the broadcaster if not already set
        if verification_service.broadcaster._event_loop is None:
            verification_service.broadcaster.set_event_loop(asyncio.get_running_loop())

        # Subscribe to progress updates
        await verification_service.broadcaster.subscribe(job_id, websocket)

        try:
            # Send current state immediately
            progress = verification_service.get_progress(job_id)
            if progress:
                await websocket.send_json(
                    {
                        "type": "snapshot",
                        "job_id": job_id,
                        "status": progress["status"],
                        "percentage": progress["percentage"],
                        "processed": progress["processed_count"],
                        "total": progress["total_questions"],
                        "in_progress_questions": progress.get("in_progress_questions", []),
                        "ema_seconds_per_item": progress.get("ema_seconds_per_item", 0),
                        "estimated_time_remaining": progress.get("estimated_time_remaining"),
                        "current_question": progress.get("current_question", ""),
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
            await verification_service.broadcaster.unsubscribe(job_id, websocket)

    @app.get("/api/verification-results/{job_id}")  # type: ignore[misc]
    async def get_verification_results(job_id: str) -> dict[str, Any]:
        """Get verification results."""
        try:
            results = verification_service.get_job_results(job_id)
            if not results:
                raise HTTPException(status_code=404, detail="Job not found or not completed")

            return {"results": results}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting verification results: {e!s}") from e

    @app.get("/api/all-verification-results")  # type: ignore[misc]
    async def get_all_verification_results() -> dict[str, Any]:
        """Get all historical verification results across all jobs."""
        try:
            results = verification_service.get_all_historical_results()
            return {"results": results}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting all verification results: {e!s}") from e

    @app.post("/api/cancel-verification/{job_id}")  # type: ignore[misc]
    async def cancel_verification_endpoint(job_id: str) -> dict[str, Any]:
        """Cancel verification job."""
        try:
            success = verification_service.cancel_job(job_id)
            if not success:
                raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")

            return {"message": "Job cancelled successfully"}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e!s}") from e

    @app.get("/api/export-verification/{job_id}")  # type: ignore[misc]
    async def export_verification_endpoint(job_id: str, fmt: str = "json") -> FileResponse:
        """Export verification results."""
        try:
            from karenina.benchmark.exporter import (
                create_export_filename,
                export_verification_results_csv,
                export_verification_results_json,
            )

            # Get job and results
            job = verification_service.jobs.get(job_id)
            if not job or job.status != "completed":
                raise HTTPException(status_code=404, detail="Job not found or not completed")

            results = verification_service.get_job_results(job_id)
            if not results:
                raise HTTPException(status_code=404, detail="No results available")

            # Get global rubric for CSV export
            global_rubric = None
            if fmt.lower() == "csv":
                from ..services.rubric_service import rubric_service

                global_rubric = rubric_service.get_current_rubric()

            # Export based on format
            if fmt.lower() == "csv":
                content = export_verification_results_csv(job, results, global_rubric)
                media_type = "text/csv"
            else:
                content = export_verification_results_json(job, results)
                media_type = "application/json"

            filename = create_export_filename(job, fmt.lower())

            # Create temporary file for download
            temp_dir = Path(tempfile.gettempdir()) / "otarbench_exports"
            temp_dir.mkdir(exist_ok=True)

            temp_file = temp_dir / filename
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)

            return FileResponse(path=temp_file, filename=filename, media_type=media_type)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error exporting results: {e!s}") from e
