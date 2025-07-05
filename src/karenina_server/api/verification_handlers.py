"""Benchmark verification API handlers."""

import tempfile
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse


def register_verification_routes(app, verification_service):
    """Register verification-related routes."""

    @app.get("/api/finished-templates")
    async def get_finished_templates_endpoint():
        """Get list of finished templates for verification."""
        try:
            # This is a placeholder - in a real implementation, you'd get this from your data store
            # For now, return empty list since we don't have access to the checkpoint data here
            return {"finished_templates": []}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting finished templates: {e!s}") from e

    @app.post("/api/start-verification")
    async def start_verification_endpoint(request: dict):
        """Start verification job."""
        try:
            from karenina.benchmark.models import FinishedTemplate, VerificationConfig

            # Parse request
            config_data = request.get("config", {})
            question_ids = request.get("question_ids")
            finished_templates_data = request.get("finished_templates", [])
            run_name = request.get("run_name")  # Optional user-defined run name

            # Create config
            config = VerificationConfig(**config_data)

            # Create finished templates (needed for rubric validation)
            finished_templates = [FinishedTemplate(**template_data) for template_data in finished_templates_data]

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
                finished_templates=finished_templates, config=config, question_ids=question_ids, run_name=run_name
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

    @app.get("/api/verification-progress/{job_id}")
    async def get_verification_progress(job_id: str):
        """Get verification progress."""
        try:
            progress = verification_service.get_progress(job_id)
            if not progress:
                raise HTTPException(status_code=404, detail="Job not found")

            return progress

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting verification progress: {e!s}") from e

    @app.get("/api/verification-results/{job_id}")
    async def get_verification_results(job_id: str):
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

    @app.get("/api/all-verification-results")
    async def get_all_verification_results():
        """Get all historical verification results across all jobs."""
        try:
            results = verification_service.get_all_historical_results()
            return {"results": results}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting all verification results: {e!s}") from e

    @app.post("/api/cancel-verification/{job_id}")
    async def cancel_verification_endpoint(job_id: str):
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

    @app.get("/api/export-verification/{job_id}")
    async def export_verification_endpoint(job_id: str, fmt: str = "json"):
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

            # Export based on format
            if fmt.lower() == "csv":
                content = export_verification_results_csv(job, results)
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
