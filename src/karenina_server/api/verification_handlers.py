"""Benchmark verification API handlers."""

from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..constants import TEMP_EXPORT_DIR
from ..utils.rubric_utils import build_rubric_from_dict

if TYPE_CHECKING:
    from karenina.schemas.workflow import VerificationResult

    from ..services.verification_service import VerificationService

# Configure logging
logger = logging.getLogger(__name__)


def _deduplicate_results_by_replicate(
    matching_results: list[VerificationResult],
) -> list[VerificationResult]:
    """Deduplicate verification results by replicate number, keeping the most relevant.

    Groups results by replicate number and selects one result per replicate.
    Priority: Show errors first (so users can see failures), otherwise
    show the most recent successful result.

    Args:
        matching_results: List of VerificationResult objects for a single question/model.

    Returns:
        List of deduplicated results, one per replicate number.
    """
    replicate_groups: dict[int, list[VerificationResult]] = defaultdict(list)
    for result in matching_results:
        replicate_num = result.metadata.replicate or 0
        replicate_groups[replicate_num].append(result)

    deduplicated = []
    for replicate_num in sorted(replicate_groups.keys()):
        candidates = replicate_groups[replicate_num]
        failed_results = [r for r in candidates if not r.metadata.completed_without_errors]

        if failed_results:
            # Show the most recent failure
            selected = max(failed_results, key=lambda r: r.metadata.timestamp or "")
        else:
            # Show the most recent successful attempt
            selected = max(candidates, key=lambda r: r.metadata.timestamp or "")

        deduplicated.append(selected)

    return deduplicated


def _build_heatmap_cell(result: VerificationResult) -> dict[str, Any]:
    """Build a heatmap cell data dict from a VerificationResult.

    Extracts template pass/fail, rubric score, abstention status, execution
    metadata, token usage, and rubric trait scores for display in heatmap.

    Args:
        result: A VerificationResult object.

    Returns:
        Dict with cell data for heatmap display.
    """
    cell_data: dict[str, Any] = {
        "replicate": result.metadata.replicate,
        "passed": None,
        "score": None,
        "abstained": False,
        "insufficient": False,
        "error": result.metadata.error is not None,
    }

    # Template verification status
    if result.template and hasattr(result.template, "verify_result"):
        cell_data["passed"] = result.template.verify_result

    if result.template and hasattr(result.template, "abstention_detected"):
        cell_data["abstained"] = result.template.abstention_detected or False

    if result.template and hasattr(result.template, "sufficiency_detected"):
        cell_data["insufficient"] = result.template.sufficiency_detected is False

    # Rubric score
    if result.rubric and hasattr(result.rubric, "overall_score"):
        cell_data["score"] = result.rubric.overall_score

    # Execution type
    has_agent = (
        result.template and hasattr(result.template, "agent_metrics") and result.template.agent_metrics is not None
    )
    cell_data["execution_type"] = "Agent" if has_agent else "Standard"

    # Token usage
    if result.template and hasattr(result.template, "usage_metadata") and result.template.usage_metadata:
        total_usage = result.template.usage_metadata.get("total", {})
        inp = total_usage.get("input_tokens", 0)
        out = total_usage.get("output_tokens", 0)
        cell_data["input_tokens"] = int(inp) if inp is not None and isinstance(inp, int | float) else 0
        cell_data["output_tokens"] = int(out) if out is not None and isinstance(out, int | float) else 0
    else:
        cell_data["input_tokens"] = 0
        cell_data["output_tokens"] = 0

    # Agent iterations
    if has_agent:
        cell_data["iterations"] = result.template.agent_metrics.get("iterations", 0)
    else:
        cell_data["iterations"] = 0

    # Rubric trait scores for badge overlays
    if result.rubric:
        rubric_scores: dict[str, dict[str, bool | int | float]] = {}
        if hasattr(result.rubric, "llm_trait_scores") and result.rubric.llm_trait_scores:
            rubric_scores["llm"] = result.rubric.llm_trait_scores
        if hasattr(result.rubric, "regex_trait_scores") and result.rubric.regex_trait_scores:
            rubric_scores["regex"] = result.rubric.regex_trait_scores
        if hasattr(result.rubric, "callable_trait_scores") and result.rubric.callable_trait_scores:
            rubric_scores["callable"] = result.rubric.callable_trait_scores
        if rubric_scores:
            cell_data["rubric_scores"] = rubric_scores

    return cell_data


def _compute_token_stats(results: list[VerificationResult], question_id: str) -> dict[str, float]:
    """Compute token usage statistics for a question across replicates.

    Args:
        results: List of VerificationResult objects for a model.
        question_id: The question ID to filter by.

    Returns:
        Dict with input/output token median and std dev.
    """
    import numpy as np

    matching = [r for r in results if r.metadata.question_id == question_id]

    input_tokens = []
    output_tokens = []

    for r in matching:
        if (
            r.template
            and hasattr(r.template, "usage_metadata")
            and r.template.usage_metadata
            and "total" in r.template.usage_metadata
        ):
            total_usage = r.template.usage_metadata["total"]
            inp = total_usage.get("input_tokens", 0)
            out = total_usage.get("output_tokens", 0)
            if inp is not None and isinstance(inp, int | float) and inp > 0:
                input_tokens.append(inp)
            if out is not None and isinstance(out, int | float) and out > 0:
                output_tokens.append(out)

    return {
        "input_median": float(np.median(input_tokens)) if input_tokens else 0.0,
        "input_std": float(np.std(input_tokens)) if input_tokens else 0.0,
        "output_median": float(np.median(output_tokens)) if output_tokens else 0.0,
        "output_std": float(np.std(output_tokens)) if output_tokens else 0.0,
    }


def register_verification_routes(app: FastAPI, verification_service: VerificationService) -> None:
    """Register verification-related routes.

    Args:
        app: The FastAPI application instance to register routes on.
        verification_service: The verification service for running verification jobs.
    """

    @app.get("/api/finished-templates")
    async def get_finished_templates_endpoint() -> dict[str, Any]:
        """Get list of finished templates for verification."""
        try:
            # This is a placeholder - in a real implementation, you'd get this from your data store
            # For now, return empty list since we don't have access to the checkpoint data here
            return {"finished_templates": []}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting finished templates: {e!s}") from e

    @app.post("/api/start-verification")
    async def start_verification_endpoint(request: dict[str, Any]) -> dict[str, Any]:
        """Start verification job."""
        try:
            import json

            from karenina.schemas.workflow import FinishedTemplate, VerificationConfig

            # Parse request
            config_data = request.get("config", {})
            question_ids = request.get("question_ids")
            finished_templates_data = request.get("finished_templates", [])
            run_name = request.get("run_name")  # Optional user-defined run name
            storage_url = request.get("storage_url")  # Optional database URL for auto-save
            benchmark_name = request.get("benchmark_name")  # Optional benchmark name for auto-save

            # Log verification request details for debugging
            logger.debug(
                "Received verification request",
                extra={
                    "rubric_enabled": config_data.get("rubric_enabled", False),
                    "template_count": len(finished_templates_data),
                },
            )

            # Check if any templates have metric traits
            templates_with_metric_traits = [
                t
                for t in finished_templates_data
                if t.get("question_rubric") and t.get("question_rubric", {}).get("metric_traits")
            ]
            logger.debug(
                "Templates with metric traits: %d / %d",
                len(templates_with_metric_traits),
                len(finished_templates_data),
            )

            if templates_with_metric_traits:
                sample = templates_with_metric_traits[0]
                logger.debug(
                    "Sample metric trait: %s",
                    json.dumps(sample["question_rubric"]["metric_traits"][0], indent=2),
                )

            # Create config
            config = VerificationConfig(**config_data)

            # Create finished templates (needed for rubric validation)
            finished_templates = [FinishedTemplate(**template_data) for template_data in finished_templates_data]

            # Convert question_rubric dicts to Rubric objects using shared helper
            for template in finished_templates:
                if template.question_rubric:
                    rubric = build_rubric_from_dict(template.question_rubric)
                    if rubric:
                        template.question_rubric = rubric

            # Log parsed templates for debugging
            templates_with_metric_traits_parsed = [
                t
                for t in finished_templates
                if t.question_rubric and hasattr(t.question_rubric, "metric_traits") and t.question_rubric.metric_traits
            ]
            logger.debug("Parsed templates with metric traits: %d", len(templates_with_metric_traits_parsed))
            if templates_with_metric_traits_parsed:
                sample = templates_with_metric_traits_parsed[0]
                logger.debug(
                    "Sample parsed metric trait: name=%s, evaluation_mode=%s",
                    sample.question_rubric.metric_traits[0].name,
                    sample.question_rubric.metric_traits[0].evaluation_mode,
                )

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

    @app.get("/api/verification-progress/{job_id}")
    async def get_verification_progress(job_id: str) -> dict[str, Any]:
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

    @app.websocket("/ws/verification-progress/{job_id}")
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
                        "start_time": progress.get("start_time"),  # Unix timestamp for client-side live clock
                        "duration_seconds": progress.get("duration_seconds"),
                        "last_task_duration": progress.get("last_task_duration"),
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

    @app.get("/api/verification-results/{job_id}")
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

    @app.get("/api/all-verification-results")
    async def get_all_verification_results() -> dict[str, Any]:
        """Get all historical verification results across all jobs."""
        try:
            results = verification_service.get_all_historical_results()
            return {"results": results}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting all verification results: {e!s}") from e

    @app.post("/api/cancel-verification/{job_id}")
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

    @app.get("/api/export-verification/{job_id}")
    async def export_verification_endpoint(job_id: str, fmt: str = "json") -> FileResponse:
        """Export verification results."""
        try:
            from karenina.benchmark.verification.results_exporter import (
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

            # Get global rubric for export (needed for both CSV and JSON)
            from ..services.rubric_service import rubric_service

            global_rubric = rubric_service.get_current_rubric()

            # Export based on format
            if fmt.lower() == "csv":
                content = export_verification_results_csv(job, results, global_rubric)
                media_type = "text/csv"
            else:
                content = export_verification_results_json(job, results, global_rubric)
                media_type = "application/json"

            filename = create_export_filename(job, fmt.lower())

            # Create temporary file for download
            temp_dir = Path(tempfile.gettempdir()) / TEMP_EXPORT_DIR
            temp_dir.mkdir(exist_ok=True)

            temp_file = temp_dir / filename
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)

            return FileResponse(path=temp_file, filename=filename, media_type=media_type)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error exporting results: {e!s}") from e

    @app.post("/api/verification/summary")
    async def compute_summary_endpoint(request: dict[str, Any]) -> dict[str, Any]:
        """
        Compute summary statistics for verification results.

        Request body:
            - results: Dict of verification results (result_id -> VerificationResult)
            - run_name: Optional run name to filter by (null for all results)

        Returns:
            Dictionary with summary statistics from VerificationResultSet.get_summary()
        """
        try:
            from karenina.schemas.workflow import VerificationResult, VerificationResultSet

            # Parse request
            results_dict = request.get("results", {})
            run_name_filter = request.get("run_name")

            # Convert dict to list of VerificationResult objects
            results_list = []
            for result_id, result_data in results_dict.items():
                try:
                    result = VerificationResult(**result_data)
                    # Filter by run_name if specified
                    if run_name_filter is None or result.metadata.run_name == run_name_filter:
                        results_list.append(result)
                except Exception as e:
                    logger.warning("Failed to parse result %s: %s", result_id, e)
                    continue

            if not results_list:
                raise HTTPException(
                    status_code=400,
                    detail=f"No valid results found{f' for run_name={run_name_filter}' if run_name_filter else ''}",
                )

            # Create VerificationResultSet and compute summary
            result_set = VerificationResultSet(results=results_list)
            summary: dict[str, Any] = result_set.get_summary()

            return summary

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error computing summary: {e!s}") from e

    @app.post("/api/verification/compare-models")
    async def compare_models_endpoint(request: dict[str, Any]) -> dict[str, Any]:
        """
        Compare multiple models with per-model summaries and heatmap data.

        Request body:
            - results: Dict of verification results (result_id -> VerificationResult)
            - models: List of model configs to compare [{answering_model, mcp_config}]
            - parsing_model: Parsing model to filter by (same judge for fair comparison)
            - replicate: Optional replicate number to filter by (for per-replicate comparison)

        Returns:
            - model_summaries: Dict mapping model key to summary stats
            - heatmap_data: List of questions with results by model
        """
        try:
            from karenina.schemas.workflow import VerificationResult, VerificationResultSet

            # Parse request
            results_dict = request.get("results", {})
            models_to_compare = request.get("models", [])
            parsing_model_filter = request.get("parsing_model")

            if not models_to_compare:
                raise HTTPException(status_code=400, detail="At least one model must be specified")

            # Convert dict to list of VerificationResult objects
            all_results = []

            for result_id, result_data in results_dict.items():
                try:
                    result = VerificationResult(**result_data)
                    # Filter by parsing model if specified
                    if parsing_model_filter is not None and result.metadata.parsing_model != parsing_model_filter:
                        continue

                    all_results.append(result)

                except Exception as e:
                    logger.warning("Failed to parse result %s: %s", result_id, e)
                    continue

            if not all_results:
                raise HTTPException(status_code=400, detail="No valid results found")

            # Group results by model
            model_results = {}

            for model_config in models_to_compare:
                answering_model = model_config.get("answering_model")
                mcp_config_str = str(model_config.get("mcp_config", "[]"))  # JSON string from frontend
                model_key = f"{answering_model}|{mcp_config_str}"

                # Parse expected MCP servers from config
                import json

                try:
                    expected_mcp_servers = json.loads(mcp_config_str)
                    if not isinstance(expected_mcp_servers, list):
                        expected_mcp_servers = []
                except Exception:
                    expected_mcp_servers = []

                # Sort for consistent comparison
                expected_mcp_servers_sorted = sorted(expected_mcp_servers)

                # Filter results for this model
                filtered = []
                for r in all_results:
                    if r.metadata.answering_model == answering_model:
                        # Get MCP servers from result
                        result_mcp_servers: list[str] = []
                        if r.template and hasattr(r.template, "answering_mcp_servers"):
                            result_mcp_servers = r.template.answering_mcp_servers or []

                        # Sort for comparison
                        result_mcp_servers_sorted = sorted(result_mcp_servers)

                        # Match if MCP servers are the same
                        if result_mcp_servers_sorted == expected_mcp_servers_sorted:
                            filtered.append(r)

                if filtered:
                    model_results[model_key] = filtered

            if not model_results:
                raise HTTPException(status_code=400, detail="No results found for specified models")

            # Compute per-model summaries using all replicates
            model_summaries = {}
            for model_key, results_list in model_results.items():
                result_set = VerificationResultSet(results=results_list)
                summary = result_set.get_summary()
                model_summaries[model_key] = summary

            # Generate heatmap data (question x model matrix) with all replicates
            # Collect all unique questions with their keywords
            questions_map = {}  # question_id -> (question_text, keywords)
            for result in all_results:
                if result.metadata.question_id not in questions_map:
                    questions_map[result.metadata.question_id] = (
                        result.metadata.question_text,
                        result.metadata.keywords or [],
                    )

            heatmap_data = []
            for question_id, (question_text, keywords) in questions_map.items():
                question_row = {
                    "question_id": question_id,
                    "question_text": question_text,
                    "keywords": keywords,
                    "results_by_model": {},
                }

                # For each model, find all replicates for this question
                for model_key, results_list in model_results.items():
                    matching_results = [r for r in results_list if r.metadata.question_id == question_id]

                    if matching_results:
                        # Deduplicate and build cell data using helper functions
                        deduplicated_results = _deduplicate_results_by_replicate(matching_results)
                        replicates_data = [_build_heatmap_cell(r) for r in deduplicated_results]
                        question_row["results_by_model"][model_key] = {"replicates": replicates_data}
                    else:
                        question_row["results_by_model"][model_key] = {"replicates": []}

                heatmap_data.append(question_row)

            # Generate per-question token data for bar charts using helper function
            question_token_data = []
            for question_id, (question_text, _keywords) in questions_map.items():
                question_data = {
                    "question_id": question_id,
                    "question_text": question_text[:50] + ("..." if len(question_text) > 50 else ""),
                    "models": [],
                }

                for model_key, results_list in model_results.items():
                    stats = _compute_token_stats(results_list, question_id)
                    if stats["input_median"] > 0 or stats["output_median"] > 0:
                        parts = model_key.split("|")
                        model_display_name = f"{parts[0]} (MCP: {parts[1]})" if len(parts) >= 2 else parts[0]
                        question_data["models"].append(
                            {
                                "model_key": model_key,
                                "model_display_name": model_display_name,
                                "input_tokens_median": stats["input_median"],
                                "input_tokens_std": stats["input_std"],
                                "output_tokens_median": stats["output_median"],
                                "output_tokens_std": stats["output_std"],
                            }
                        )

                if question_data["models"]:
                    question_token_data.append(question_data)

            return {
                "model_summaries": model_summaries,
                "heatmap_data": heatmap_data,
                "question_token_data": question_token_data,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error comparing models: {e!s}") from e
