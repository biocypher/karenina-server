"""Service for managing verification jobs with progress tracking."""

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from karenina.schemas.domain import Rubric
from karenina.schemas.workflow import (
    FinishedTemplate,
    VerificationConfig,
    VerificationJob,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.utils.checkpoint import generate_template_id

from .progress_broadcaster import ProgressBroadcaster

# Configure logging
logger = logging.getLogger(__name__)


class VerificationService:
    """Service for managing verification jobs."""

    def __init__(self, max_workers: int | None = None):
        # Use KARENINA_ASYNC_ENABLED and KARENINA_ASYNC_MAX_WORKERS to control concurrent jobs
        if max_workers is None:
            async_enabled = os.getenv("KARENINA_ASYNC_ENABLED", "true").lower() == "true"
            # Use max_workers from env (default 2) if async, otherwise 1 for sequential
            max_workers = int(os.getenv("KARENINA_ASYNC_MAX_WORKERS", "2")) if async_enabled else 1

        logger.info(f"🔧 VerificationService initializing with max_workers={max_workers}")
        logger.info(f"   KARENINA_ASYNC_ENABLED={os.getenv('KARENINA_ASYNC_ENABLED', 'not set')}")
        logger.info(f"   KARENINA_ASYNC_MAX_WORKERS={os.getenv('KARENINA_ASYNC_MAX_WORKERS', 'not set')}")

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, VerificationJob] = {}
        self.futures: dict[str, Any] = {}
        # Store all historical results keyed by job_id
        self.historical_results: dict[str, dict[str, VerificationResult]] = {}
        self.broadcaster = ProgressBroadcaster()

    def start_verification(
        self,
        finished_templates: list[FinishedTemplate],
        config: VerificationConfig,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        storage_url: str | None = None,
        benchmark_name: str | None = None,
    ) -> str:
        """Start a new verification job."""
        # Validate rubric availability if rubric evaluation is enabled
        if getattr(config, "rubric_enabled", False):
            from ..services.rubric_service import rubric_service

            if not rubric_service.has_any_rubric(finished_templates):
                raise ValueError(
                    "Rubric evaluation is enabled but no rubrics are configured. Please create a global rubric or include question-specific rubrics in your templates."
                )

        job_id = str(uuid.uuid4())

        # Auto-generate run name if not provided
        if not run_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            run_name = f"Run_{timestamp}"

        # Filter templates if specific question IDs are requested
        if question_ids:
            templates_to_verify = [t for t in finished_templates if t.question_id in question_ids]
        else:
            templates_to_verify = finished_templates

        # Calculate total combinations for progress tracking
        if hasattr(config, "answering_models") and config.answering_models:
            total_combinations = (
                len(templates_to_verify)
                * len(config.answering_models)
                * len(config.parsing_models)
                * config.replicate_count
            )
        else:
            # Legacy single model mode
            total_combinations = len(templates_to_verify) * getattr(config, "replicate_count", 1)

        # Create job
        job = VerificationJob(
            job_id=job_id,
            run_name=run_name,
            status="pending",
            config=config,
            total_questions=total_combinations,
            storage_url=storage_url,  # Store for auto-save
            benchmark_name=benchmark_name,  # Store benchmark name for auto-save
        )

        self.jobs[job_id] = job

        # Submit to thread pool
        logger.info(f"📋 Submitting verification job {job_id} (run_name={run_name}) to executor")
        logger.info(f"   Active jobs: {sum(1 for j in self.jobs.values() if j.status in ['pending', 'running'])}")
        future = self.executor.submit(self._run_verification, job, templates_to_verify)
        self.futures[job_id] = future

        return job_id

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get the status of a verification job."""
        job = self.jobs.get(job_id)
        return job.to_dict() if job else None

    def get_job_results(self, job_id: str) -> dict[str, VerificationResult] | None:
        """Get the results of a completed job."""
        job = self.jobs.get(job_id)
        if job and job.status == "completed":
            return job.results  # type: ignore[no-any-return]
        return None

    def get_all_historical_results(self) -> dict[str, VerificationResult]:
        """Get all historical results across all completed jobs."""
        all_results = {}
        for job_results in self.historical_results.values():
            all_results.update(job_results)
        return all_results

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a verification job."""
        job = self.jobs.get(job_id)
        if job and job.status in ["pending", "running"]:
            job.status = "cancelled"
            return True
        return False

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> None:
        """Clean up old jobs to prevent memory leaks."""
        current_time = time.time()
        jobs_to_remove = []

        for job_id, job in self.jobs.items():
            if job.start_time:
                age_hours = (current_time - job.start_time) / 3600
                if age_hours > max_age_hours:
                    jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            if job_id in self.futures:
                del self.futures[job_id]

    def _load_current_rubric(self) -> Rubric | None:
        """
        Load the current rubric from the service.

        Returns:
            The current rubric if available, None otherwise
        """
        try:
            from ..services.rubric_service import rubric_service

            return rubric_service.get_current_rubric()
        except Exception as e:
            logger.error(f"Failed to load rubric: {e}")
            return None

    # Helper methods are now in karenina.benchmark.verification.batch_runner
    # This service wraps the batch runner with job management and progress tracking

    def _run_verification(self, job: VerificationJob, templates: list[FinishedTemplate]) -> None:
        """Execute verification using batch runner with job management."""
        try:
            from karenina.benchmark.verification import run_verification_batch

            job.status = "running"
            job.start_time = time.time()

            # Load global rubric if needed
            global_rubric = self._load_current_rubric() if getattr(job.config, "rubric_enabled", False) else None

            # Emit job started
            self._emit_progress_event(job.job_id, "job_started")

            # Create progress callback to track task status and broadcast updates
            def progress_callback(current: int, total: int, result: VerificationResult | None) -> None:
                """Update job progress and broadcast to WebSocket subscribers."""
                if result:
                    # Distinguish between starting and completion events by timestamp
                    # Empty timestamp = starting, non-empty = completion
                    if not result.timestamp or result.timestamp == "":
                        # Task is starting
                        job.task_started(result.question_id)
                        job.current_question = result.question_id
                        job.percentage = ((current - 1) / total) * 100 if total > 0 else 0

                        # Broadcast task started event
                        self._emit_progress_event(
                            job.job_id,
                            "task_started",
                            {"question_id": result.question_id, "current": current, "total": total},
                        )
                    else:
                        # Task is finished
                        success = result.completed_without_errors

                        # task_finished now calculates duration internally from task_start_times
                        job.task_finished(result.question_id, success)
                        job.percentage = (current / total) * 100 if total > 0 else 0

                        # Broadcast task completed event
                        self._emit_progress_event(
                            job.job_id,
                            "task_completed",
                            {
                                "question_id": result.question_id,
                                "current": current,
                                "total": total,
                                "success": success,
                            },
                        )

            # Run verification using batch runner
            # Note: batch runner handles task generation, execution, and auto-save
            results = run_verification_batch(
                templates=templates,
                config=job.config,
                run_name=job.run_name,
                job_id=job.job_id,
                global_rubric=global_rubric,
                async_enabled=None,  # Let batch_runner read from env
                max_workers=None,  # Let batch_runner read from env
                storage_url=job.storage_url,
                benchmark_name=job.benchmark_name,
                progress_callback=progress_callback,
            )

            # Update job with results
            job.results = results
            job.processed_count = len(results)
            job.successful_count = sum(1 for r in results.values() if r.completed_without_errors)
            job.failed_count = len(results) - job.successful_count

            # Finalize
            job.status = "completed"
            job.end_time = time.time()
            job.percentage = 100.0
            job.last_task_duration = None  # Clear last task duration on completion
            job.in_progress_questions = []
            self.historical_results[job.job_id] = job.results.copy()

            self._emit_progress_event(job.job_id, "job_completed")

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = time.time()
            self._emit_progress_event(job.job_id, "job_failed", {"error": str(e)})

    def _create_error_results_for_template(
        self, job: VerificationJob, template: FinishedTemplate, error: Exception
    ) -> dict[str, VerificationResult]:
        """Create error results for all model combinations of a template."""
        from datetime import datetime

        error_results = {}

        # Create error for each combination including replicates
        for answering_model in job.config.answering_models:
            for parsing_model in job.config.parsing_models:
                # Create errors for all replicates
                for replicate in range(1, job.config.replicate_count + 1):
                    # For single replicate, don't include replicate numbers
                    if job.config.replicate_count == 1:
                        combination_id = f"{template.question_id}_{answering_model.id}_{parsing_model.id}"
                        answering_replicate = None
                        parsing_replicate = None
                    else:
                        # For multiple replicates, include replicate number in ID and track separately
                        combination_id = (
                            f"{template.question_id}_{answering_model.id}_{parsing_model.id}_rep{replicate}"
                        )
                        answering_replicate = replicate
                        parsing_replicate = replicate

                    # For OpenRouter interface, don't include provider in the model string
                    if answering_model.interface == "openrouter":
                        answering_model_str = answering_model.model_name
                    else:
                        answering_model_str = f"{answering_model.model_provider}/{answering_model.model_name}"

                    if parsing_model.interface == "openrouter":
                        parsing_model_str = parsing_model.model_name
                    else:
                        parsing_model_str = f"{parsing_model.model_provider}/{parsing_model.model_name}"

                    error_result = VerificationResult(
                        metadata=VerificationResultMetadata(
                            question_id=template.question_id,
                            template_id=generate_template_id(template.template_code),
                            completed_without_errors=False,
                            error=f"Verification error: {error!s}",
                            question_text=template.question_text,
                            keywords=template.keywords,
                            answering_model=answering_model_str,
                            parsing_model=parsing_model_str,
                            answering_system_prompt=answering_model.system_prompt,
                            parsing_system_prompt=parsing_model.system_prompt,
                            execution_time=0.0,
                            timestamp=datetime.now().isoformat(),
                            run_name=job.run_name,
                            job_id=job.job_id,
                            answering_replicate=answering_replicate,
                            parsing_replicate=parsing_replicate,
                        ),
                        template=VerificationResultTemplate(
                            raw_llm_response="",
                        ),
                    )
                    error_results[combination_id] = error_result

        return error_results

    # Auto-save is now handled by batch_runner.auto_save_results()

    def get_progress(self, job_id: str) -> dict[str, Any] | None:
        """Get progress information for a job."""
        job = self.jobs.get(job_id)
        if not job:
            return None

        # Calculate duration
        duration = None
        if job.start_time:
            duration = job.end_time - job.start_time if job.end_time else time.time() - job.start_time

        progress_data = {
            "job_id": job.job_id,
            "run_name": job.run_name,
            "status": job.status,
            "total_questions": job.total_questions,
            "processed_count": job.processed_count,
            "successful_count": job.successful_count,
            "failed_count": job.failed_count,
            "percentage": job.percentage,
            "current_question": job.current_question,
            "start_time": job.start_time,  # Unix timestamp for client-side live clock
            "duration_seconds": duration,
            "last_task_duration": job.last_task_duration,
            "error": job.error_message,
            "in_progress_questions": job.in_progress_questions,
        }

        # Include results if completed
        if job.status == "completed":
            progress_data["results"] = job.results

        return progress_data

    def _emit_progress_event(self, job_id: str, event_type: str, extra_data: dict[str, Any] | None = None) -> None:
        """Emit a progress event to WebSocket subscribers."""
        job = self.jobs.get(job_id)
        if not job:
            return

        # Calculate current duration
        duration = None
        if job.start_time:
            duration = job.end_time - job.start_time if job.end_time else time.time() - job.start_time

        event_data = {
            "type": event_type,
            "job_id": job_id,
            "status": job.status,
            "percentage": job.percentage,
            "processed": job.processed_count,
            "total": job.total_questions,
            "in_progress_questions": job.in_progress_questions,
            "start_time": job.start_time,  # Send start_time for client-side elapsed time calculation
            "duration_seconds": duration,
            "last_task_duration": job.last_task_duration,
            "current_question": job.current_question,
        }

        if extra_data:
            event_data.update(extra_data)

        # Broadcast from thread (thread-safe)
        self.broadcaster.broadcast_from_thread(job_id, event_data)


# Global service instance
verification_service = VerificationService()
