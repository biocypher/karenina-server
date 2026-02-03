"""Service for managing verification jobs with progress tracking."""

import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Any

from karenina.schemas.domain import Rubric
from karenina.schemas.verification import ModelIdentity
from karenina.schemas.workflow import (
    FinishedTemplate,
    VerificationConfig,
    VerificationJob,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultSet,
    VerificationResultTemplate,
)
from karenina.utils.checkpoint import generate_template_id

from .progress_broadcaster import ProgressBroadcaster

# Configure logging
logger = logging.getLogger(__name__)

# Cleanup configuration constants
_CLEANUP_INTERVAL_SECONDS = 3600  # Run cleanup at most once per hour
_MAX_JOB_AGE_HOURS = 24  # Remove jobs older than 24 hours
_MAX_HISTORICAL_RESULTS = 1000  # Keep at most 1000 historical result sets


class JobStatus(Enum):
    """Enum for job status values.

    Provides type-safe status values and enables validation of status transitions.
    Valid transitions:
        PENDING -> RUNNING
        RUNNING -> COMPLETED | FAILED | CANCELLED
        PENDING -> CANCELLED
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def valid_transitions(cls) -> dict["JobStatus", set["JobStatus"]]:
        """Return valid state transitions."""
        return {
            cls.PENDING: {cls.RUNNING, cls.CANCELLED},
            cls.RUNNING: {cls.COMPLETED, cls.FAILED, cls.CANCELLED},
            cls.COMPLETED: set(),  # Terminal state
            cls.FAILED: set(),  # Terminal state
            cls.CANCELLED: set(),  # Terminal state
        }

    def can_transition_to(self, new_status: "JobStatus") -> bool:
        """Check if transition to new_status is valid."""
        return new_status in self.valid_transitions().get(self, set())


class VerificationService:
    """Service for managing verification jobs.

    Lock Hierarchy (conc-005):
        1. _shutdown_lock - Service-level shutdown coordination
        2. _master_lock - Protects _job_locks dict modification
        3. _job_locks[job_id] - Per-job state protection

    Always acquire locks in this order to prevent deadlocks.
    """

    def __init__(self, max_workers: int | None = None):
        """Initialize the verification service.

        Args:
            max_workers: Maximum concurrent verification jobs. If None, reads from
                KARENINA_ASYNC_MAX_WORKERS env var (default 2) when KARENINA_ASYNC_ENABLED
                is true, otherwise uses 1 for sequential execution.
        """
        # Use KARENINA_ASYNC_ENABLED and KARENINA_ASYNC_MAX_WORKERS to control concurrent jobs
        if max_workers is None:
            async_enabled = os.getenv("KARENINA_ASYNC_ENABLED", "true").lower() == "true"
            # Use max_workers from env (default 2) if async, otherwise 1 for sequential
            max_workers = int(os.getenv("KARENINA_ASYNC_MAX_WORKERS", "2")) if async_enabled else 1

        logger.info(f"🔧 VerificationService initializing with max_workers={max_workers}")
        logger.info(f"   KARENINA_ASYNC_ENABLED={os.getenv('KARENINA_ASYNC_ENABLED', 'not set')}")
        logger.info(f"   KARENINA_ASYNC_MAX_WORKERS={os.getenv('KARENINA_ASYNC_MAX_WORKERS', 'not set')}")

        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="verify-")
        self.jobs: dict[str, VerificationJob] = {}
        self.futures: dict[str, Any] = {}
        # Store all historical results keyed by job_id (OrderedDict for LRU eviction)
        self.historical_results: OrderedDict[str, VerificationResultSet] = OrderedDict()
        self.broadcaster = ProgressBroadcaster()
        # Track last cleanup time to avoid excessive cleanup checks
        self._last_cleanup_time = time.time()
        # Per-job locks for thread-safe status access (conc-002)
        self._job_locks: dict[str, threading.Lock] = {}
        # Master lock for job_locks dict modification
        self._master_lock = threading.Lock()
        # Shutdown coordination (conc-001)
        self._shutdown_event = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False

    def _get_job_lock(self, job_id: str) -> threading.Lock:
        """Get or create a lock for a specific job.

        Thread-safe: uses master lock to protect job_locks dict modification.
        """
        with self._master_lock:
            if job_id not in self._job_locks:
                self._job_locks[job_id] = threading.Lock()
            return self._job_locks[job_id]

    def _remove_job_lock(self, job_id: str) -> None:
        """Remove a job's lock when the job is cleaned up."""
        with self._master_lock:
            self._job_locks.pop(job_id, None)

    def shutdown(self, wait: bool = True, cancel_pending: bool = False) -> None:
        """Gracefully shut down the verification service (conc-001).

        This method ensures clean shutdown of the ThreadPoolExecutor,
        preventing resource leaks and orphaned threads.

        Args:
            wait: If True, block until all running jobs complete.
                  If False, return immediately after signaling shutdown.
            cancel_pending: If True, attempt to cancel pending (not yet started) futures.
                           Running jobs will still complete unless interrupted.

        Thread-safety: This method is thread-safe and idempotent.
        """
        with self._shutdown_lock:
            if self._is_shutdown:
                logger.debug("VerificationService.shutdown() called but already shut down")
                return

            logger.info("🛑 VerificationService shutting down...")
            self._is_shutdown = True
            self._shutdown_event.set()

            # Optionally cancel pending futures
            cancelled_count = 0
            if cancel_pending:
                for job_id, future in list(self.futures.items()):
                    if future.cancel():
                        cancelled_count += 1
                        # Mark job as cancelled
                        job = self.jobs.get(job_id)
                        if job and job.status == "pending":
                            self._force_status(job_id, JobStatus.CANCELLED)
                if cancelled_count > 0:
                    logger.info(f"   Cancelled {cancelled_count} pending jobs")

            # Count in-flight jobs
            running_jobs = sum(1 for j in self.jobs.values() if j.status == "running")
            if running_jobs > 0:
                logger.info(f"   Waiting for {running_jobs} running job(s) to complete...")

        # Shutdown executor (outside lock to avoid blocking other operations)
        # wait=True ensures running tasks complete; wait=False returns immediately
        self.executor.shutdown(wait=wait, cancel_futures=cancel_pending)
        logger.info("✓ VerificationService shutdown complete")

    def is_shutdown(self) -> bool:
        """Check if the service has been shut down."""
        return self._is_shutdown

    def _get_job_status(self, job_id: str) -> JobStatus | None:
        """Get job status in a thread-safe manner.

        Returns:
            JobStatus enum value or None if job doesn't exist.
        """
        job = self.jobs.get(job_id)
        if not job:
            return None

        with self._get_job_lock(job_id):
            return JobStatus(job.status)

    def _transition_status(self, job_id: str, expected: JobStatus, new: JobStatus) -> bool:
        """Atomically transition job status with validation.

        Args:
            job_id: The job to update
            expected: The expected current status (for optimistic locking)
            new: The new status to transition to

        Returns:
            True if transition succeeded, False if:
            - Job doesn't exist
            - Current status != expected
            - Transition is invalid per state machine
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Cannot transition status: job {job_id} not found")
            return False

        with self._get_job_lock(job_id):
            current = JobStatus(job.status)

            # Check expected status matches
            if current != expected:
                logger.warning(
                    f"Status transition failed for job {job_id}: expected {expected.value}, found {current.value}"
                )
                return False

            # Validate transition
            if not current.can_transition_to(new):
                logger.warning(f"Invalid status transition for job {job_id}: {current.value} -> {new.value}")
                return False

            # Perform transition
            job.status = new.value
            logger.debug(f"Job {job_id} status: {current.value} -> {new.value}")
            return True

    def _force_status(self, job_id: str, new: JobStatus) -> bool:
        """Force-set job status without transition validation.

        Use sparingly - only for error recovery scenarios.
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        with self._get_job_lock(job_id):
            old_status = job.status
            job.status = new.value
            logger.debug(f"Job {job_id} status forced: {old_status} -> {new.value}")
            return True

    def _handle_job_completion(self, job_id: str, future: Future[None]) -> None:
        """Handle job completion callback from ThreadPoolExecutor (conc-004).

        This callback is invoked when the future completes (success or failure).
        It ensures that any unhandled exceptions in the worker thread are surfaced
        and the job is properly marked as failed, preventing jobs from hanging
        indefinitely in 'running' state.

        Args:
            job_id: The job identifier
            future: The completed Future object
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Job completion callback: job {job_id} not found")
            return

        # Check if the future raised an exception
        try:
            # This will re-raise any exception from the worker thread
            future.result(timeout=0)  # Non-blocking since future is already done
        except Exception as e:
            # Log the full exception with stack trace
            logger.error(
                f"Job {job_id} worker thread raised an unhandled exception: {e}",
                exc_info=True,
            )

            # Check if job is already in a terminal state (may have been handled in _run_verification)
            current_status = self._get_job_status(job_id)
            if current_status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                logger.debug(
                    f"Job {job_id} already in terminal state {current_status.value}, "
                    "exception was likely handled in _run_verification"
                )
                return

            # Job is still in RUNNING state but worker died - mark as failed
            with self._get_job_lock(job_id):
                job.error_message = f"Worker thread exception: {e}"
                job.end_time = time.time()

            # Force status to FAILED since the job may be in an unexpected state
            self._force_status(job_id, JobStatus.FAILED)

            # Emit job_failed event so clients are notified
            self._emit_progress_event(job_id, "job_failed", {"error": str(e)})

            logger.info(f"Job {job_id} marked as failed due to worker thread exception")

    def start_verification(
        self,
        finished_templates: list[FinishedTemplate],
        config: VerificationConfig,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        storage_url: str | None = None,
        benchmark_name: str | None = None,
    ) -> str:
        """Start a new verification job.

        Submits verification work to the thread pool and returns immediately.
        Progress can be tracked via get_progress() or WebSocket subscription.

        Args:
            finished_templates: List of templates to verify.
            config: Verification configuration (models, replicates, etc.).
            question_ids: Optional list to filter which questions to verify.
            run_name: Optional name for this run. Auto-generated if not provided.
            storage_url: Optional database URL for auto-saving results.
            benchmark_name: Optional benchmark name for auto-save metadata.

        Returns:
            Job ID string (UUID) for tracking this verification job.

        Raises:
            RuntimeError: If the service is shutting down.
            ValueError: If rubric evaluation is enabled but no rubrics exist.
        """
        # Check if service is shutting down (conc-001)
        if self._is_shutdown:
            raise RuntimeError("VerificationService is shutting down, cannot accept new jobs")

        # Opportunistically run cleanup to prevent memory leaks
        self._maybe_cleanup()

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
            # Validate that at least one requested question_id exists in templates
            if not templates_to_verify:
                available_ids = [t.question_id for t in finished_templates]
                raise ValueError(
                    f"None of the requested question_ids exist in finished_templates. "
                    f"Requested: {question_ids}, Available: {available_ids}"
                )
        else:
            templates_to_verify = finished_templates

        # Validate we have templates to verify
        if not templates_to_verify:
            raise ValueError("No templates to verify. finished_templates is empty.")

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

        # Add done callback to catch unhandled exceptions (conc-004)
        # This ensures jobs don't hang in 'running' state if the worker thread fails
        future.add_done_callback(lambda f: self._handle_job_completion(job_id, f))

        return job_id

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get the status of a verification job.

        Thread-safe. Opportunistically runs cleanup to prevent memory leaks.

        Args:
            job_id: The job identifier returned by start_verification().

        Returns:
            Dict with job status fields (job_id, status, percentage, etc.),
            or None if the job doesn't exist.
        """
        # Opportunistically run cleanup to prevent memory leaks
        self._maybe_cleanup()

        job = self.jobs.get(job_id)
        if not job:
            return None

        # Use job lock to ensure consistent read of all fields
        with self._get_job_lock(job_id):
            result: dict[str, Any] = job.to_dict()
            return result

    def get_job_results(self, job_id: str) -> VerificationResultSet | None:
        """Get the results of a completed job.

        Thread-safe. Only returns results for jobs with status 'completed'.

        Args:
            job_id: The job identifier returned by start_verification().

        Returns:
            VerificationResultSet containing all results if job is completed,
            or None if job doesn't exist or is not yet completed.
        """
        job = self.jobs.get(job_id)
        if not job:
            return None

        # Use job lock to check status and get results atomically
        with self._get_job_lock(job_id):
            if job.status == "completed":
                return job.result_set
            return None

    def get_all_historical_results(self) -> VerificationResultSet:
        """Get all historical results across all completed jobs.

        Combines results from all jobs in the historical cache (up to
        MAX_HISTORICAL_RESULTS entries, with LRU eviction).

        Returns:
            VerificationResultSet containing all historical results.
            May be empty if no jobs have completed.
        """
        all_results = []
        for result_set in self.historical_results.values():
            all_results.extend(result_set.results)
        return VerificationResultSet(results=all_results)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a verification job (thread-safe).

        Uses atomic status transition to safely cancel from PENDING or RUNNING states.
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        # Try to cancel from PENDING state first, then RUNNING state
        return self._transition_status(job_id, JobStatus.PENDING, JobStatus.CANCELLED) or self._transition_status(
            job_id, JobStatus.RUNNING, JobStatus.CANCELLED
        )

    def cleanup_old_jobs(self, max_age_hours: int = _MAX_JOB_AGE_HOURS) -> int:
        """Clean up old jobs to prevent memory leaks.

        Args:
            max_age_hours: Maximum age in hours before a job is removed

        Returns:
            Number of jobs removed
        """
        current_time = time.time()
        jobs_to_remove = []

        for job_id, job in self.jobs.items():
            if job.start_time:
                age_hours = (current_time - job.start_time) / 3600
                if age_hours > max_age_hours:
                    jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            self.futures.pop(job_id, None)  # Use pop() for atomic removal
            self.historical_results.pop(job_id, None)  # Also clean historical results
            self._remove_job_lock(job_id)  # Clean up per-job lock

        return len(jobs_to_remove)

    def _cleanup_historical_results(self) -> int:
        """Trim historical_results to stay within the max limit.

        Uses LRU eviction - removes oldest entries first.

        Returns:
            Number of result sets removed
        """
        removed_count = 0
        while len(self.historical_results) > _MAX_HISTORICAL_RESULTS:
            # Remove oldest entry (first item in OrderedDict)
            self.historical_results.popitem(last=False)
            removed_count += 1
        return removed_count

    def _maybe_cleanup(self) -> None:
        """Run cleanup if sufficient time has passed since last cleanup.

        This method is called opportunistically during normal operations
        to prevent unbounded memory growth without requiring a background thread.
        """
        current_time = time.time()
        if current_time - self._last_cleanup_time >= _CLEANUP_INTERVAL_SECONDS:
            jobs_removed = self.cleanup_old_jobs()
            results_removed = self._cleanup_historical_results()
            self._last_cleanup_time = current_time

            if jobs_removed > 0 or results_removed > 0:
                logger.info(
                    f"🧹 Periodic cleanup: removed {jobs_removed} old jobs, "
                    f"trimmed {results_removed} historical results"
                )

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
        """Execute verification using batch runner with job management (thread-safe)."""
        try:
            from karenina.benchmark.verification import run_verification_batch

            # Thread-safe status transition: PENDING -> RUNNING
            if not self._transition_status(job.job_id, JobStatus.PENDING, JobStatus.RUNNING):
                # Job may have been cancelled before we started
                logger.warning(f"Job {job.job_id} could not transition to RUNNING, aborting")
                return

            # Set start_time under lock for consistency
            with self._get_job_lock(job.job_id):
                job.start_time = time.time()

            # Load global rubric if needed
            global_rubric = self._load_current_rubric() if getattr(job.config, "rubric_enabled", False) else None

            # Emit job started
            self._emit_progress_event(job.job_id, "job_started")

            # Create progress callback to track task status and broadcast updates
            # Note: callback is called from the batch runner, possibly from multiple threads
            def progress_callback(current: int, total: int, result: VerificationResult | None) -> None:
                """Update job progress and broadcast to WebSocket subscribers (thread-safe)."""
                if result:
                    # Get replicate info from result metadata
                    replicate = result.metadata.replicate if result.metadata else None

                    # Distinguish between starting and completion events by timestamp
                    # Empty timestamp = starting, non-empty = completion
                    if not result.timestamp or result.timestamp == "":
                        # Task is starting - update under lock for thread safety
                        with self._get_job_lock(job.job_id):
                            job.task_started(result.question_id, replicate=replicate)
                            job.current_question = result.question_id
                            job.percentage = ((current - 1) / total) * 100 if total > 0 else 0

                        # Broadcast task started event
                        self._emit_progress_event(
                            job.job_id,
                            "task_started",
                            {
                                "question_id": result.question_id,
                                "replicate": replicate,
                                "current": current,
                                "total": total,
                            },
                        )
                    else:
                        # Task is finished - update under lock for thread safety
                        success = result.completed_without_errors

                        with self._get_job_lock(job.job_id):
                            # task_finished now calculates duration internally from task_start_times
                            job.task_finished(result.question_id, success, replicate=replicate)
                            job.percentage = (current / total) * 100 if total > 0 else 0

                        # Broadcast task completed event
                        self._emit_progress_event(
                            job.job_id,
                            "task_completed",
                            {
                                "question_id": result.question_id,
                                "replicate": replicate,
                                "current": current,
                                "total": total,
                                "success": success,
                            },
                        )

            # Run verification using batch runner
            # Note: batch runner handles task generation, execution, and auto-save
            # Note: job.job_id is kept for server-side job tracking (WebSocket, API endpoints)
            # but is not passed to the verification batch runner anymore
            results = run_verification_batch(
                templates=templates,
                config=job.config,
                run_name=job.run_name,
                global_rubric=global_rubric,
                async_enabled=None,  # Let batch_runner read from env
                max_workers=None,  # Let batch_runner read from env
                storage_url=job.storage_url,
                benchmark_name=job.benchmark_name,
                progress_callback=progress_callback,
            )

            # Finalize - update all fields atomically under lock
            with self._get_job_lock(job.job_id):
                # Store the VerificationResultSet directly
                job.result_set = results
                job.processed_count = len(results)
                # Count successful and failed results
                job.successful_count = sum(1 for r in results if r.metadata.completed_without_errors)
                job.failed_count = sum(1 for r in results if not r.metadata.completed_without_errors)
                job.end_time = time.time()
                job.percentage = 100.0
                job.last_task_duration = None  # Clear last task duration on completion
                job.in_progress_questions = []

            # Thread-safe status transition: RUNNING -> COMPLETED
            if not self._transition_status(job.job_id, JobStatus.RUNNING, JobStatus.COMPLETED):
                logger.warning(f"Job {job.job_id} could not transition to COMPLETED")

            # Store result_set in historical results (outside lock, safe since we just set it)
            if job.result_set:
                self.historical_results[job.job_id] = job.result_set

            self._emit_progress_event(job.job_id, "job_completed")

        except Exception as e:
            logger.error(f"Verification failed: {e}", exc_info=True)
            # Update error fields atomically under lock
            with self._get_job_lock(job.job_id):
                job.error_message = str(e)
                job.end_time = time.time()

            # Thread-safe status transition: RUNNING -> FAILED
            # Use _force_status since we might be in an unexpected state after exception
            self._force_status(job.job_id, JobStatus.FAILED)
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
                        rep = None
                    else:
                        # For multiple replicates, include replicate number in ID
                        combination_id = (
                            f"{template.question_id}_{answering_model.id}_{parsing_model.id}_rep{replicate}"
                        )
                        rep = replicate

                    answering_identity = ModelIdentity.from_model_config(answering_model, role="answering")
                    parsing_identity = ModelIdentity.from_model_config(parsing_model, role="parsing")
                    ts = datetime.now().isoformat()

                    error_result = VerificationResult(
                        metadata=VerificationResultMetadata(
                            question_id=template.question_id,
                            template_id=generate_template_id(template.template_code),
                            completed_without_errors=False,
                            error=f"Verification error: {error!s}",
                            question_text=template.question_text,
                            keywords=template.keywords,
                            answering=answering_identity,
                            parsing=parsing_identity,
                            answering_system_prompt=answering_model.system_prompt,
                            parsing_system_prompt=parsing_model.system_prompt,
                            execution_time=0.0,
                            timestamp=ts,
                            result_id=VerificationResultMetadata.compute_result_id(
                                question_id=template.question_id,
                                answering=answering_identity,
                                parsing=parsing_identity,
                                timestamp=ts,
                                replicate=rep,
                            ),
                            run_name=job.run_name,
                            replicate=rep,
                        ),
                        template=VerificationResultTemplate(
                            raw_llm_response="",
                        ),
                    )
                    error_results[combination_id] = error_result

        return error_results

    # Auto-save is now handled by batch_runner.auto_save_results()

    def get_progress(self, job_id: str) -> dict[str, Any] | None:
        """Get detailed progress information for a job.

        Thread-safe. Returns comprehensive progress data including percentage,
        duration, current question, and result_set if completed.

        Args:
            job_id: The job identifier returned by start_verification().

        Returns:
            Dict with progress fields:
                - job_id, run_name, status, percentage
                - total_questions, processed_count, successful_count, failed_count
                - current_question, in_progress_questions
                - start_time (unix timestamp), duration_seconds
                - last_task_duration, error (if any)
                - result_set (if completed)
            Returns None if job doesn't exist.
        """
        job = self.jobs.get(job_id)
        if not job:
            return None

        # Read all job fields atomically under lock
        with self._get_job_lock(job_id):
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
                "in_progress_questions": list(job.in_progress_questions),  # Copy list
            }

            # Include result_set if completed
            if job.status == "completed" and job.result_set:
                # Pass result_set directly - it's a VerificationResultSet that serializes properly
                progress_data["result_set"] = job.result_set

        return progress_data

    def _emit_progress_event(self, job_id: str, event_type: str, extra_data: dict[str, Any] | None = None) -> None:
        """Emit a progress event to WebSocket subscribers (thread-safe)."""
        job = self.jobs.get(job_id)
        if not job:
            return

        # Read all job fields atomically under lock
        with self._get_job_lock(job_id):
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
                "in_progress_questions": list(job.in_progress_questions),  # Copy list
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
