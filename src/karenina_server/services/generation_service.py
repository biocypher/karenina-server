"""Service for managing answer template generation with progress tracking."""

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, TypeAlias

from karenina.domain.answers.generator import generate_answer_template
from karenina.utils.code import extract_and_combine_codeblocks

from .progress_broadcaster import ProgressBroadcaster

# Configure logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

ConfigType: TypeAlias = dict[str, Any]


class TemplateGenerationJob:
    """Represents a template generation job."""

    def __init__(
        self,
        job_id: str,
        questions_data: dict[str, Any],
        config: ConfigType,
        total_questions: int,
        force_regenerate: bool = False,
        async_enabled: bool | None = None,
        max_workers: int | None = None,
    ):
        self.job_id = job_id
        self.questions_data = questions_data
        self.config = config
        self.total_questions = total_questions
        self.force_regenerate = force_regenerate

        # Read async settings from parameters or environment
        if async_enabled is None:
            self.async_enabled = os.getenv("KARENINA_ASYNC_ENABLED", "true").lower() == "true"
        else:
            self.async_enabled = async_enabled

        if max_workers is None:
            max_workers_env = os.getenv("KARENINA_ASYNC_MAX_WORKERS")
            self.max_workers = int(max_workers_env) if max_workers_env else 2
        else:
            self.max_workers = max_workers

        # Status tracking
        self.status = "pending"  # pending, running, completed, failed, cancelled
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.error_message: str | None = None

        # Progress tracking
        self.processed_count = 0
        self.successful_count = 0
        self.failed_count = 0
        self.percentage = 0.0
        self.current_question = ""
        self.last_task_duration: float | None = None

        # WebSocket streaming progress fields
        self.in_progress_questions: list[str] = []

        # Task timing tracking (maps question_id to start time)
        self.task_start_times: dict[str, float] = {}

        # Results
        self.results: dict[str, Any] = {}
        self.result: dict[str, Any] | None = None
        self.cancelled = False

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API response."""
        # Calculate duration if job has started
        duration = None
        if self.start_time:
            duration = self.end_time - self.start_time if self.end_time else time.time() - self.start_time

        return {
            "job_id": self.job_id,
            "status": self.status,
            "total_questions": self.total_questions,
            "completed_questions": self.processed_count,
            "current_question_id": self.current_question,
            "duration_seconds": duration,
            "last_task_duration": self.last_task_duration,
            "error_message": self.error_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "in_progress_questions": self.in_progress_questions,
        }

    def task_started(self, question_id: str) -> None:
        """Mark a task as started and record start time."""
        if question_id not in self.in_progress_questions:
            self.in_progress_questions.append(question_id)

        # Record task start time
        self.task_start_times[question_id] = time.time()

    def task_finished(self, question_id: str, success: bool) -> None:
        """Mark a task as finished, calculate duration, and update counts."""
        # Calculate task duration from recorded start time
        task_duration = 0.0
        if question_id in self.task_start_times:
            task_duration = time.time() - self.task_start_times[question_id]
            # Clean up start time
            del self.task_start_times[question_id]

        # Remove from in-progress list
        if question_id in self.in_progress_questions:
            self.in_progress_questions.remove(question_id)

        # Update counts
        self.processed_count += 1
        if success:
            self.successful_count += 1
        else:
            self.failed_count += 1

        # Update percentage
        self.percentage = (self.processed_count / self.total_questions) * 100 if self.total_questions > 0 else 0.0

        # Track last task duration
        self.last_task_duration = task_duration


class GenerationService:
    """Service for managing template generation jobs.

    Lock Hierarchy (conc-005):
        1. _shutdown_lock - Service-level shutdown coordination

    Always acquire locks in this order to prevent deadlocks.
    """

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="generate-")
        self.jobs: dict[str, TemplateGenerationJob] = {}
        self.futures: dict[str, Any] = {}  # Add missing futures dict
        self.broadcaster = ProgressBroadcaster()
        # Shutdown coordination (conc-001)
        self._shutdown_event = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False

    def shutdown(self, wait: bool = True, cancel_pending: bool = False) -> None:
        """Gracefully shut down the generation service (conc-001).

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
                logger.debug("GenerationService.shutdown() called but already shut down")
                return

            logger.info("🛑 GenerationService shutting down...")
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
                            job.status = "cancelled"
                            job.cancelled = True
                if cancelled_count > 0:
                    logger.info(f"   Cancelled {cancelled_count} pending jobs")

            # Count in-flight jobs
            running_jobs = sum(1 for j in self.jobs.values() if j.status == "running")
            if running_jobs > 0:
                logger.info(f"   Waiting for {running_jobs} running job(s) to complete...")

        # Shutdown executor (outside lock to avoid blocking other operations)
        self.executor.shutdown(wait=wait, cancel_futures=cancel_pending)
        logger.info("✓ GenerationService shutdown complete")

    def is_shutdown(self) -> bool:
        """Check if the service has been shut down."""
        return self._is_shutdown

    def start_generation(
        self,
        questions_data: dict[str, Any],
        config: dict[str, Any],
        force_regenerate: bool = False,
        async_enabled: bool | None = None,
        max_workers: int | None = None,
    ) -> str:
        """Start a new template generation job."""
        # Check if service is shutting down (conc-001)
        if self._is_shutdown:
            raise RuntimeError("GenerationService is shutting down, cannot accept new jobs")

        job_id = str(uuid.uuid4())

        # Create job
        job = TemplateGenerationJob(
            job_id=job_id,
            questions_data=questions_data,
            config=config,
            total_questions=len(questions_data),
            force_regenerate=force_regenerate,
            async_enabled=async_enabled,
            max_workers=max_workers,
        )

        self.jobs[job_id] = job

        # Submit to thread pool
        future = self.executor.submit(self._generate_templates, job)
        self.futures[job_id] = future

        return job_id

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get the status of a generation job."""
        job = self.jobs.get(job_id)
        return job.to_dict() if job else None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a generation job."""
        job = self.jobs.get(job_id)
        if job and job.status in ["pending", "running"]:
            job.cancelled = True
            job.status = "cancelled"
            return True
        return False

    def get_job_results(self, job_id: str) -> dict[str, Any] | None:
        """Get the results of a completed job."""
        job = self.jobs.get(job_id)
        if job and job.status == "completed":
            return job.results
        return None

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

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> None:
        """Clean up old jobs to prevent memory leaks."""
        current_time = time.time()
        jobs_to_remove: list[str] = []

        for job_id, job in self.jobs.items():
            if job.start_time is None:
                continue
            age_hours = (current_time - job.start_time) / 3600
            if age_hours > max_age_hours:
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.jobs[job_id]

    def _generate_single_template(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Generate a single template from a task dictionary.

        Args:
            task: Dictionary containing all parameters needed for generation

        Returns:
            Dictionary containing generation result
        """
        question_id = task["question_id"]
        question_data = task["question_data"]
        model_name = task["model_name"]
        model_provider = task["model_provider"]
        temperature = task["temperature"]
        interface = task["interface"]
        endpoint_base_url = task.get("endpoint_base_url")
        endpoint_api_key = task.get("endpoint_api_key")

        try:
            # Generate template using the structured generator
            raw_template_response = generate_answer_template(
                question=question_data.get("question", ""),
                raw_answer=question_data.get("raw_answer", ""),
                model=model_name,
                model_provider=model_provider,
                temperature=temperature,
                interface=interface,
                endpoint_base_url=endpoint_base_url,
                endpoint_api_key=endpoint_api_key,
            )

            # Extract only the Python code blocks from the LLM response
            template_code = extract_and_combine_codeblocks(raw_template_response)

            # Check if we got any code blocks
            if not template_code.strip():
                # If no code blocks found, use the raw response as fallback but mark as potential issue
                template_code = raw_template_response
                template_result = {
                    "success": True,
                    "template_code": template_code,
                    "error": "Warning: No code blocks found in response",
                    "raw_response": raw_template_response,
                    "question_id": question_id,
                }
            else:
                template_result = {
                    "success": True,
                    "template_code": template_code,
                    "error": None,
                    "raw_response": raw_template_response,
                    "question_id": question_id,
                }

            return template_result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "template_code": "",
                "question_id": question_id,
                "raw_response": "",
            }

    def _execute_sequential(self, tasks: list[dict[str, Any]], job: TemplateGenerationJob) -> list[dict[str, Any]]:
        """Execute tasks sequentially with progress tracking.

        Args:
            tasks: List of task dictionaries to process
            job: The job object for tracking progress

        Returns:
            List of results in the same order as input tasks
        """
        results = []
        for task in tasks:
            if job.cancelled:
                job.status = "cancelled"
                self._emit_progress_event(job.job_id, "job_cancelled")
                break

            question_id = task["question_id"]
            question_data = task["question_data"]
            job.current_question = question_data.get("question", "Unknown question")[:50] + "..."

            # Track task start
            job.task_started(question_id)
            self._emit_progress_event(job.job_id, "task_started", {"question_id": question_id})

            # Generate template
            result = self._generate_single_template(task)
            results.append(result)

            # Track task completion
            success = result.get("success", False)
            job.task_finished(question_id, success)
            self._emit_progress_event(
                job.job_id,
                "task_completed",
                {"question_id": question_id, "success": success},
            )

        return results

    def _execute_parallel(self, tasks: list[dict[str, Any]], job: TemplateGenerationJob) -> list[dict[str, Any]]:
        """Execute tasks in parallel using ThreadPoolExecutor.

        Args:
            tasks: List of task dictionaries to process
            job: The job object for tracking progress

        Returns:
            List of results in the same order as input tasks
        """
        # Initialize results list to maintain order
        results: list[dict[str, Any] | None] = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=job.max_workers) as executor:
            # Submit all tasks and track with original index
            future_to_index = {}
            for idx, task in enumerate(tasks):
                question_id = task["question_id"]

                # Track task start
                job.task_started(question_id)
                self._emit_progress_event(job.job_id, "task_started", {"question_id": question_id})

                # Submit task
                future = executor.submit(self._generate_single_template, task)
                future_to_index[future] = idx

            # Collect results as they complete
            for future in as_completed(future_to_index):
                if job.cancelled:
                    job.status = "cancelled"
                    self._emit_progress_event(job.job_id, "job_cancelled")
                    # Cancel remaining futures
                    for f in future_to_index:
                        f.cancel()
                    break

                idx = future_to_index[future]
                task = tasks[idx]
                question_id = task["question_id"]

                try:
                    result = future.result()
                    results[idx] = result
                    success = result.get("success", False)

                    # Track task completion
                    job.task_finished(question_id, success)
                    self._emit_progress_event(
                        job.job_id,
                        "task_completed",
                        {"question_id": question_id, "success": success},
                    )

                except Exception as e:
                    # Store exception as result
                    results[idx] = {
                        "success": False,
                        "error": str(e),
                        "template_code": "",
                        "question_id": question_id,
                        "raw_response": "",
                    }
                    job.task_finished(question_id, False)
                    self._emit_progress_event(
                        job.job_id,
                        "task_completed",
                        {"question_id": question_id, "success": False},
                    )

        return [r for r in results if r is not None]

    def _generate_templates(self, job: TemplateGenerationJob) -> None:
        """Generate templates for all questions in the job."""
        try:
            job.status = "running"
            job.start_time = time.time()
            self._emit_progress_event(job.job_id, "job_started")

            # Extract config values
            config = job.config
            if hasattr(config, "model_name"):
                # New config format (Pydantic model)
                model_name = config.model_name
                model_provider = config.model_provider  # type: ignore[attr-defined]
                temperature = config.temperature  # type: ignore[attr-defined]
                interface = getattr(config, "interface", "langchain")
                endpoint_base_url = getattr(config, "endpoint_base_url", None)
                endpoint_api_key = getattr(config, "endpoint_api_key", None)
                # Extract secret value if it's a SecretStr
                if endpoint_api_key is not None and hasattr(endpoint_api_key, "get_secret_value"):
                    endpoint_api_key = endpoint_api_key.get_secret_value()
            else:
                # Old config format (dict)
                config_dict = config
                model_name = config_dict.get("model_name", config_dict.get("model", "claude-haiku-4-5"))
                interface = config_dict.get("interface", "langchain")
                # Only set default provider for langchain interface
                if interface == "langchain":
                    model_provider = config_dict.get("model_provider", "anthropic")
                else:
                    model_provider = config_dict.get("model_provider", "")
                temperature = config_dict.get("temperature", 0.1)
                endpoint_base_url = config_dict.get("endpoint_base_url", None)
                endpoint_api_key = config_dict.get("endpoint_api_key", None)

            # Build list of all generation tasks
            generation_tasks = []
            for question_id, question_data in job.questions_data.items():
                task = {
                    "question_id": question_id,
                    "question_data": question_data,
                    "model_name": model_name,
                    "model_provider": model_provider,
                    "temperature": temperature,
                    "interface": interface,
                    "endpoint_base_url": endpoint_base_url,
                    "endpoint_api_key": endpoint_api_key,
                }
                generation_tasks.append(task)

            # Execute tasks based on async setting
            if job.async_enabled:
                # Parallel execution using ThreadPoolExecutor
                results = self._execute_parallel(generation_tasks, job)
            else:
                # Sequential execution (original behavior)
                results = self._execute_sequential(generation_tasks, job)

            # Store results
            for result in results:
                if isinstance(result, Exception):
                    # Handle exceptions
                    question_id = "unknown"
                    job.results[question_id] = {"success": False, "error": str(result), "template_code": ""}
                else:
                    question_id = result["question_id"]
                    job.results[question_id] = result

            # Job completed successfully
            job.status = "completed"
            job.end_time = time.time()
            job.percentage = 100.0

            # Create final result
            # Calculate average generation time from total duration
            total_time = (job.end_time or 0) - (job.start_time or 0)
            average_time = total_time / job.total_questions if job.total_questions > 0 else 0

            job.result = {
                "templates": job.results,
                "total_templates": job.total_questions,
                "successful_generations": job.successful_count,
                "failed_generations": job.failed_count,
                "average_generation_time": average_time,
                "model_info": {"name": model_name, "provider": model_provider, "temperature": temperature},
            }

            # Emit completion event
            self._emit_progress_event(job.job_id, "job_completed")

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = time.time()
            # Emit failure event
            self._emit_progress_event(job.job_id, "job_failed", {"error": str(e)})

    def get_progress(self, job_id: str) -> dict[str, Any] | None:
        """Get progress information for a job."""
        job = self.jobs.get(job_id)
        if not job:
            return None

        # Calculate duration
        duration = None
        if job.start_time:
            duration = job.end_time - job.start_time if job.end_time else time.time() - job.start_time

        return {
            "job_id": job.job_id,
            "status": job.status,
            "total_questions": job.total_questions,
            "processed_count": job.processed_count,
            "successful_count": job.successful_count,
            "failed_count": job.failed_count,
            "percentage": job.percentage,
            "current_question": job.current_question,
            "duration_seconds": duration,
            "last_task_duration": job.last_task_duration,
            "error_message": job.error_message,
            "start_time": job.start_time,
            "end_time": job.end_time,
            "in_progress_questions": job.in_progress_questions,
        }


# Global service instance
generation_service = GenerationService()
