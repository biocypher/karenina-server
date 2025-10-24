"""Service for managing answer template generation with progress tracking."""

import asyncio
import contextlib
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Type alias for config - using TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any, TypeAlias

from fastapi import WebSocket
from karenina.answers.generator import generate_answer_template
from karenina.utils.async_utils import AsyncConfig, execute_with_config
from karenina.utils.code_parser import extract_and_combine_codeblocks

if TYPE_CHECKING:
    pass

ConfigType: TypeAlias = dict[str, Any]

# EMA smoothing factor for time-per-item estimation
PROGRESS_EMA_ALPHA = 0.3


class TemplateGenerationJob:
    """Represents a template generation job."""

    def __init__(
        self,
        job_id: str,
        questions_data: dict[str, Any],
        config: ConfigType,
        total_questions: int,
        force_regenerate: bool = False,
        async_config: AsyncConfig | None = None,
    ):
        self.job_id = job_id
        self.questions_data = questions_data
        self.config = config
        self.total_questions = total_questions
        self.force_regenerate = force_regenerate
        self.async_config = async_config or AsyncConfig.from_env()

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
        self.estimated_time_remaining: float | None = None

        # WebSocket streaming progress fields
        self.in_progress_questions: list[str] = []
        self.ema_seconds_per_item: float = 0.0
        self.last_update_ts: float | None = None

        # Results
        self.results: dict[str, Any] = {}
        self.result: dict[str, Any] | None = None
        self.cancelled = False

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API response."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "total_questions": self.total_questions,
            "completed_questions": self.processed_count,
            "current_question_id": self.current_question,
            "estimated_remaining_time": self.estimated_time_remaining,
            "error_message": self.error_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            # WebSocket streaming fields
            "in_progress_questions": self.in_progress_questions,
            "ema_seconds_per_item": self.ema_seconds_per_item,
        }

    def _estimate_remaining_time(self) -> int | None:
        """Estimate remaining time based on current progress."""
        if self.processed_count == 0:
            return None

        if self.start_time is None:
            return None

        elapsed = time.time() - self.start_time
        avg_time_per_question = elapsed / self.processed_count
        remaining_questions = self.total_questions - self.processed_count
        return int(avg_time_per_question * remaining_questions)

    def task_started(self, question_id: str) -> None:
        """Mark a task as started and add to in-progress list."""
        if question_id not in self.in_progress_questions:
            self.in_progress_questions.append(question_id)
        self.last_update_ts = time.time()

    def task_finished(self, question_id: str, success: bool, duration_seconds: float) -> None:
        """Mark a task as finished, update counts and EMA."""
        # Remove from in-progress list
        if question_id in self.in_progress_questions:
            self.in_progress_questions.remove(question_id)

        # Update counts
        self.processed_count += 1
        if success:
            self.successful_count += 1
        else:
            self.failed_count += 1

        # Update EMA for time estimation
        if self.ema_seconds_per_item == 0.0:
            # First item: initialize EMA with actual duration
            self.ema_seconds_per_item = duration_seconds
        else:
            # Update EMA: new_ema = alpha * current + (1 - alpha) * previous
            self.ema_seconds_per_item = (
                PROGRESS_EMA_ALPHA * duration_seconds + (1 - PROGRESS_EMA_ALPHA) * self.ema_seconds_per_item
            )

        # Calculate estimated remaining time using EMA
        remaining_questions = self.total_questions - self.processed_count
        self.estimated_time_remaining = self.ema_seconds_per_item * remaining_questions

        # Update percentage
        self.percentage = (self.processed_count / self.total_questions) * 100 if self.total_questions > 0 else 0.0

        # Update timestamp
        self.last_update_ts = time.time()


class ProgressBroadcaster:
    """Manages WebSocket connections for broadcasting progress updates."""

    def __init__(self) -> None:
        self.subscribers: dict[str, list[WebSocket]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for thread-safe broadcasting."""
        self._event_loop = loop

    async def subscribe(self, job_id: str, websocket: WebSocket) -> None:
        """Add a WebSocket subscriber for a job."""
        async with self._lock:
            self.subscribers[job_id].append(websocket)

    async def unsubscribe(self, job_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket subscriber."""
        async with self._lock:
            if job_id in self.subscribers:
                with contextlib.suppress(ValueError):
                    self.subscribers[job_id].remove(websocket)
                # Clean up empty lists
                if not self.subscribers[job_id]:
                    del self.subscribers[job_id]

    async def broadcast(self, job_id: str, event_data: dict[str, Any]) -> None:
        """Broadcast a progress event to all subscribers of a job."""
        async with self._lock:
            if job_id not in self.subscribers:
                return

            # Send to all subscribers, removing dead connections
            dead_sockets = []
            for websocket in self.subscribers[job_id]:
                try:
                    await websocket.send_json(event_data)
                except Exception:
                    dead_sockets.append(websocket)

            # Clean up dead connections
            for dead_socket in dead_sockets:
                with contextlib.suppress(ValueError):
                    self.subscribers[job_id].remove(dead_socket)

            # Clean up empty lists
            if not self.subscribers[job_id]:
                del self.subscribers[job_id]

    def broadcast_from_thread(self, job_id: str, event_data: dict[str, Any]) -> None:
        """Thread-safe method to broadcast from worker threads."""
        if self._event_loop and not self._event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.broadcast(job_id, event_data), self._event_loop)

    async def cleanup_job(self, job_id: str) -> None:
        """Close all connections for a job and clean up."""
        async with self._lock:
            if job_id in self.subscribers:
                for websocket in self.subscribers[job_id]:
                    with contextlib.suppress(Exception):
                        await websocket.close()
                del self.subscribers[job_id]


class GenerationService:
    """Service for managing template generation jobs."""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, TemplateGenerationJob] = {}
        self.futures: dict[str, Any] = {}  # Add missing futures dict
        self.broadcaster = ProgressBroadcaster()

    def start_generation(
        self,
        questions_data: dict[str, Any],
        config: dict[str, Any],
        force_regenerate: bool = False,
        async_config: AsyncConfig | None = None,
    ) -> str:
        """Start a new template generation job."""
        job_id = str(uuid.uuid4())

        # Create job
        job = TemplateGenerationJob(
            job_id=job_id,
            questions_data=questions_data,
            config=config,
            total_questions=len(questions_data),
            force_regenerate=force_regenerate,
            async_config=async_config,
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

        event_data = {
            "type": event_type,
            "job_id": job_id,
            "status": job.status,
            "percentage": job.percentage,
            "processed": job.processed_count,
            "total": job.total_questions,
            "in_progress_questions": job.in_progress_questions,
            "ema_seconds_per_item": job.ema_seconds_per_item,
            "estimated_time_remaining": job.estimated_time_remaining,
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

        try:
            # Generate template using the structured generator
            raw_template_response = generate_answer_template(
                question=question_data.get("question", ""),
                raw_answer=question_data.get("raw_answer", ""),
                model=model_name,
                model_provider=model_provider,
                temperature=temperature,
                interface=interface,
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

    def _update_job_progress(self, percentage: float, message: str) -> None:
        """Update job progress from async callback."""
        # This will be used by the async wrapper to update progress
        # The job object will be set before calling async execution
        if hasattr(self, "_current_job") and self._current_job:
            job = self._current_job
            # Extract question info from message if available
            if ":" in message:
                _, question_info = message.split(":", 1)
                job.current_question = question_info.strip()[:50] + "..."

            job.percentage = percentage

            # Calculate time estimates
            if job.start_time is not None:
                elapsed_time = time.time() - job.start_time
                if job.processed_count > 0:
                    avg_time_per_question = elapsed_time / job.processed_count
                    remaining_questions = job.total_questions - job.processed_count
                    job.estimated_time_remaining = avg_time_per_question * remaining_questions

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
            else:
                # Old config format (dict)
                config_dict = config
                model_name = config_dict.get("model_name", config_dict.get("model", "gemini-2.0-flash"))
                interface = config_dict.get("interface", "langchain")
                # Only set default provider for langchain interface
                if interface == "langchain":
                    model_provider = config_dict.get("model_provider", "google_genai")
                else:
                    model_provider = config_dict.get("model_provider", "")
                temperature = config_dict.get("temperature", 0.1)

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
                }
                generation_tasks.append(task)

            # Set current job for progress callbacks
            self._current_job = job

            if job.async_config.enabled:
                # Async execution: chunk and parallelize
                def progress_callback(percentage: float, message: str) -> None:
                    self._update_job_progress(percentage, message)

                # Track when tasks start and complete
                task_start_times: dict[str, float] = {}

                def on_task_start(task: dict[str, Any]) -> None:
                    question_id = task["question_id"]
                    job.task_started(question_id)
                    task_start_times[question_id] = time.time()
                    self._emit_progress_event(job.job_id, "task_started", {"question_id": question_id})

                def on_task_done(task: dict[str, Any], result: dict[str, Any] | Exception) -> None:
                    question_id = task["question_id"]
                    task_duration = time.time() - task_start_times.get(question_id, time.time())
                    success = not isinstance(result, Exception) and result.get("success", False)
                    job.task_finished(question_id, success, task_duration)
                    self._emit_progress_event(
                        job.job_id, "task_completed", {"question_id": question_id, "success": success}
                    )

                try:
                    results = asyncio.run(
                        execute_with_config(
                            items=generation_tasks,
                            sync_function=self._generate_single_template,
                            config=job.async_config,
                            progress_callback=progress_callback,
                            on_task_start=on_task_start,
                            on_task_done=on_task_done,
                        )
                    )
                except Exception as e:
                    job.status = "failed"
                    job.error_message = f"Async execution failed: {str(e)}"
                    job.end_time = time.time()
                    self._emit_progress_event(job.job_id, "job_failed", {"error": str(e)})
                    return
            else:
                # Sync execution: simple loop (original behavior)
                results = []
                for _i, task in enumerate(generation_tasks):
                    if job.cancelled:
                        job.status = "cancelled"
                        self._emit_progress_event(job.job_id, "job_cancelled")
                        return

                    question_id = task["question_id"]
                    question_data = task["question_data"]
                    job.current_question = question_data.get("question", "Unknown question")[:50] + "..."

                    # Track task start
                    job.task_started(question_id)
                    self._emit_progress_event(job.job_id, "task_started", {"question_id": question_id})
                    task_start_time = time.time()

                    # Generate template
                    result = self._generate_single_template(task)
                    results.append(result)

                    # Track task completion
                    task_duration = time.time() - task_start_time
                    job.task_finished(question_id, result.get("success", False), task_duration)
                    self._emit_progress_event(
                        job.job_id,
                        "task_completed",
                        {"question_id": question_id, "success": result.get("success", False)},
                    )

            # Process results and update job
            for result in results:
                if isinstance(result, Exception):
                    # Handle exceptions from async execution
                    question_id = "unknown"
                    job.results[question_id] = {"success": False, "error": str(result), "template_code": ""}
                    job.failed_count += 1
                else:
                    question_id = result["question_id"]
                    job.results[question_id] = result
                    if not job.async_config.enabled:
                        # Counts already updated in sync mode
                        continue

                    job.processed_count += 1
                    if result.get("success", False):
                        job.successful_count += 1
                    else:
                        job.failed_count += 1

            # Job completed successfully
            job.status = "completed"
            job.end_time = time.time()
            job.percentage = 100.0

            # Create final result
            total_time = (job.end_time or 0) - (job.start_time or 0)
            job.result = {
                "templates": job.results,
                "total_templates": job.total_questions,
                "successful_generations": job.successful_count,
                "failed_generations": job.failed_count,
                "average_generation_time": total_time / job.total_questions if job.total_questions > 0 else 0,
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
        finally:
            # Clean up reference to current job
            if hasattr(self, "_current_job"):
                delattr(self, "_current_job")

    def get_progress(self, job_id: str) -> dict[str, Any] | None:
        """Get progress information for a job."""
        job = self.jobs.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "status": job.status,
            "total_questions": job.total_questions,
            "processed_count": job.processed_count,
            "successful_count": job.successful_count,
            "failed_count": job.failed_count,
            "percentage": job.percentage,
            "current_question": job.current_question,
            "estimated_time_remaining": job.estimated_time_remaining,
            "error_message": job.error_message,
            "start_time": job.start_time,
            "end_time": job.end_time,
            # WebSocket streaming fields
            "in_progress_questions": job.in_progress_questions,
            "ema_seconds_per_item": job.ema_seconds_per_item,
        }

    def generate_rubric_traits(
        self,
        system_prompt: str,
        user_prompt: str,
        model_provider: str = "google_genai",
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        interface: str = "langchain",
    ) -> str:
        """
        Generate rubric traits using LLM.

        This is a simple synchronous method for trait generation.
        For now, we don't use the job queue system as trait generation is typically fast.
        """
        # Generate response using appropriate interface
        if interface == "openrouter":
            # Use init_chat_model_unified for OpenRouter
            from karenina.llm.interface import call_model, init_chat_model_unified

            chat_model = init_chat_model_unified(
                provider="",  # Empty provider for OpenRouter
                model=model_name,
                temperature=temperature,
                interface=interface,
            )
            # Create messages for chat model
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response = chat_model.invoke(messages)
            return str(response.content)
        else:
            # Use call_model for langchain interface
            from karenina.llm.interface import call_model

            response = call_model(
                model=model_name,
                provider=model_provider,
                message=user_prompt,
                system_message=system_prompt,
                temperature=temperature,
            )
            return str(response.message)


# Global service instance
generation_service = GenerationService()
