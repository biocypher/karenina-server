"""Service for managing answer template generation with progress tracking."""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

# Type alias for config - using TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any, TypeAlias

from karenina.answers.generator import generate_answer_template
from karenina.utils.async_utils import AsyncConfig, execute_with_config
from karenina.utils.code_parser import extract_and_combine_codeblocks

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
        custom_system_prompt: str | None = None,
        force_regenerate: bool = False,
        async_config: AsyncConfig | None = None,
    ):
        self.job_id = job_id
        self.questions_data = questions_data
        self.config = config
        self.total_questions = total_questions
        self.custom_system_prompt = custom_system_prompt
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


class GenerationService:
    """Service for managing template generation jobs."""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, TemplateGenerationJob] = {}
        self.futures: dict[str, Any] = {}  # Add missing futures dict

    def start_generation(
        self,
        questions_data: dict[str, Any],
        config: dict[str, Any],
        custom_system_prompt: str | None = None,
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
            custom_system_prompt=custom_system_prompt,
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
        custom_system_prompt = task["custom_system_prompt"]

        try:
            # Generate template using the generator
            raw_template_response = generate_answer_template(
                question=question_data.get("question", ""),
                raw_answer=question_data.get("raw_answer", ""),
                model=model_name,
                model_provider=model_provider,
                temperature=temperature,
                custom_system_prompt=custom_system_prompt,
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
                    "custom_system_prompt": job.custom_system_prompt,
                }
                generation_tasks.append(task)

            # Set current job for progress callbacks
            self._current_job = job

            if job.async_config.enabled:
                # Async execution: chunk and parallelize
                def progress_callback(percentage: float, message: str) -> None:
                    self._update_job_progress(percentage, message)

                try:
                    results = asyncio.run(
                        execute_with_config(
                            items=generation_tasks,
                            sync_function=self._generate_single_template,
                            config=job.async_config,
                            progress_callback=progress_callback,
                        )
                    )
                except Exception as e:
                    job.status = "failed"
                    job.error_message = f"Async execution failed: {str(e)}"
                    job.end_time = time.time()
                    return
            else:
                # Sync execution: simple loop (original behavior)
                results = []
                for _i, task in enumerate(generation_tasks):
                    if job.cancelled:
                        job.status = "cancelled"
                        return

                    question_data = task["question_data"]
                    job.current_question = question_data.get("question", "Unknown question")[:50] + "..."

                    result = self._generate_single_template(task)
                    results.append(result)

                    job.processed_count += 1
                    if result.get("success", False):
                        job.successful_count += 1
                    else:
                        job.failed_count += 1

                    # Update progress
                    job.percentage = (job.processed_count / job.total_questions) * 100

                    # Calculate time estimates
                    if job.start_time is not None:
                        elapsed_time = time.time() - job.start_time
                        if job.processed_count > 0:
                            avg_time_per_question = elapsed_time / job.processed_count
                            remaining_questions = job.total_questions - job.processed_count
                            job.estimated_time_remaining = avg_time_per_question * remaining_questions

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

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = time.time()
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
