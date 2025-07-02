"""Service for managing verification jobs with progress tracking."""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from karenina.benchmark.models import FinishedTemplate, VerificationConfig, VerificationJob, VerificationResult
from karenina.benchmark.verifier import run_question_verification
from karenina.schemas.rubric_class import Rubric


class VerificationService:
    """Service for managing verification jobs."""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, VerificationJob] = {}
        self.futures: dict[str, any] = {}
        # Store all historical results keyed by job_id
        self.historical_results: dict[str, dict[str, VerificationResult]] = {}

    def start_verification(
        self,
        finished_templates: list[FinishedTemplate],
        config: VerificationConfig,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> str:
        """Start a new verification job."""
        # Validate rubric availability if rubric evaluation is enabled
        if getattr(config, "rubric_enabled", False):
            current_rubric = self._load_current_rubric()
            if current_rubric is None:
                raise ValueError(
                    "Rubric evaluation is enabled but no rubric is configured. Please create a rubric first."
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
            job_id=job_id, run_name=run_name, status="pending", config=config, total_questions=total_combinations
        )

        self.jobs[job_id] = job

        # Submit to thread pool
        future = self.executor.submit(self._run_verification, job, templates_to_verify)
        self.futures[job_id] = future

        return job_id

    def get_job_status(self, job_id: str) -> dict | None:
        """Get the status of a verification job."""
        job = self.jobs.get(job_id)
        return job.to_dict() if job else None

    def get_job_results(self, job_id: str) -> dict[str, VerificationResult] | None:
        """Get the results of a completed job."""
        job = self.jobs.get(job_id)
        if job and job.status == "completed":
            return job.results
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

    def cleanup_old_jobs(self, max_age_hours: int = 24):
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
        Load the current rubric from the API.

        Returns:
            The current rubric if available, None otherwise
        """
        try:
            # Import here to avoid circular imports
            from ..api.rubric_handlers import current_rubric

            return current_rubric
        except Exception as e:
            print(f"Warning: Failed to load rubric: {e}")
            return None

    def _run_verification(self, job: VerificationJob, templates: list[FinishedTemplate]):
        """Run verification for all templates in the job."""
        try:
            job.status = "running"
            job.start_time = time.time()

            # Load rubric if rubric evaluation is enabled
            rubric = None
            if getattr(job.config, "rubric_enabled", False):
                rubric = self._load_current_rubric()
                if rubric:
                    print(f"Loaded rubric '{rubric.title}' with {len(rubric.traits)} traits for verification")
                else:
                    print("Warning: Rubric evaluation enabled but no rubric found")

            for i, template in enumerate(templates):
                if job.status == "cancelled":
                    return

                # Update current question info
                job.current_question = (
                    template.question_text[:50] + "..." if len(template.question_text) > 50 else template.question_text
                )

                try:
                    # Run verification for this question (returns dict of results for all model combinations)
                    question_results = run_question_verification(
                        question_id=template.question_id,
                        question_text=template.question_text,
                        template_code=template.template_code,
                        config=job.config,
                        rubric=rubric,
                    )

                    # Process each model combination result
                    for combination_id, result in question_results.items():
                        # Add run metadata to result
                        result.run_name = job.run_name
                        result.job_id = job.job_id

                        job.results[combination_id] = result
                        job.processed_count += 1

                        if result.success:
                            job.successful_count += 1
                        else:
                            job.failed_count += 1

                    # Update progress
                    job.percentage = (job.processed_count / job.total_questions) * 100

                    # Calculate time estimates
                    if job.processed_count > 0:
                        elapsed_time = time.time() - job.start_time
                        avg_time_per_question = elapsed_time / job.processed_count
                        remaining_questions = job.total_questions - job.processed_count
                        job.estimated_time_remaining = avg_time_per_question * remaining_questions

                except Exception as e:
                    # Create error results for all model combinations
                    from datetime import datetime

                    # Handle legacy config vs new multi-model config
                    if hasattr(job.config, "answering_models") and job.config.answering_models:
                        # Multi-model config - create error for each combination including replicates
                        for answering_model in job.config.answering_models:
                            for parsing_model in job.config.parsing_models:
                                # Create errors for all replicates
                                for replicate in range(1, job.config.replicate_count + 1):
                                    # For single replicate, don't include replicate numbers
                                    if job.config.replicate_count == 1:
                                        combination_id = (
                                            f"{template.question_id}_{answering_model.id}_{parsing_model.id}"
                                        )
                                        answering_replicate = None
                                        parsing_replicate = None
                                    else:
                                        # For multiple replicates, include replicate number in ID and track separately
                                        combination_id = f"{template.question_id}_{answering_model.id}_{parsing_model.id}_rep{replicate}"
                                        answering_replicate = replicate
                                        parsing_replicate = replicate

                                    # For OpenRouter interface, don't include provider in the model string
                                    if answering_model.interface == "openrouter":
                                        answering_model_str = answering_model.model_name
                                    else:
                                        answering_model_str = (
                                            f"{answering_model.model_provider}/{answering_model.model_name}"
                                        )

                                    if parsing_model.interface == "openrouter":
                                        parsing_model_str = parsing_model.model_name
                                    else:
                                        parsing_model_str = f"{parsing_model.model_provider}/{parsing_model.model_name}"

                                    error_result = VerificationResult(
                                        question_id=template.question_id,
                                        success=False,
                                        error=f"Verification error: {e!s}",
                                        question_text=template.question_text,
                                        raw_llm_response="",
                                        answering_model=answering_model_str,
                                        parsing_model=parsing_model_str,
                                        execution_time=0.0,
                                        timestamp=datetime.now().isoformat(),
                                        answering_system_prompt=answering_model.system_prompt,
                                        parsing_system_prompt=parsing_model.system_prompt,
                                        run_name=job.run_name,
                                        job_id=job.job_id,
                                        answering_replicate=answering_replicate,
                                        parsing_replicate=parsing_replicate,
                                    )
                                    job.results[combination_id] = error_result
                                    job.processed_count += 1
                                    job.failed_count += 1
                    else:
                        # Legacy single model config - handle replicates
                        for replicate in range(1, getattr(job.config, "replicate_count", 1) + 1):
                            # For single replicate, don't include replicate numbers
                            if getattr(job.config, "replicate_count", 1) == 1:
                                combination_id = template.question_id
                                answering_replicate = None
                                parsing_replicate = None
                            else:
                                # For multiple replicates, include replicate number in ID and track separately
                                combination_id = f"{template.question_id}_rep{replicate}"
                                answering_replicate = replicate
                                parsing_replicate = replicate

                            error_result = VerificationResult(
                                question_id=template.question_id,
                                success=False,
                                error=f"Verification error: {e!s}",
                                question_text=template.question_text,
                                raw_llm_response="",
                                answering_model=f"{job.config.answering_model_provider or 'unknown'}/{job.config.answering_model_name or 'unknown'}",
                                parsing_model=f"{job.config.parsing_model_provider or 'unknown'}/{job.config.parsing_model_name or 'unknown'}",
                                execution_time=0.0,
                                timestamp=datetime.now().isoformat(),
                                answering_system_prompt=job.config.answering_system_prompt,
                                parsing_system_prompt=job.config.parsing_system_prompt,
                                run_name=job.run_name,
                                job_id=job.job_id,
                                answering_replicate=answering_replicate,
                                parsing_replicate=parsing_replicate,
                            )
                            job.results[combination_id] = error_result
                            job.processed_count += 1
                            job.failed_count += 1

                    job.percentage = (job.processed_count / job.total_questions) * 100

            # Job completed successfully
            job.status = "completed"
            job.end_time = time.time()
            job.percentage = 100.0

            # Store results in historical collection
            self.historical_results[job.job_id] = job.results.copy()

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = time.time()

    def get_progress(self, job_id: str) -> dict | None:
        """Get progress information for a job."""
        job = self.jobs.get(job_id)
        if not job:
            return None

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
            "estimated_time_remaining": job.estimated_time_remaining,
            "error": job.error_message,
        }

        # Include results if completed
        if job.status == "completed":
            progress_data["results"] = job.results

        return progress_data


# Global service instance
verification_service = VerificationService()
