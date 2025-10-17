"""Service for managing verification jobs with progress tracking."""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from karenina.benchmark.models import FinishedTemplate, VerificationConfig, VerificationJob, VerificationResult
from karenina.benchmark.verification.orchestrator import run_question_verification
from karenina.schemas.rubric_class import Rubric
from karenina.utils.async_utils import AsyncConfig

# Configure logging
logger = logging.getLogger(__name__)


class VerificationService:
    """Service for managing verification jobs."""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, VerificationJob] = {}
        self.futures: dict[str, Any] = {}
        # Store all historical results keyed by job_id
        self.historical_results: dict[str, dict[str, VerificationResult]] = {}

    def start_verification(
        self,
        finished_templates: list[FinishedTemplate],
        config: VerificationConfig,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        async_config: AsyncConfig | None = None,
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
            async_config=async_config,
            storage_url=storage_url,  # Store for auto-save
            benchmark_name=benchmark_name,  # Store benchmark name for auto-save
        )

        self.jobs[job_id] = job

        # Submit to thread pool
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

    def _process_single_template(
        self, job: VerificationJob, template: FinishedTemplate, global_rubric: Rubric | None
    ) -> dict[str, VerificationResult]:
        """Process a single template and return its results."""
        if job.status == "cancelled":
            return {}

        # Update current question info
        job.current_question = (
            template.question_text[:50] + "..." if len(template.question_text) > 50 else template.question_text
        )

        try:
            # Prepare rubric for this question (merge global + question-specific)
            merged_rubric = None
            if getattr(job.config, "rubric_enabled", False):
                question_rubric = None
                if template.question_rubric:
                    # Convert dict to Rubric object
                    try:
                        question_rubric = Rubric.model_validate(template.question_rubric)
                    except Exception as e:
                        logger.warning(f"Failed to parse question rubric for {template.question_id}: {e}")

                try:
                    from karenina.schemas.rubric_class import merge_rubrics

                    merged_rubric = merge_rubrics(global_rubric, question_rubric)
                except ValueError as e:
                    logger.error(f"Error merging rubrics for question {template.question_id}: {e}")
                    # Fall back to global rubric only
                    merged_rubric = global_rubric

            # Resolve few-shot examples using FewShotConfig
            few_shot_examples = None
            few_shot_config = job.config.get_few_shot_config()

            if few_shot_config is not None and few_shot_config.enabled:
                few_shot_examples = few_shot_config.resolve_examples_for_question(
                    question_id=template.question_id,
                    available_examples=template.few_shot_examples,
                    question_text=template.question_text,
                )

            # Run verification for this question (returns dict of results for all model combinations)
            question_results = run_question_verification(
                question_id=template.question_id,
                question_text=template.question_text,
                template_code=template.template_code,
                config=job.config,
                rubric=merged_rubric,
                async_config=job.async_config,
                keywords=template.keywords,
                few_shot_examples=few_shot_examples,
            )

            # Add run metadata to each result
            for _combination_id, result in question_results.items():
                result.run_name = job.run_name
                result.job_id = job.job_id

            return question_results  # type: ignore[no-any-return]

        except Exception as e:
            # Create error results for all model combinations
            return self._create_error_results_for_template(job, template, e)

    def _run_verification(self, job: VerificationJob, templates: list[FinishedTemplate]) -> None:
        """Run verification for all templates in the job."""
        try:
            job.status = "running"
            job.start_time = time.time()

            # Load global rubric if rubric evaluation is enabled
            global_rubric = None
            if getattr(job.config, "rubric_enabled", False):
                global_rubric = self._load_current_rubric()

            # Check if we should use async processing for multiple templates
            if job.async_config and job.async_config.enabled and len(templates) > 1:
                # Process templates in parallel
                self._run_verification_async(job, templates, global_rubric)
            else:
                # Process templates sequentially (original behavior)
                self._run_verification_sync(job, templates, global_rubric)

        except Exception as e:
            logger.error(f"Verification job failed: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = time.time()

    def _run_verification_sync(
        self, job: VerificationJob, templates: list[FinishedTemplate], global_rubric: Rubric | None
    ) -> None:
        """Run verification synchronously (original behavior)."""
        for template in templates:
            if job.status == "cancelled":
                return

            # Process this template and get its results
            question_results = self._process_single_template(job, template, global_rubric)

            # Process each model combination result
            for combination_id, result in question_results.items():
                job.results[combination_id] = result
                job.processed_count += 1

                if result.completed_without_errors:
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

        # Job completed successfully
        job.status = "completed"
        job.end_time = time.time()
        job.percentage = 100.0

        # Store results in historical collection
        self.historical_results[job.job_id] = job.results.copy()

        # Auto-save to database if configured
        self._auto_save_results(job, templates)

    def _run_verification_async(
        self, job: VerificationJob, templates: list[FinishedTemplate], global_rubric: Rubric | None
    ) -> None:
        """Run verification asynchronously for multiple templates in parallel."""
        import asyncio

        from karenina.utils.async_utils import execute_with_config

        def process_template_wrapper(template: FinishedTemplate) -> dict[str, VerificationResult]:
            """Wrapper function for async processing."""
            return self._process_single_template(job, template, global_rubric)

        def update_progress(percentage: float, message: str) -> None:
            """Progress callback for async processing."""
            job.percentage = percentage
            # Extract processed count from message if available
            if "Processed" in message and "/" in message:
                try:
                    processed_str = message.split("Processed ")[1].split("/")[0]
                    processed_count = int(processed_str)
                    job.processed_count = processed_count
                except (ValueError, IndexError):
                    pass

        try:
            # Run templates in parallel using async utilities
            template_results = asyncio.run(
                execute_with_config(
                    items=templates,
                    sync_function=process_template_wrapper,
                    config=job.async_config,
                    progress_callback=update_progress,
                )
            )

            # Process all results
            for _i, question_results in enumerate(template_results):
                if isinstance(question_results, Exception):
                    # Handle async execution error
                    logger.error(f"Template processing failed: {question_results}")
                    continue

                # Process each model combination result for this template
                for combination_id, result in question_results.items():
                    job.results[combination_id] = result

                    if result.completed_without_errors:
                        job.successful_count += 1
                    else:
                        job.failed_count += 1

            # Ensure processed count is accurate
            job.processed_count = sum(
                len(results) if not isinstance(results, Exception) else 0 for results in template_results
            )

            # Job completed successfully
            job.status = "completed"
            job.end_time = time.time()
            job.percentage = 100.0

            # Store results in historical collection
            self.historical_results[job.job_id] = job.results.copy()

            # Auto-save to database if configured
            self._auto_save_results(job, templates)

        except Exception as e:
            logger.error(f"Async verification failed: {e}")
            # Fall back to sync processing
            logger.info("Falling back to synchronous processing")
            self._run_verification_sync(job, templates, global_rubric)

    def _create_error_results_for_template(
        self, job: VerificationJob, template: FinishedTemplate, error: Exception
    ) -> dict[str, VerificationResult]:
        """Create error results for all model combinations of a template."""
        from datetime import datetime

        error_results = {}

        # Handle legacy config vs new multi-model config
        if hasattr(job.config, "answering_models") and job.config.answering_models:
            # Multi-model config - create error for each combination including replicates
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
                            question_id=template.question_id,
                            success=False,
                            error=f"Verification error: {error!s}",
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
                        error_results[combination_id] = error_result
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
                    error=f"Verification error: {error!s}",
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
                error_results[combination_id] = error_result

        return error_results

    def _auto_save_results(self, job: VerificationJob, templates: list[FinishedTemplate]) -> None:
        """Auto-save verification results to database if configured."""
        import os

        # Check AUTOSAVE_DATABASE environment variable (default: True)
        autosave_enabled = os.getenv("AUTOSAVE_DATABASE", "true").lower() in ("true", "1", "yes")

        if not autosave_enabled:
            logger.info("Auto-save to database is disabled (AUTOSAVE_DATABASE=false)")
            return

        if not job.storage_url:
            logger.debug("No storage URL provided, skipping database auto-save")
            return

        if not job.benchmark_name:
            logger.warning("No benchmark name provided, cannot auto-save results")
            return

        try:
            from karenina.benchmark import Benchmark
            from karenina.storage import DBConfig, get_benchmark_summary, save_benchmark, save_verification_results

            # Create database config
            db_config = DBConfig(storage_url=job.storage_url)

            # Check if benchmark already exists
            existing_benchmarks = get_benchmark_summary(db_config, benchmark_name=job.benchmark_name)

            if not existing_benchmarks:
                # Benchmark doesn't exist, create it
                logger.info(f"Creating new benchmark '{job.benchmark_name}' in database")
                benchmark = Benchmark.create(
                    name=job.benchmark_name,
                    description=f"Auto-created for verification run: {job.run_name}",
                    version="1.0.0",
                )

                # Add questions from templates
                for template in templates:
                    # Add question using text format to ensure question_id is preserved
                    benchmark.add_question(
                        question=template.question_text,
                        raw_answer="[Placeholder - see template]",
                        answer_template=template.template_code,
                        question_id=template.question_id,  # Explicitly set question_id to match template
                    )

                # Save benchmark to database
                save_benchmark(benchmark, db_config)

            # Save verification results (job.results is already in the correct format)
            save_verification_results(
                results=job.results,
                db_config=db_config,
                run_id=job.job_id,
                benchmark_name=job.benchmark_name,
                run_name=job.run_name,
                config=job.config.model_dump(),
            )

            logger.info(
                f"Auto-saved verification results to database: {job.storage_url} (benchmark: {job.benchmark_name})"
            )

        except Exception as e:
            # Don't fail the verification job if auto-save fails
            logger.error(f"Failed to auto-save results to database: {e}")

    def get_progress(self, job_id: str) -> dict[str, Any] | None:
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
