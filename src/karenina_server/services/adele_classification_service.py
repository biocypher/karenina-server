"""Service for managing ADeLe question classification jobs with progress tracking."""

import logging
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from karenina.integrations.adele import (
    ADELE_TRAIT_NAMES,
    QuestionClassificationResult,
    QuestionClassifier,
    get_adele_trait,
)
from karenina.integrations.adele.schemas import AdeleTraitInfo

from karenina_server.schemas.adele import AdeleModelConfig

from .progress_broadcaster import ProgressBroadcaster

logger = logging.getLogger(__name__)

# Cleanup configuration
_CLEANUP_INTERVAL_SECONDS = 3600  # Run cleanup at most once per hour
_MAX_JOB_AGE_HOURS = 24  # Remove jobs older than 24 hours
_MAX_HISTORICAL_RESULTS = 100  # Keep at most 100 historical result sets


class ClassificationJobStatus(Enum):
    """Enum for classification job status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def valid_transitions(cls) -> dict["ClassificationJobStatus", set["ClassificationJobStatus"]]:
        """Return valid state transitions."""
        return {
            cls.PENDING: {cls.RUNNING, cls.CANCELLED},
            cls.RUNNING: {cls.COMPLETED, cls.FAILED, cls.CANCELLED},
            cls.COMPLETED: set(),
            cls.FAILED: set(),
            cls.CANCELLED: set(),
        }

    def can_transition_to(self, new_status: "ClassificationJobStatus") -> bool:
        """Check if transition to new_status is valid."""
        return new_status in self.valid_transitions().get(self, set())


@dataclass
class ClassificationJob:
    """Represents a batch classification job."""

    job_id: str
    status: str = "pending"
    total_questions: int = 0
    completed_count: int = 0
    current_question_id: str | None = None
    trait_names: list[str] = field(default_factory=list)
    results: dict[str, QuestionClassificationResult] = field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None
    error_message: str | None = None
    percentage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API response."""
        duration = None
        if self.start_time:
            duration = (self.end_time or time.time()) - self.start_time

        return {
            "job_id": self.job_id,
            "status": self.status,
            "total_questions": self.total_questions,
            "completed_count": self.completed_count,
            "current_question_id": self.current_question_id,
            "trait_names": self.trait_names,
            "percentage": self.percentage,
            "start_time": self.start_time,
            "duration_seconds": duration,
            "error": self.error_message,
        }


class AdeleClassificationService:
    """Service for managing ADeLe question classification jobs.

    Provides both synchronous single-question classification and
    asynchronous batch classification with progress tracking.
    """

    def __init__(self, max_workers: int = 2):
        """Initialize the classification service.

        Args:
            max_workers: Maximum concurrent classification jobs.
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="adele-")
        self.jobs: dict[str, ClassificationJob] = {}
        self.futures: dict[str, Future[None]] = {}
        self.historical_results: OrderedDict[str, dict[str, QuestionClassificationResult]] = OrderedDict()
        self.broadcaster = ProgressBroadcaster()

        # Thread safety
        self._job_locks: dict[str, threading.Lock] = {}
        self._master_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False
        self._last_cleanup_time = time.time()

        # Lazy classifier initialization
        self._classifier: QuestionClassifier | None = None

    @property
    def classifier(self) -> QuestionClassifier:
        """Lazily initialize and return the classifier."""
        if self._classifier is None:
            self._classifier = QuestionClassifier()
        return self._classifier

    def _get_job_lock(self, job_id: str) -> threading.Lock:
        """Get or create a lock for a specific job."""
        with self._master_lock:
            if job_id not in self._job_locks:
                self._job_locks[job_id] = threading.Lock()
            return self._job_locks[job_id]

    def _remove_job_lock(self, job_id: str) -> None:
        """Remove a job's lock when the job is cleaned up."""
        with self._master_lock:
            self._job_locks.pop(job_id, None)

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shut down the classification service."""
        with self._shutdown_lock:
            if self._is_shutdown:
                return

            logger.info("🛑 AdeleClassificationService shutting down...")
            self._is_shutdown = True
            self._shutdown_event.set()

        self.executor.shutdown(wait=wait)
        logger.info("✓ AdeleClassificationService shutdown complete")

    def get_available_traits(self) -> list[AdeleTraitInfo]:
        """Get list of available ADeLe traits with their info.

        Returns:
            List of AdeleTraitInfo objects describing each available trait.
        """
        traits_info = []
        for name in ADELE_TRAIT_NAMES:
            trait = get_adele_trait(name)
            info = AdeleTraitInfo(
                name=trait.name,
                code=trait.description.split("(")[-1].rstrip(")")
                if trait.description and "(" in trait.description
                else "",
                description=trait.description,
                classes=trait.classes or {},
                class_names=list(trait.classes.keys()) if trait.classes else [],
            )
            traits_info.append(info)
        return traits_info

    def _create_classifier(self, llm_config: AdeleModelConfig | None = None) -> QuestionClassifier:
        """Create a classifier with the given config or return the default.

        Args:
            llm_config: Optional model configuration. If None, returns the default classifier.

        Returns:
            QuestionClassifier instance configured as specified.
        """
        if llm_config is None:
            return self.classifier

        return QuestionClassifier(
            model_name=llm_config.model_name,
            provider=llm_config.provider,
            temperature=llm_config.temperature,
            interface=llm_config.interface,
            endpoint_base_url=llm_config.endpoint_base_url,
            endpoint_api_key=llm_config.endpoint_api_key,
            trait_eval_mode=llm_config.trait_eval_mode,
        )

    def classify_single(
        self,
        question_text: str,
        trait_names: list[str] | None = None,
        question_id: str | None = None,
        llm_config: AdeleModelConfig | None = None,
    ) -> QuestionClassificationResult:
        """Classify a single question synchronously.

        Args:
            question_text: The question text to classify.
            trait_names: Optional list of trait names. If None, uses all 18 traits.
            question_id: Optional question identifier.
            llm_config: Optional model configuration. If None, uses default settings.

        Returns:
            QuestionClassificationResult with scores and labels.
        """
        classifier = self._create_classifier(llm_config)
        return classifier.classify_single(
            question_text=question_text,
            trait_names=trait_names,
            question_id=question_id,
        )

    def start_batch_job(
        self,
        questions: list[dict[str, str]],
        trait_names: list[str] | None = None,
        llm_config: AdeleModelConfig | None = None,
    ) -> str:
        """Start a batch classification job asynchronously.

        Args:
            questions: List of dicts with 'question_id' and 'question_text' keys.
            trait_names: Optional list of trait names to evaluate. If None, uses all 18.
            llm_config: Optional model configuration. If None, uses default settings.

        Returns:
            Job ID string for tracking the job.

        Raises:
            RuntimeError: If the service is shutting down.
            ValueError: If questions list is empty.
        """
        if self._is_shutdown:
            raise RuntimeError("AdeleClassificationService is shutting down")

        if not questions:
            raise ValueError("Questions list cannot be empty")

        self._maybe_cleanup()

        job_id = str(uuid.uuid4())
        effective_traits = trait_names if trait_names else ADELE_TRAIT_NAMES

        job = ClassificationJob(
            job_id=job_id,
            status="pending",
            total_questions=len(questions),
            trait_names=list(effective_traits),
        )

        self.jobs[job_id] = job

        logger.info(f"📋 Submitting ADeLe classification job {job_id} ({len(questions)} questions)")
        future = self.executor.submit(self._run_batch_classification, job, questions, llm_config)
        self.futures[job_id] = future
        future.add_done_callback(lambda f: self._handle_job_completion(job_id, f))

        return job_id

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get the status of a classification job.

        Args:
            job_id: The job identifier.

        Returns:
            Dict with job status fields, or None if job doesn't exist.
        """
        self._maybe_cleanup()

        job = self.jobs.get(job_id)
        if not job:
            return None

        with self._get_job_lock(job_id):
            return job.to_dict()

    def get_job_results(self, job_id: str) -> dict[str, QuestionClassificationResult] | None:
        """Get the results of a completed job.

        Args:
            job_id: The job identifier.

        Returns:
            Dict mapping question_id to classification result, or None if not found/incomplete.
        """
        job = self.jobs.get(job_id)
        if not job:
            return None

        with self._get_job_lock(job_id):
            if job.status == "completed":
                return job.results.copy()
            return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a classification job.

        Args:
            job_id: The job identifier.

        Returns:
            True if cancellation succeeded, False otherwise.
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        with self._get_job_lock(job_id):
            current = ClassificationJobStatus(job.status)
            if current.can_transition_to(ClassificationJobStatus.CANCELLED):
                job.status = ClassificationJobStatus.CANCELLED.value
                return True
        return False

    def _run_batch_classification(
        self,
        job: ClassificationJob,
        questions: list[dict[str, str]],
        llm_config: AdeleModelConfig | None = None,
    ) -> None:
        """Execute batch classification in a worker thread."""
        try:
            # Transition to running
            with self._get_job_lock(job.job_id):
                current = ClassificationJobStatus(job.status)
                if not current.can_transition_to(ClassificationJobStatus.RUNNING):
                    logger.warning(f"Job {job.job_id} could not transition to RUNNING")
                    return
                job.status = ClassificationJobStatus.RUNNING.value
                job.start_time = time.time()

            self._emit_progress_event(job.job_id, "job_started")

            # Convert questions to tuple format for classifier
            question_tuples = [(q["question_id"], q["question_text"]) for q in questions]

            def progress_callback(completed: int, total: int) -> None:
                """Update progress and broadcast."""
                with self._get_job_lock(job.job_id):
                    job.completed_count = completed
                    job.percentage = (completed / total) * 100 if total > 0 else 0
                    if completed < len(question_tuples):
                        job.current_question_id = question_tuples[completed][0]

                self._emit_progress_event(
                    job.job_id,
                    "progress",
                    {"completed": completed, "total": total},
                )

            # Create classifier (with custom config if provided)
            classifier = self._create_classifier(llm_config)

            # Run classification
            results = classifier.classify_batch(
                questions=question_tuples,
                trait_names=job.trait_names if job.trait_names else None,
                on_progress=progress_callback,
            )

            # Complete
            with self._get_job_lock(job.job_id):
                job.results = results
                job.status = ClassificationJobStatus.COMPLETED.value
                job.end_time = time.time()
                job.percentage = 100.0
                job.current_question_id = None

            # Store in historical results
            self.historical_results[job.job_id] = results

            self._emit_progress_event(job.job_id, "job_completed")

        except Exception as e:
            logger.error(f"Classification job {job.job_id} failed: {e}", exc_info=True)
            with self._get_job_lock(job.job_id):
                job.error_message = str(e)
                job.end_time = time.time()
                job.status = ClassificationJobStatus.FAILED.value

            self._emit_progress_event(job.job_id, "job_failed", {"error": str(e)})

    def _handle_job_completion(self, job_id: str, future: Future[None]) -> None:
        """Handle job completion callback."""
        job = self.jobs.get(job_id)
        if not job:
            return

        try:
            future.result(timeout=0)
        except Exception as e:
            logger.error(f"Job {job_id} worker exception: {e}", exc_info=True)
            current = self._get_job_status_enum(job_id)
            if current not in (
                ClassificationJobStatus.COMPLETED,
                ClassificationJobStatus.FAILED,
                ClassificationJobStatus.CANCELLED,
            ):
                with self._get_job_lock(job_id):
                    job.error_message = f"Worker thread exception: {e}"
                    job.end_time = time.time()
                    job.status = ClassificationJobStatus.FAILED.value
                self._emit_progress_event(job_id, "job_failed", {"error": str(e)})

    def _get_job_status_enum(self, job_id: str) -> ClassificationJobStatus | None:
        """Get job status as enum."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        with self._get_job_lock(job_id):
            return ClassificationJobStatus(job.status)

    def _emit_progress_event(
        self,
        job_id: str,
        event_type: str,
        extra_data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a progress event to WebSocket subscribers."""
        job = self.jobs.get(job_id)
        if not job:
            return

        with self._get_job_lock(job_id):
            duration = None
            if job.start_time:
                duration = (job.end_time or time.time()) - job.start_time

            event_data = {
                "type": event_type,
                "job_id": job_id,
                "status": job.status,
                "percentage": job.percentage,
                "completed": job.completed_count,
                "total": job.total_questions,
                "current_question_id": job.current_question_id,
                "duration_seconds": duration,
            }

        if extra_data:
            event_data.update(extra_data)

        self.broadcaster.broadcast_from_thread(job_id, event_data)

    def _maybe_cleanup(self) -> None:
        """Run cleanup if sufficient time has passed."""
        current_time = time.time()
        if current_time - self._last_cleanup_time >= _CLEANUP_INTERVAL_SECONDS:
            self._cleanup_old_jobs()
            self._cleanup_historical_results()
            self._last_cleanup_time = current_time

    def _cleanup_old_jobs(self) -> int:
        """Remove old jobs to prevent memory leaks."""
        current_time = time.time()
        jobs_to_remove = []

        for job_id, job in self.jobs.items():
            if job.start_time:
                age_hours = (current_time - job.start_time) / 3600
                if age_hours > _MAX_JOB_AGE_HOURS:
                    jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            self.futures.pop(job_id, None)
            self.historical_results.pop(job_id, None)
            self._remove_job_lock(job_id)

        return len(jobs_to_remove)

    def _cleanup_historical_results(self) -> int:
        """Trim historical results to stay within limit."""
        removed = 0
        while len(self.historical_results) > _MAX_HISTORICAL_RESULTS:
            self.historical_results.popitem(last=False)
            removed += 1
        return removed


# Global service instance
adele_classification_service = AdeleClassificationService()
