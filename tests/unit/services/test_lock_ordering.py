"""Unit tests for lock ordering and deadlock prevention (conc-005).

These tests verify that services follow the documented lock hierarchy
and that locks are properly released after operations.
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from karenina_server.api.auth_handlers import CsrfTokenStore
from karenina_server.services.generation_service import GenerationService
from karenina_server.services.verification_service import JobStatus, VerificationService


@pytest.mark.unit
@pytest.mark.service
class TestCsrfTokenStoreLocking:
    """Test CsrfTokenStore lock behavior."""

    def test_store_lock_released_after_generate(self):
        """Test that _store_lock is released after generate_token."""
        store = CsrfTokenStore()
        store.clear_all()

        store.generate_token("client-1")

        # Lock should be released - we should be able to acquire it
        acquired = store._store_lock.acquire(blocking=False)
        assert acquired, "_store_lock was not released after generate_token"
        store._store_lock.release()

    def test_store_lock_released_after_validate(self):
        """Test that _store_lock is released after validate_token."""
        store = CsrfTokenStore()
        store.clear_all()

        token = store.generate_token("client-1")
        store.validate_token("client-1", token)

        # Lock should be released
        acquired = store._store_lock.acquire(blocking=False)
        assert acquired, "_store_lock was not released after validate_token"
        store._store_lock.release()

    def test_concurrent_token_operations(self):
        """Test that concurrent token operations don't deadlock."""
        store = CsrfTokenStore()
        store.clear_all()
        results = []
        errors = []

        def generate_and_validate(client_id: str):
            try:
                token = store.generate_token(client_id)
                valid = store.validate_token(client_id, token)
                results.append((client_id, valid))
            except Exception as e:
                errors.append((client_id, e))

        # Run 10 concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(generate_and_validate, f"client-{i}") for i in range(10)]
            for future in futures:
                future.result(timeout=5.0)  # Should complete within 5 seconds

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(valid for _, valid in results)


@pytest.mark.unit
@pytest.mark.service
class TestVerificationServiceLocking:
    """Test VerificationService lock hierarchy."""

    def test_job_lock_released_after_get_status(self):
        """Test that job lock is released after get_job_status."""
        service = VerificationService(max_workers=1)
        service.jobs["test-job"] = MagicMock(
            job_id="test-job",
            status="pending",
            to_dict=MagicMock(return_value={"status": "pending"}),
        )

        # Get status (should acquire and release lock)
        service.get_job_status("test-job")

        # Lock should be released
        lock = service._get_job_lock("test-job")
        acquired = lock.acquire(blocking=False)
        assert acquired, "Job lock was not released after get_job_status"
        lock.release()

    def test_master_lock_released_after_get_job_lock(self):
        """Test that master lock is released after getting a job lock."""
        service = VerificationService(max_workers=1)

        # Get a job lock (creates one if needed)
        service._get_job_lock("test-job")

        # Master lock should be released
        acquired = service._master_lock.acquire(blocking=False)
        assert acquired, "_master_lock was not released after _get_job_lock"
        service._master_lock.release()

    def test_status_transition_releases_lock(self):
        """Test that lock is released after status transition."""
        service = VerificationService(max_workers=1)
        service.jobs["test-job"] = MagicMock(job_id="test-job", status="pending")

        # Transition status
        service._transition_status("test-job", JobStatus.PENDING, JobStatus.RUNNING)

        # Lock should be released
        lock = service._get_job_lock("test-job")
        acquired = lock.acquire(blocking=False)
        assert acquired, "Job lock was not released after _transition_status"
        lock.release()

    def test_shutdown_lock_released_after_is_shutdown(self):
        """Test that shutdown lock is not held during is_shutdown check."""
        service = VerificationService(max_workers=1)

        # Call is_shutdown (should not hold locks)
        _ = service.is_shutdown()

        # Shutdown lock should not be held (it's not acquired in is_shutdown)
        acquired = service._shutdown_lock.acquire(blocking=False)
        assert acquired, "_shutdown_lock was unexpectedly held"
        service._shutdown_lock.release()


@pytest.mark.unit
@pytest.mark.service
class TestGenerationServiceLocking:
    """Test GenerationService lock behavior."""

    def test_shutdown_lock_released_after_is_shutdown(self):
        """Test that shutdown lock is not held after is_shutdown check."""
        service = GenerationService(max_workers=1)

        # Call is_shutdown
        _ = service.is_shutdown()

        # Shutdown lock should not be held
        acquired = service._shutdown_lock.acquire(blocking=False)
        assert acquired, "_shutdown_lock was unexpectedly held"
        service._shutdown_lock.release()


@pytest.mark.unit
@pytest.mark.service
class TestLockHierarchy:
    """Test lock hierarchy is correctly documented and followed."""

    def test_verification_service_documents_lock_hierarchy(self):
        """Test that VerificationService docstring documents lock hierarchy."""
        docstring = VerificationService.__doc__ or ""
        assert "Lock Hierarchy" in docstring, "VerificationService missing lock hierarchy docs"
        assert "_shutdown_lock" in docstring
        assert "_master_lock" in docstring
        assert "_job_locks" in docstring

    def test_generation_service_documents_lock_hierarchy(self):
        """Test that GenerationService docstring documents lock hierarchy."""
        docstring = GenerationService.__doc__ or ""
        assert "Lock Hierarchy" in docstring, "GenerationService missing lock hierarchy docs"
        assert "_shutdown_lock" in docstring

    def test_csrf_token_store_documents_lock_hierarchy(self):
        """Test that CsrfTokenStore docstring documents lock hierarchy."""
        docstring = CsrfTokenStore.__doc__ or ""
        assert "Lock Hierarchy" in docstring, "CsrfTokenStore missing lock hierarchy docs"
        assert "_lock" in docstring
        assert "_store_lock" in docstring


@pytest.mark.unit
@pytest.mark.service
class TestNoDeadlockScenarios:
    """Test that common operations don't cause deadlocks."""

    def test_concurrent_job_status_queries(self):
        """Test that concurrent status queries don't deadlock."""
        service = VerificationService(max_workers=1)
        job_ids = [f"job-{i}" for i in range(5)]

        # Create mock jobs
        for job_id in job_ids:
            service.jobs[job_id] = MagicMock(
                job_id=job_id,
                status="running",
                to_dict=MagicMock(return_value={"status": "running"}),
            )

        results = []
        errors = []

        def query_status(job_id: str):
            try:
                for _ in range(10):  # Multiple queries per thread
                    status = service.get_job_status(job_id)
                    results.append((job_id, status is not None))
            except Exception as e:
                errors.append((job_id, e))

        # Run concurrent queries
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_status, job_id) for job_id in job_ids]
            for future in futures:
                future.result(timeout=10.0)

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50  # 5 jobs * 10 queries each
