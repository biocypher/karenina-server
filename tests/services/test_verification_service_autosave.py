"""Tests for verification service auto-save functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from karenina.benchmark.models import FinishedTemplate, VerificationConfig


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield f"sqlite:///{db_path}"

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_template():
    """Create a sample finished template for testing."""
    return FinishedTemplate(
        question_id="test_q1",
        question_text="What is 2+2?",
        question_preview="What is 2+2?",
        template_code="class Answer(BaseModel):\n    result: int\n\n    def verify(self) -> bool:\n        return self.result == 4",
        last_modified="2024-01-01T00:00:00",
        few_shot_examples=None,
        question_rubric=None,
        keywords=["math", "arithmetic"],
    )


@pytest.fixture
def basic_config():
    """Create a basic verification config."""
    return VerificationConfig(
        answering_model_provider="openai",
        answering_model_name="gpt-4.1-mini",
        parsing_model_provider="openai",
        parsing_model_name="gpt-4.1-mini",
        replicate_count=1,
    )


class TestAutoSaveEnabled:
    """Test auto-save functionality when enabled."""

    def test_autosave_with_storage_url(self, temp_sqlite_db, sample_template, basic_config, monkeypatch):
        """Test that results are saved when storage_url is provided and AUTOSAVE_DATABASE=true."""
        from karenina_server.services.verification_service import VerificationService

        # Set environment variable
        monkeypatch.setenv("AUTOSAVE_DATABASE", "true")

        # Initialize database
        from karenina.storage import DBConfig, init_database

        init_database(DBConfig(storage_url=temp_sqlite_db))

        service = VerificationService()

        # Mock run_question_verification to avoid actual LLM calls
        with patch("karenina_server.services.verification_service.run_question_verification") as mock_verify:
            from karenina.benchmark.models import VerificationResult

            mock_result = VerificationResult(
                question_id="test_q1",
                success=True,
                verified=True,
                question_text="What is 2+2?",
                raw_llm_response='{"result": 4}',
                answering_model="openai/gpt-4.1-mini",
                parsing_model="openai/gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2024-01-01T00:00:00",
            )
            mock_verify.return_value = {"test_q1": mock_result}

            # Start verification with storage_url and benchmark_name
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=basic_config,
                storage_url=temp_sqlite_db,
                benchmark_name="Test Benchmark",
            )

            # Wait for completion
            import time

            max_wait = 10
            waited = 0
            while service.jobs[job_id].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            assert service.jobs[job_id].status == "completed"

            # Give extra time for database transactions to complete
            time.sleep(0.2)

            # Verify results were saved to database
            from karenina.storage import DBConfig, get_verification_run_summary

            summaries = get_verification_run_summary(DBConfig(storage_url=temp_sqlite_db))
            assert len(summaries) > 0

    def test_autosave_creates_benchmark_if_not_exists(self, temp_sqlite_db, sample_template, basic_config, monkeypatch):
        """Test that auto-save creates benchmark entry if it doesn't exist."""
        monkeypatch.setenv("AUTOSAVE_DATABASE", "true")

        from karenina.storage import DBConfig, init_database

        init_database(DBConfig(storage_url=temp_sqlite_db))

        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        with patch("karenina_server.services.verification_service.run_question_verification") as mock_verify:
            from karenina.benchmark.models import VerificationResult

            mock_result = VerificationResult(
                question_id="test_q1",
                success=True,
                verified=True,
                question_text="What is 2+2?",
                raw_llm_response='{"result": 4}',
                answering_model="openai/gpt-4.1-mini",
                parsing_model="openai/gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2024-01-01T00:00:00",
            )
            mock_verify.return_value = {"test_q1": mock_result}

            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=basic_config,
                run_name="Test Run",
                storage_url=temp_sqlite_db,
                benchmark_name="Test Run",  # Use run name as benchmark name
            )

            import time

            max_wait = 10
            waited = 0
            while service.jobs[job_id].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            # Give extra time for database transactions to complete
            time.sleep(0.2)

            # Check that benchmark was created
            from karenina.storage import DBConfig, get_benchmark_summary

            summaries = get_benchmark_summary(DBConfig(storage_url=temp_sqlite_db))
            # Auto-save uses run_name as benchmark name
            run_names = [s["benchmark_name"] for s in summaries]
            assert "Test Run" in run_names


class TestAutoSaveDisabled:
    """Test auto-save functionality when disabled."""

    def test_autosave_disabled_by_env_var(self, temp_sqlite_db, sample_template, basic_config, monkeypatch):
        """Test that results are NOT saved when AUTOSAVE_DATABASE=false."""
        monkeypatch.setenv("AUTOSAVE_DATABASE", "false")

        from karenina.storage import DBConfig, init_database

        init_database(DBConfig(storage_url=temp_sqlite_db))

        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        with patch("karenina_server.services.verification_service.run_question_verification") as mock_verify:
            from karenina.benchmark.models import VerificationResult

            mock_result = VerificationResult(
                question_id="test_q1",
                success=True,
                verified=True,
                question_text="What is 2+2?",
                raw_llm_response='{"result": 4}',
                answering_model="openai/gpt-4.1-mini",
                parsing_model="openai/gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2024-01-01T00:00:00",
            )
            mock_verify.return_value = {"test_q1": mock_result}

            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=basic_config,
                storage_url=temp_sqlite_db,
            )

            import time

            max_wait = 10
            waited = 0
            while service.jobs[job_id].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            assert service.jobs[job_id].status == "completed"

            # Verify NO results were saved to database
            from karenina.storage import DBConfig, get_verification_run_summary

            summaries = get_verification_run_summary(DBConfig(storage_url=temp_sqlite_db))
            assert len(summaries) == 0

    def test_autosave_skipped_without_storage_url(self, sample_template, basic_config, monkeypatch):
        """Test that auto-save is skipped when no storage_url is provided."""
        monkeypatch.setenv("AUTOSAVE_DATABASE", "true")

        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        with patch("karenina_server.services.verification_service.run_question_verification") as mock_verify:
            from karenina.benchmark.models import VerificationResult

            mock_result = VerificationResult(
                question_id="test_q1",
                success=True,
                verified=True,
                question_text="What is 2+2?",
                raw_llm_response='{"result": 4}',
                answering_model="openai/gpt-4.1-mini",
                parsing_model="openai/gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2024-01-01T00:00:00",
            )
            mock_verify.return_value = {"test_q1": mock_result}

            # No storage_url provided
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=basic_config,
            )

            import time

            max_wait = 10
            waited = 0
            while service.jobs[job_id].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            # Should complete successfully without auto-save
            assert service.jobs[job_id].status == "completed"


class TestAutoSaveErrorHandling:
    """Test error handling in auto-save functionality."""

    def test_autosave_failure_does_not_fail_job(self, sample_template, basic_config, monkeypatch):
        """Test that auto-save failures don't cause the verification job to fail."""
        monkeypatch.setenv("AUTOSAVE_DATABASE", "true")

        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        with patch("karenina_server.services.verification_service.run_question_verification") as mock_verify:
            from karenina.benchmark.models import VerificationResult

            mock_result = VerificationResult(
                question_id="test_q1",
                success=True,
                verified=True,
                question_text="What is 2+2?",
                raw_llm_response='{"result": 4}',
                answering_model="openai/gpt-4.1-mini",
                parsing_model="openai/gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2024-01-01T00:00:00",
            )
            mock_verify.return_value = {"test_q1": mock_result}

            # Provide invalid storage URL to trigger save error
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=basic_config,
                storage_url="invalid://url",
            )

            import time

            max_wait = 10
            waited = 0
            while service.jobs[job_id].status not in ["completed", "failed"] and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            # Job should still complete even though auto-save failed
            assert service.jobs[job_id].status == "completed"


class TestStorageUrlInJob:
    """Test that storage_url is properly stored in VerificationJob."""

    def test_storage_url_stored_in_job(self, temp_sqlite_db, sample_template, basic_config):
        """Test that storage_url is stored in the job object."""
        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        job_id = service.start_verification(
            finished_templates=[sample_template],
            config=basic_config,
            storage_url=temp_sqlite_db,
        )

        job = service.jobs[job_id]
        assert job.storage_url == temp_sqlite_db

    def test_no_storage_url_in_job(self, sample_template, basic_config):
        """Test that storage_url is None when not provided."""
        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        job_id = service.start_verification(
            finished_templates=[sample_template],
            config=basic_config,
        )

        job = service.jobs[job_id]
        assert job.storage_url is None
