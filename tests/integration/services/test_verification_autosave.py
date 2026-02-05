"""Integration tests for verification service auto-save functionality.

Tests database persistence during verification runs.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from karenina.schemas.verification import ModelIdentity
from karenina.schemas.workflow import FinishedTemplate, VerificationConfig, VerificationResult
from karenina.schemas.workflow.verification.result_components import VerificationResultMetadata


def make_mock_result(question_id: str = "test_q1") -> VerificationResult:
    """Create a mock verification result with proper structure."""
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="no_template",
            completed_without_errors=True,
            question_text="What is 2+2?",
            answering=ModelIdentity(interface="langchain", model_name="claude-haiku-4-5"),
            parsing=ModelIdentity(interface="langchain", model_name="claude-haiku-4-5"),
            execution_time=1.0,
            timestamp="2024-01-01T00:00:00",
            result_id="test123456789012",
        )
    )


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield f"sqlite:///{db_path}"
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
    from karenina.schemas.config import ModelConfig

    return VerificationConfig(
        answering_models=[
            ModelConfig(
                id="test-answering",
                model_provider="openai",
                model_name="gpt-4.1-mini",
                interface="langchain",
                temperature=0.0,
                system_prompt="You are a helpful assistant.",
            )
        ],
        parsing_models=[
            ModelConfig(
                id="test-parsing",
                model_provider="openai",
                model_name="gpt-4.1-mini",
                interface="langchain",
                temperature=0.0,
                system_prompt="You are a validation assistant.",
            )
        ],
        replicate_count=1,
    )


@pytest.mark.integration
@pytest.mark.service
class TestAutoSaveEnabled:
    """Test auto-save functionality when enabled."""

    def test_autosave_with_storage_url(self, temp_sqlite_db, sample_template, basic_config, monkeypatch):
        """Test that results are saved when storage_url is provided and AUTOSAVE_DATABASE=true."""
        from karenina_server.services.verification_service import VerificationService

        monkeypatch.setenv("AUTOSAVE_DATABASE", "true")

        from karenina.storage import DBConfig, init_database

        init_database(DBConfig(storage_url=temp_sqlite_db))

        service = VerificationService()

        def mock_run_single_verification(**kwargs):
            return make_mock_result()

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=basic_config,
                storage_url=temp_sqlite_db,
                benchmark_name="Test Benchmark",
            )

            max_wait = 10
            waited = 0.0
            while service.jobs[job_id].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            assert service.jobs[job_id].status == "completed"
            time.sleep(0.2)

            from karenina.storage import DBConfig, get_database_statistics

            stats = get_database_statistics(DBConfig(storage_url=temp_sqlite_db))
            assert stats["total_verification_runs"] > 0


@pytest.mark.integration
@pytest.mark.service
class TestAutoSaveDisabled:
    """Test auto-save functionality when disabled."""

    def test_autosave_disabled_by_env_var(self, temp_sqlite_db, sample_template, basic_config, monkeypatch):
        """Test that results are NOT saved when AUTOSAVE_DATABASE=false."""
        monkeypatch.setenv("AUTOSAVE_DATABASE", "false")

        from karenina.storage import DBConfig, init_database

        init_database(DBConfig(storage_url=temp_sqlite_db))

        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        def mock_run_single_verification(**kwargs):
            return make_mock_result()

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=basic_config,
                storage_url=temp_sqlite_db,
            )

            max_wait = 10
            waited = 0.0
            while service.jobs[job_id].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            assert service.jobs[job_id].status == "completed"

            from karenina.storage import DBConfig, get_database_statistics

            stats = get_database_statistics(DBConfig(storage_url=temp_sqlite_db))
            assert stats["total_verification_runs"] == 0

    def test_autosave_skipped_without_storage_url(self, sample_template, basic_config, monkeypatch):
        """Test that auto-save is skipped when no storage_url is provided."""
        monkeypatch.setenv("AUTOSAVE_DATABASE", "true")

        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        def mock_run_single_verification(**kwargs):
            return make_mock_result()

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=basic_config,
            )

            max_wait = 10
            waited = 0.0
            while service.jobs[job_id].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            assert service.jobs[job_id].status == "completed"


@pytest.mark.integration
@pytest.mark.service
class TestAutoSaveErrorHandling:
    """Test error handling in auto-save functionality."""

    def test_autosave_failure_does_not_fail_job(self, sample_template, basic_config, monkeypatch):
        """Test that auto-save failures don't cause the verification job to fail."""
        monkeypatch.setenv("AUTOSAVE_DATABASE", "true")

        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        def mock_run_single_verification(**kwargs):
            return make_mock_result()

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=basic_config,
                storage_url="invalid://url",
            )

            max_wait = 10
            waited = 0.0
            while service.jobs[job_id].status not in ["completed", "failed"] and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            assert service.jobs[job_id].status == "completed"


@pytest.mark.integration
@pytest.mark.service
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
