"""Integration tests for the verification pipeline.

Tests verification task queue generation, execution, and result handling.
"""

import time
from unittest.mock import patch

import pytest
from karenina.schemas.verification import ModelIdentity
from karenina.schemas.workflow import FinishedTemplate, ModelConfig, VerificationConfig, VerificationResult
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
def multi_model_config():
    """Create a multi-model verification config."""
    return VerificationConfig(
        answering_models=[
            ModelConfig(
                id="answering-1",
                model_provider="openai",
                model_name="gpt-4.1-mini",
                interface="langchain",
                temperature=0.0,
                system_prompt="You are a helpful assistant.",
            ),
            ModelConfig(
                id="answering-2",
                model_provider="google_genai",
                model_name="gemini-2.5-flash",
                interface="langchain",
                temperature=0.0,
                system_prompt="You are a helpful assistant.",
            ),
        ],
        parsing_models=[
            ModelConfig(
                id="parsing-1",
                model_provider="openai",
                model_name="gpt-4.1-mini",
                interface="langchain",
                temperature=0.0,
                system_prompt="You are a validation assistant.",
            )
        ],
        replicate_count=3,
    )


@pytest.mark.integration
@pytest.mark.service
class TestTaskQueueGeneration:
    """Test task queue generation via combinatorial expansion."""

    def test_task_queue_size(self, sample_template, multi_model_config):
        """Test that task queue has correct size based on combinatorial expansion."""
        from karenina.benchmark.verification import generate_task_queue

        task_queue = generate_task_queue(
            templates=[sample_template],
            config=multi_model_config,
            global_rubric=None,
            run_name="Test Run",
        )

        # Expected: 1 template x 2 answering models x 1 parsing model x 3 replicates = 6 tasks
        assert len(task_queue) == 6

    def test_task_queue_contains_all_combinations(self, sample_template, multi_model_config):
        """Test that task queue contains all model + replicate combinations."""
        from karenina.benchmark.verification import generate_task_queue

        task_queue = generate_task_queue(
            templates=[sample_template],
            config=multi_model_config,
            global_rubric=None,
            run_name="Test Run",
        )

        combinations = [
            (task["answering_model"].id, task["parsing_model"].id, task["replicate"]) for task in task_queue
        ]

        expected = [
            ("answering-1", "parsing-1", 1),
            ("answering-1", "parsing-1", 2),
            ("answering-1", "parsing-1", 3),
            ("answering-2", "parsing-1", 1),
            ("answering-2", "parsing-1", 2),
            ("answering-2", "parsing-1", 3),
        ]

        assert sorted(combinations) == sorted(expected)

    def test_single_replicate_no_replicate_number(self, sample_template):
        """Test that single replicate doesn't include replicate number in task."""
        from karenina.benchmark.verification import generate_task_queue

        single_replicate_config = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="answering-1",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    interface="langchain",
                    temperature=0.0,
                    system_prompt="Test",
                )
            ],
            parsing_models=[
                ModelConfig(
                    id="parsing-1",
                    model_provider="openai",
                    model_name="gpt-4.1-mini",
                    interface="langchain",
                    temperature=0.0,
                    system_prompt="Test",
                )
            ],
            replicate_count=1,
        )

        task_queue = generate_task_queue(
            templates=[sample_template],
            config=single_replicate_config,
            global_rubric=None,
            run_name="Test Run",
        )

        assert len(task_queue) == 1
        assert task_queue[0]["replicate"] is None


@pytest.mark.integration
@pytest.mark.service
class TestUniqueKeyGeneration:
    """Test unique result key generation."""

    @pytest.mark.skip(reason="job_id parameter removed from generate_task_queue API")
    def test_keys_include_job_id_and_timestamp(self, sample_template, multi_model_config):
        """Test that result keys include job_id and timestamp."""
        pass


@pytest.mark.integration
@pytest.mark.service
class TestAsyncControl:
    """Test async execution control via environment variable."""

    def test_sequential_execution_when_async_disabled(self, sample_template, multi_model_config, monkeypatch):
        """Test that tasks execute sequentially when KARENINA_ASYNC_ENABLED=false."""
        monkeypatch.setenv("KARENINA_ASYNC_ENABLED", "false")

        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()
        execution_order = []

        def mock_run_single_verification(**kwargs):
            execution_order.append(time.time())
            time.sleep(0.01)
            return make_mock_result()

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=multi_model_config,
            )

            max_wait = 10
            waited = 0.0
            while service.jobs[job_id].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            assert execution_order == sorted(execution_order)


@pytest.mark.integration
@pytest.mark.service
class TestResultAccumulation:
    """Test that results accumulate across runs instead of overwriting."""

    def test_historical_results_accumulate(self, sample_template, multi_model_config):
        """Test that historical results accumulate instead of being overwritten."""
        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        def mock_run_single_verification(**kwargs):
            return make_mock_result()

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            job_id_1 = service.start_verification(
                finished_templates=[sample_template],
                config=multi_model_config,
            )

            max_wait = 10
            waited = 0.0
            while service.jobs[job_id_1].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            time.sleep(0.001)

            job_id_2 = service.start_verification(
                finished_templates=[sample_template],
                config=multi_model_config,
            )

            waited = 0.0
            while service.jobs[job_id_2].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            all_results = service.get_all_historical_results()
            assert len(all_results) == 12


@pytest.mark.integration
@pytest.mark.service
class TestProgressCallbacksInParallelMode:
    """Test that progress callbacks work correctly in parallel mode."""

    def test_parallel_mode_calls_progress_callback(self, sample_template, multi_model_config, monkeypatch):
        """Test that progress callbacks are invoked in parallel mode."""
        monkeypatch.setenv("KARENINA_ASYNC_ENABLED", "true")
        monkeypatch.setenv("KARENINA_ASYNC_MAX_WORKERS", "2")

        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        def mock_run_single_verification(**kwargs):
            time.sleep(0.01)
            return make_mock_result()

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=multi_model_config,
            )

            max_wait = 10
            waited = 0.0
            while service.jobs[job_id].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            job = service.jobs[job_id]
            assert job.processed_count == 6
            assert job.successful_count == 6
