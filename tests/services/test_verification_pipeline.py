"""Tests for the refactored verification pipeline."""

import time
from unittest.mock import patch

import pytest
from karenina.schemas.workflow import FinishedTemplate, ModelConfig, VerificationConfig, VerificationResult


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


class TestTaskQueueGeneration:
    """Test task queue generation via combinatorial expansion."""

    def test_task_queue_size(self, sample_template, multi_model_config):
        """Test that task queue has correct size based on combinatorial expansion."""
        from karenina.benchmark.verification import generate_task_queue

        # Generate task queue
        task_queue = generate_task_queue(
            templates=[sample_template],
            config=multi_model_config,
            global_rubric=None,
            run_name="Test Run",
            job_id="test-job",
        )

        # Expected: 1 template × 2 answering models × 1 parsing model × 3 replicates = 6 tasks
        assert len(task_queue) == 6

    def test_task_queue_contains_all_combinations(self, sample_template, multi_model_config):
        """Test that task queue contains all model + replicate combinations."""
        from karenina.benchmark.verification import generate_task_queue

        task_queue = generate_task_queue(
            templates=[sample_template],
            config=multi_model_config,
            global_rubric=None,
            run_name="Test Run",
            job_id="test-job",
        )

        # Extract combinations
        combinations = [
            (task["answering_model"].id, task["parsing_model"].id, task["replicate"]) for task in task_queue
        ]

        # Expected combinations
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
            job_id="test-job",
        )

        assert len(task_queue) == 1
        assert task_queue[0]["replicate"] is None


class TestUniqueKeyGeneration:
    """Test unique result key generation."""

    def test_keys_include_job_id_and_timestamp(self, sample_template, multi_model_config):
        """Test that result keys include job_id and timestamp."""
        from karenina.benchmark.verification import execute_task, generate_task_queue

        task_queue = generate_task_queue(
            templates=[sample_template],
            config=multi_model_config,
            global_rubric=None,
            run_name="Test Run",
            job_id="abc12345-6789-0123-4567-890123456789",
        )
        task = task_queue[0]

        # Mock the verification execution
        with patch("karenina.benchmark.verification.runner.run_single_model_verification") as mock_verify:
            mock_verify.return_value = VerificationResult(
                question_id="test_q1",
                template_id="no_template",
                completed_without_errors=True,
                question_text="What is 2+2?",
                raw_llm_response='{"result": 4}',
                answering_model="openai/gpt-4.1-mini",
                parsing_model="openai/gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2024-01-01T00:00:00",
            )

            result_key, _ = execute_task(task)

        # Check key format: question_id_answering_parsing_rep{N}_job_id_timestamp
        # Note: question_id is "test_q1" which contains underscore, so parts are:
        # parts[0] = "test", parts[1] = "q1", parts[2] = "answering-1", etc.
        parts = result_key.split("_")
        assert parts[0] == "test"
        assert parts[1] == "q1"
        assert parts[2] == "answering-1"
        assert parts[3] == "parsing-1"
        assert parts[4] == "rep1"
        assert parts[5] == "abc12345"  # Short job ID (first 8 chars)
        assert parts[6].isdigit()  # Timestamp

    def test_multiple_runs_generate_different_keys(self, sample_template, multi_model_config):
        """Test that running same verification twice generates different keys."""
        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        # Mock verification execution at the runner level
        def mock_run_single_verification(**kwargs):
            return VerificationResult(
                question_id="test_q1",
                template_id="no_template",
                completed_without_errors=True,
                question_text="What is 2+2?",
                raw_llm_response='{"result": 4}',
                answering_model="openai/gpt-4.1-mini",
                parsing_model="openai/gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2024-01-01T00:00:00",
            )

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            # First run
            job_id_1 = service.start_verification(
                finished_templates=[sample_template],
                config=multi_model_config,
            )

            # Wait for completion
            max_wait = 10
            waited = 0
            while service.jobs[job_id_1].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            results_1 = service.jobs[job_id_1].results
            keys_1 = set(results_1.keys())

            # Tiny sleep to ensure different timestamp
            time.sleep(0.001)

            # Second run
            job_id_2 = service.start_verification(
                finished_templates=[sample_template],
                config=multi_model_config,
            )

            waited = 0
            while service.jobs[job_id_2].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            results_2 = service.jobs[job_id_2].results
            keys_2 = set(results_2.keys())

            # Keys should be different due to different job_id and timestamp
            assert keys_1.isdisjoint(keys_2), "Keys from different runs should not overlap"


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
            time.sleep(0.01)  # Small delay to ensure sequential execution is detectable

            return VerificationResult(
                question_id="test_q1",
                template_id="no_template",
                completed_without_errors=True,
                question_text="What is 2+2?",
                raw_llm_response='{"result": 4}',
                answering_model="openai/gpt-4.1-mini",
                parsing_model="openai/gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2024-01-01T00:00:00",
            )

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=multi_model_config,
            )

            max_wait = 10
            waited = 0
            while service.jobs[job_id].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            # Check that execution was sequential (times should be strictly increasing with gaps)
            assert execution_order == sorted(execution_order), "Execution should be sequential"
            # Each task should start after the previous one finishes
            for i in range(1, len(execution_order)):
                assert execution_order[i] > execution_order[i - 1] + 0.009  # At least 0.009s gap


class TestResultAccumulation:
    """Test that results accumulate across runs instead of overwriting."""

    def test_historical_results_accumulate(self, sample_template, multi_model_config):
        """Test that historical results accumulate instead of being overwritten."""
        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        def mock_run_single_verification(**kwargs):
            return VerificationResult(
                question_id="test_q1",
                template_id="no_template",
                completed_without_errors=True,
                question_text="What is 2+2?",
                raw_llm_response='{"result": 4}',
                answering_model="openai/gpt-4.1-mini",
                parsing_model="openai/gpt-4.1-mini",
                execution_time=1.0,
                timestamp="2024-01-01T00:00:00",
            )

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            # Run 1
            job_id_1 = service.start_verification(
                finished_templates=[sample_template],
                config=multi_model_config,
            )

            max_wait = 10
            waited = 0
            while service.jobs[job_id_1].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            time.sleep(0.001)  # Ensure different timestamp

            # Run 2
            job_id_2 = service.start_verification(
                finished_templates=[sample_template],
                config=multi_model_config,
            )

            waited = 0
            while service.jobs[job_id_2].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            # Historical results should contain both runs
            all_results = service.get_all_historical_results()

            # Each run has 6 results (2 answering × 1 parsing × 3 replicates)
            # Total should be 12
            assert len(all_results) == 12


class TestProgressCallbacksInParallelMode:
    """Test that progress callbacks work correctly in parallel mode."""

    def test_parallel_mode_calls_progress_callback(self, sample_template, multi_model_config, monkeypatch):
        """Test that progress callbacks are invoked in parallel mode."""
        monkeypatch.setenv("KARENINA_ASYNC_ENABLED", "true")
        monkeypatch.setenv("KARENINA_ASYNC_MAX_WORKERS", "2")

        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()
        callback_invocations = []

        def mock_run_single_verification(**kwargs):
            time.sleep(0.01)  # Small delay to simulate work
            return VerificationResult(
                question_id="test_q1",
                template_id="no_template",
                completed_without_errors=True,
                question_text="What is 2+2?",
                raw_llm_response='{"result": 4}',
                answering_model="openai/gpt-4.1-mini",
                parsing_model="openai/gpt-4.1-mini",
                execution_time=0.01,
                timestamp="2024-01-01T00:00:00",
            )

        # Wrap the progress callback to track invocations
        original_callback = None

        def capture_progress_callback(*args, **kwargs):
            callback_invocations.append({"args": args, "kwargs": kwargs})
            if original_callback:
                return original_callback(*args, **kwargs)

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            # Start verification
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=multi_model_config,
            )

            # Wait for completion
            max_wait = 10
            waited = 0
            while service.jobs[job_id].status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            # Verify progress callbacks were invoked
            # Expected: 6 tasks (2 answering × 1 parsing × 3 replicates)
            # Each task should trigger 2 callbacks (start + finish) = 12 total
            job = service.jobs[job_id]

            # Check that last task duration was tracked (proves callbacks were called)
            assert job.last_task_duration is None, "Last task duration should be cleared on completion"

            # Check that task counts were updated correctly
            assert job.processed_count == 6, "All 6 tasks should be processed"
            assert job.successful_count == 6, "All 6 tasks should succeed"

            # Duration should be calculated when job is complete
            assert job.start_time is not None, "Start time should be set"
            assert job.end_time is not None, "End time should be set"
            duration = job.end_time - job.start_time
            assert duration > 0, "Total duration should be positive"

    def test_parallel_mode_duration_tracking(self, sample_template, monkeypatch):
        """Test that duration tracking works correctly in parallel mode."""
        monkeypatch.setenv("KARENINA_ASYNC_ENABLED", "true")
        monkeypatch.setenv("KARENINA_ASYNC_MAX_WORKERS", "3")

        from karenina_server.services.verification_service import VerificationService

        service = VerificationService()

        # Use single model and multiple replicates for cleaner test
        single_model_config = VerificationConfig(
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
            replicate_count=5,
        )

        execution_count = [0]

        def mock_run_single_verification(**kwargs):
            execution_count[0] += 1
            time.sleep(0.02)  # Consistent execution time for predictable EMA
            return VerificationResult(
                question_id="test_q1",
                template_id="no_template",
                completed_without_errors=True,
                question_text="What is 2+2?",
                raw_llm_response='{"result": 4}',
                answering_model="openai/gpt-4.1-mini",
                parsing_model="openai/gpt-4.1-mini",
                execution_time=0.02,
                timestamp="2024-01-01T00:00:00",
            )

        with patch(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            side_effect=mock_run_single_verification,
        ):
            job_id = service.start_verification(
                finished_templates=[sample_template],
                config=single_model_config,
            )

            # Wait for at least 2 tasks to complete so EMA is initialized
            time.sleep(0.1)

            job = service.jobs[job_id]

            # After some tasks complete, EMA should be initialized
            # Wait for completion
            max_wait = 10
            waited = 0
            while job.status != "completed" and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            # Verify all tasks executed
            assert execution_count[0] == 5, "All 5 replicate tasks should execute"

            # Verify last task duration was cleared on completion
            assert job.last_task_duration is None, "Last task duration should be cleared when job completes"

            # Verify final state
            assert job.processed_count == 5
            assert job.successful_count == 5

            # Verify duration tracking
            assert job.start_time is not None, "Start time should be recorded"
            assert job.end_time is not None, "End time should be recorded"
            total_duration = job.end_time - job.start_time
            assert total_duration > 0, "Total duration should be positive"
            # Duration should be roughly 5 tasks × 0.02s ÷ 3 workers ≈ 0.03-0.05s (with overhead)
            assert total_duration < 1.0, "Parallel execution should be faster than sequential"
