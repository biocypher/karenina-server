"""Tests for template generation service progress tracking and EMA calculation."""

from unittest.mock import MagicMock

import pytest

from karenina_server.services.generation_service import (
    PROGRESS_EMA_ALPHA,
    GenerationService,
    TemplateGenerationJob,
)


class TestTemplateGenerationJob:
    """Test cases for TemplateGenerationJob progress tracking."""

    def test_task_started_adds_to_in_progress(self):
        """Test that task_started adds question to in_progress_questions."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {"question": "Test?", "raw_answer": "Answer"}},
            config={"model_name": "test"},
            total_questions=1,
        )

        job.task_started("q1")

        assert "q1" in job.in_progress_questions
        assert len(job.in_progress_questions) == 1
        assert job.last_update_ts is not None

    def test_task_started_prevents_duplicates(self):
        """Test that task_started doesn't add duplicates."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {"question": "Test?", "raw_answer": "Answer"}},
            config={"model_name": "test"},
            total_questions=1,
        )

        job.task_started("q1")
        job.task_started("q1")

        assert len(job.in_progress_questions) == 1

    def test_task_finished_removes_from_in_progress(self):
        """Test that task_finished removes question from in_progress_questions."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {"question": "Test?", "raw_answer": "Answer"}},
            config={"model_name": "test"},
            total_questions=1,
        )

        job.task_started("q1")
        job.task_finished("q1", success=True, duration_seconds=2.5)

        assert "q1" not in job.in_progress_questions
        assert job.processed_count == 1
        assert job.successful_count == 1
        assert job.failed_count == 0

    def test_task_finished_updates_counts(self):
        """Test that task_finished correctly updates success/failure counts."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {"question": "Test?", "raw_answer": "Answer"}, "q2": {}},
            config={"model_name": "test"},
            total_questions=2,
        )

        job.task_finished("q1", success=True, duration_seconds=1.0)
        job.task_finished("q2", success=False, duration_seconds=0.5)

        assert job.processed_count == 2
        assert job.successful_count == 1
        assert job.failed_count == 1

    def test_ema_initialization_with_first_task(self):
        """Test that EMA is initialized with the first task duration."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {"question": "Test?", "raw_answer": "Answer"}},
            config={"model_name": "test"},
            total_questions=1,
        )

        job.task_finished("q1", success=True, duration_seconds=5.0)

        assert job.ema_seconds_per_item == 5.0

    def test_ema_calculation_with_multiple_tasks(self):
        """Test that EMA is calculated correctly across multiple tasks."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {}, "q2": {}, "q3": {}},
            config={"model_name": "test"},
            total_questions=3,
        )

        # First task: EMA = duration
        job.task_finished("q1", success=True, duration_seconds=10.0)
        assert job.ema_seconds_per_item == 10.0

        # Second task: EMA = alpha * 5.0 + (1 - alpha) * 10.0
        job.task_finished("q2", success=True, duration_seconds=5.0)
        expected_ema_2 = PROGRESS_EMA_ALPHA * 5.0 + (1 - PROGRESS_EMA_ALPHA) * 10.0
        assert abs(job.ema_seconds_per_item - expected_ema_2) < 0.001

        # Third task: EMA = alpha * 8.0 + (1 - alpha) * previous_ema
        job.task_finished("q3", success=True, duration_seconds=8.0)
        expected_ema_3 = PROGRESS_EMA_ALPHA * 8.0 + (1 - PROGRESS_EMA_ALPHA) * expected_ema_2
        assert abs(job.ema_seconds_per_item - expected_ema_3) < 0.001

    def test_estimated_time_remaining_calculation(self):
        """Test that estimated_time_remaining is calculated from EMA."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {}, "q2": {}, "q3": {}, "q4": {}, "q5": {}},
            config={"model_name": "test"},
            total_questions=5,
        )

        # Process 2 out of 5 tasks
        job.task_finished("q1", success=True, duration_seconds=3.0)
        job.task_finished("q2", success=True, duration_seconds=3.0)

        # EMA should be 3.0, remaining = 3 tasks
        # estimated_time_remaining = 3.0 * 3 = 9.0
        assert job.processed_count == 2
        assert abs(job.estimated_time_remaining - 9.0) < 0.1

    def test_percentage_calculation(self):
        """Test that percentage is calculated correctly."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {}, "q2": {}, "q3": {}, "q4": {}},
            config={"model_name": "test"},
            total_questions=4,
        )

        job.task_finished("q1", success=True, duration_seconds=1.0)
        assert job.percentage == 25.0

        job.task_finished("q2", success=True, duration_seconds=1.0)
        assert job.percentage == 50.0

        job.task_finished("q3", success=True, duration_seconds=1.0)
        assert job.percentage == 75.0

        job.task_finished("q4", success=True, duration_seconds=1.0)
        assert job.percentage == 100.0

    def test_concurrent_tasks_tracking(self):
        """Test tracking multiple concurrent in-progress tasks."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {}, "q2": {}, "q3": {}},
            config={"model_name": "test"},
            total_questions=3,
        )

        # Start 3 tasks concurrently
        job.task_started("q1")
        job.task_started("q2")
        job.task_started("q3")

        assert len(job.in_progress_questions) == 3
        assert "q1" in job.in_progress_questions
        assert "q2" in job.in_progress_questions
        assert "q3" in job.in_progress_questions

        # Finish one task
        job.task_finished("q1", success=True, duration_seconds=1.0)
        assert len(job.in_progress_questions) == 2
        assert "q1" not in job.in_progress_questions

    def test_to_dict_includes_new_fields(self):
        """Test that to_dict includes WebSocket streaming fields."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {}},
            config={"model_name": "test"},
            total_questions=1,
        )

        job.task_started("q1")
        job_dict = job.to_dict()

        assert "in_progress_questions" in job_dict
        assert "ema_seconds_per_item" in job_dict
        assert job_dict["in_progress_questions"] == ["q1"]
        assert job_dict["ema_seconds_per_item"] == 0.0


class TestProgressBroadcaster:
    """Test cases for ProgressBroadcaster WebSocket management."""

    @pytest.mark.anyio
    async def test_subscribe_adds_websocket(self):
        """Test that subscribe adds a WebSocket to the registry."""
        from karenina_server.services.generation_service import ProgressBroadcaster

        broadcaster = ProgressBroadcaster()
        mock_ws = MagicMock()

        await broadcaster.subscribe("job-123", mock_ws)

        assert "job-123" in broadcaster.subscribers
        assert mock_ws in broadcaster.subscribers["job-123"]

    @pytest.mark.anyio
    async def test_unsubscribe_removes_websocket(self):
        """Test that unsubscribe removes a WebSocket from the registry."""
        from karenina_server.services.generation_service import ProgressBroadcaster

        broadcaster = ProgressBroadcaster()
        mock_ws = MagicMock()

        await broadcaster.subscribe("job-123", mock_ws)
        await broadcaster.unsubscribe("job-123", mock_ws)

        assert "job-123" not in broadcaster.subscribers

    @pytest.mark.anyio
    async def test_broadcast_sends_to_all_subscribers(self):
        """Test that broadcast sends events to all subscribers."""
        from unittest.mock import AsyncMock

        from karenina_server.services.generation_service import ProgressBroadcaster

        broadcaster = ProgressBroadcaster()
        mock_ws1 = MagicMock()
        mock_ws2 = MagicMock()
        mock_ws1.send_json = AsyncMock()
        mock_ws2.send_json = AsyncMock()

        await broadcaster.subscribe("job-123", mock_ws1)
        await broadcaster.subscribe("job-123", mock_ws2)

        event_data = {"type": "test", "message": "hello"}
        await broadcaster.broadcast("job-123", event_data)

        mock_ws1.send_json.assert_called_once_with(event_data)
        mock_ws2.send_json.assert_called_once_with(event_data)

    @pytest.mark.anyio
    async def test_broadcast_removes_dead_connections(self):
        """Test that broadcast removes WebSockets that fail to send."""
        from unittest.mock import AsyncMock

        from karenina_server.services.generation_service import ProgressBroadcaster

        broadcaster = ProgressBroadcaster()
        mock_ws_good = MagicMock()
        mock_ws_dead = MagicMock()

        # Use AsyncMock for async methods
        mock_ws_good.send_json = AsyncMock()
        mock_ws_dead.send_json = AsyncMock(side_effect=Exception("Connection closed"))

        await broadcaster.subscribe("job-123", mock_ws_good)
        await broadcaster.subscribe("job-123", mock_ws_dead)

        event_data = {"type": "test"}
        await broadcaster.broadcast("job-123", event_data)

        # Good WebSocket should still be subscribed
        assert "job-123" in broadcaster.subscribers
        assert mock_ws_good in broadcaster.subscribers["job-123"]
        # Dead WebSocket should be removed
        assert mock_ws_dead not in broadcaster.subscribers["job-123"]

    @pytest.mark.anyio
    async def test_cleanup_job_closes_all_connections(self):
        """Test that cleanup_job closes all WebSockets for a job."""
        from unittest.mock import AsyncMock

        from karenina_server.services.generation_service import ProgressBroadcaster

        broadcaster = ProgressBroadcaster()
        mock_ws1 = MagicMock()
        mock_ws2 = MagicMock()
        mock_ws1.close = AsyncMock()
        mock_ws2.close = AsyncMock()

        await broadcaster.subscribe("job-123", mock_ws1)
        await broadcaster.subscribe("job-123", mock_ws2)

        await broadcaster.cleanup_job("job-123")

        mock_ws1.close.assert_called_once()
        mock_ws2.close.assert_called_once()
        assert "job-123" not in broadcaster.subscribers


class TestGenerationServiceIntegration:
    """Integration tests for GenerationService with progress tracking."""

    def test_get_progress_includes_new_fields(self):
        """Test that get_progress returns the new WebSocket fields."""
        service = GenerationService()

        # Create a job
        job_id = service.start_generation(
            questions_data={"q1": {"question": "Test?", "raw_answer": "Answer"}},
            config={"model_name": "test", "model_provider": "test", "temperature": 0.1, "interface": "langchain"},
        )

        # Get progress
        progress = service.get_progress(job_id)

        assert progress is not None
        assert "in_progress_questions" in progress
        assert "ema_seconds_per_item" in progress
        assert isinstance(progress["in_progress_questions"], list)
        assert isinstance(progress["ema_seconds_per_item"], float)
