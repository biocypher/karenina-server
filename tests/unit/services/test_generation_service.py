"""Unit tests for template generation service progress tracking.

These are pure logic tests with no I/O, no TestClient, no external calls.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from karenina_server.services.generation_service import TemplateGenerationJob
from karenina_server.services.progress_broadcaster import ProgressBroadcaster


@pytest.mark.unit
@pytest.mark.service
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
        job.task_finished("q1", success=True)

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

        job.task_started("q1")
        job.task_finished("q1", success=True)
        job.task_started("q2")
        job.task_finished("q2", success=False)

        assert job.processed_count == 2
        assert job.successful_count == 1
        assert job.failed_count == 1

    def test_last_task_duration_tracking(self):
        """Test that last task duration is tracked correctly."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {"question": "Test?", "raw_answer": "Answer"}},
            config={"model_name": "test"},
            total_questions=1,
        )

        job.task_started("q1")
        time.sleep(0.1)  # Sleep to make duration measurable
        job.task_finished("q1", success=True)

        # Duration should be at least 0.1 seconds
        assert job.last_task_duration is not None
        assert job.last_task_duration >= 0.1

    def test_percentage_calculation(self):
        """Test that percentage is calculated correctly."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {}, "q2": {}, "q3": {}, "q4": {}},
            config={"model_name": "test"},
            total_questions=4,
        )

        job.task_started("q1")
        job.task_finished("q1", success=True)
        assert job.percentage == 25.0

        job.task_started("q2")
        job.task_finished("q2", success=True)
        assert job.percentage == 50.0

        job.task_started("q3")
        job.task_finished("q3", success=True)
        assert job.percentage == 75.0

        job.task_started("q4")
        job.task_finished("q4", success=True)
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
        job.task_finished("q1", success=True)
        assert len(job.in_progress_questions) == 2
        assert "q1" not in job.in_progress_questions

    def test_to_dict_includes_progress_fields(self):
        """Test that to_dict includes progress tracking fields."""
        job = TemplateGenerationJob(
            job_id="test-job",
            questions_data={"q1": {}},
            config={"model_name": "test"},
            total_questions=1,
        )

        job.task_started("q1")
        job_dict = job.to_dict()

        assert "in_progress_questions" in job_dict
        assert "duration_seconds" in job_dict
        assert "last_task_duration" in job_dict
        assert job_dict["in_progress_questions"] == ["q1"]
        assert job_dict["last_task_duration"] is None  # No task finished yet


@pytest.mark.unit
@pytest.mark.service
class TestProgressBroadcaster:
    """Test cases for ProgressBroadcaster WebSocket management."""

    @pytest.mark.anyio
    async def test_subscribe_adds_websocket(self):
        """Test that subscribe adds a WebSocket to the registry."""
        broadcaster = ProgressBroadcaster()
        mock_ws = MagicMock()

        await broadcaster.subscribe("job-123", mock_ws)

        assert "job-123" in broadcaster.subscribers
        assert mock_ws in broadcaster.subscribers["job-123"]

    @pytest.mark.anyio
    async def test_unsubscribe_removes_websocket(self):
        """Test that unsubscribe removes a WebSocket from the registry."""
        broadcaster = ProgressBroadcaster()
        mock_ws = MagicMock()

        await broadcaster.subscribe("job-123", mock_ws)
        await broadcaster.unsubscribe("job-123", mock_ws)

        assert "job-123" not in broadcaster.subscribers

    @pytest.mark.anyio
    async def test_broadcast_sends_to_all_subscribers(self):
        """Test that broadcast sends events to all subscribers."""
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
