"""Integration tests for template generation service.

Tests multi-threaded job execution and WebSocket broadcasting.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.integration
@pytest.mark.service
class TestGenerationServiceIntegration:
    """Integration tests for GenerationService with threading."""

    @patch("karenina_server.services.generation_service.generation_service")
    def test_start_generation_returns_job_id(self, mock_service):
        """Test that starting generation returns a valid job ID."""
        mock_service.start_generation.return_value = "test-job-id-123"

        questions_data = {
            "q1": {"question": "Test question?", "raw_answer": "Test answer"},
        }
        config = MagicMock()
        config.model_provider = "google_genai"
        config.model_name = "gemini-2.0-flash"

        job_id = mock_service.start_generation(questions_data=questions_data, config=config)

        assert job_id == "test-job-id-123"

    @patch("karenina_server.services.generation_service.generation_service")
    def test_get_progress_returns_status(self, mock_service):
        """Test that get_progress returns correct status."""
        mock_service.get_progress.return_value = {
            "job_id": "test-job-123",
            "status": "running",
            "total_questions": 5,
            "processed_count": 2,
            "percentage": 40.0,
        }

        progress = mock_service.get_progress("test-job-123")

        assert progress["status"] == "running"
        assert progress["percentage"] == 40.0

    @patch("karenina_server.services.generation_service.generation_service")
    def test_cancel_job(self, mock_service):
        """Test that cancel_job works correctly."""
        mock_service.cancel_job.return_value = True

        result = mock_service.cancel_job("test-job-123")

        assert result is True


@pytest.mark.integration
@pytest.mark.service
class TestProgressBroadcasterIntegration:
    """Integration tests for ProgressBroadcaster WebSocket functionality."""

    @pytest.mark.anyio
    async def test_multiple_subscribers(self):
        """Test broadcasting to multiple subscribers."""
        from karenina_server.services.progress_broadcaster import ProgressBroadcaster

        broadcaster = ProgressBroadcaster()

        mock_ws1 = MagicMock()
        mock_ws2 = MagicMock()
        mock_ws3 = MagicMock()

        mock_ws1.send_json = AsyncMock()
        mock_ws2.send_json = AsyncMock()
        mock_ws3.send_json = AsyncMock()

        await broadcaster.subscribe("job-123", mock_ws1)
        await broadcaster.subscribe("job-123", mock_ws2)
        await broadcaster.subscribe("job-456", mock_ws3)

        event = {"type": "progress", "percentage": 50}
        await broadcaster.broadcast("job-123", event)

        mock_ws1.send_json.assert_called_once_with(event)
        mock_ws2.send_json.assert_called_once_with(event)
        mock_ws3.send_json.assert_not_called()

    @pytest.mark.anyio
    async def test_unsubscribe_removes_websocket(self):
        """Test that unsubscribe properly removes WebSocket."""
        from karenina_server.services.progress_broadcaster import ProgressBroadcaster

        broadcaster = ProgressBroadcaster()

        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()

        await broadcaster.subscribe("job-123", mock_ws)
        assert mock_ws in broadcaster.subscribers["job-123"]

        await broadcaster.unsubscribe("job-123", mock_ws)
        assert "job-123" not in broadcaster.subscribers

    @pytest.mark.anyio
    async def test_cleanup_closes_all_connections(self):
        """Test that cleanup_job closes all WebSocket connections."""
        from karenina_server.services.progress_broadcaster import ProgressBroadcaster

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
