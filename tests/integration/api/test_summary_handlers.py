"""Integration tests for summary statistics API handlers.

Tests:
- POST /api/v2/verifications/summary
- POST /api/v2/verifications/compare
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from karenina_server.server import create_fastapi_app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    webapp_dir = Path(__file__).parent.parent.parent.parent.parent / "karenina-gui"
    app = create_fastapi_app(webapp_dir)
    return TestClient(app)


@pytest.fixture
def sample_results() -> dict[str, dict[str, object]]:
    """Create sample verification results."""
    return {
        "result-1": {
            "metadata": {
                "question_id": "q1",
                "template_id": "template-1",
                "question_text": "What is 2+2?",
                "keywords": ["math"],
                "answering": {"interface": "langchain", "model_name": "claude-haiku-4-5", "tools": []},
                "parsing": {"interface": "langchain", "model_name": "claude-haiku-4-5", "tools": []},
                "replicate": 1,
                "completed_without_errors": True,
                "error": None,
                "execution_time": 1.5,
                "timestamp": "2024-01-01T12:00:00Z",
                "run_name": "test-run-1",
                "result_id": "result00000001",
            },
            "template": {
                "raw_llm_response": "The answer is 4",
                "parsed_gt_response": {"value": "4"},
                "parsed_llm_response": {"value": "4"},
                "template_verification_performed": True,
                "verify_result": True,
                "abstained": False,
                "usage_metadata": {
                    "total": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                    }
                },
            },
            "rubric": None,
            "deep_judgment": None,
            "deep_judgment_rubric": None,
        },
        "result-2": {
            "metadata": {
                "question_id": "q2",
                "template_id": "template-2",
                "question_text": "What is the capital of France?",
                "keywords": ["geography"],
                "answering": {"interface": "langchain", "model_name": "claude-haiku-4-5", "tools": []},
                "parsing": {"interface": "langchain", "model_name": "claude-haiku-4-5", "tools": []},
                "replicate": 1,
                "completed_without_errors": True,
                "error": None,
                "execution_time": 1.2,
                "timestamp": "2024-01-01T12:01:00Z",
                "run_name": "test-run-1",
                "result_id": "result00000002",
            },
            "template": {
                "raw_llm_response": "London",
                "parsed_gt_response": {"value": "Paris"},
                "parsed_llm_response": {"value": "London"},
                "template_verification_performed": True,
                "verify_result": False,
                "abstained": False,
                "usage_metadata": {
                    "total": {
                        "input_tokens": 120,
                        "output_tokens": 40,
                    }
                },
            },
            "rubric": None,
            "deep_judgment": None,
            "deep_judgment_rubric": None,
        },
        "result-3": {
            "metadata": {
                "question_id": "q1",
                "template_id": "template-1",
                "question_text": "What is 2+2?",
                "keywords": ["math"],
                "answering": {"interface": "langchain", "model_name": "claude-sonnet-4", "tools": []},
                "parsing": {"interface": "langchain", "model_name": "claude-haiku-4-5", "tools": []},
                "replicate": 1,
                "completed_without_errors": True,
                "error": None,
                "execution_time": 2.1,
                "timestamp": "2024-01-01T12:02:00Z",
                "run_name": "test-run-1",
                "result_id": "result00000003",
            },
            "template": {
                "raw_llm_response": "Four",
                "parsed_gt_response": {"value": "4"},
                "parsed_llm_response": {"value": "4"},
                "template_verification_performed": True,
                "verify_result": True,
                "abstained": False,
                "usage_metadata": {
                    "total": {
                        "input_tokens": 150,
                        "output_tokens": 60,
                    }
                },
            },
            "rubric": None,
            "deep_judgment": None,
            "deep_judgment_rubric": None,
        },
    }


@pytest.mark.integration
@pytest.mark.api
class TestSummaryEndpoints:
    """Test summary statistics endpoints."""

    def test_summary_endpoint_all_results(self, client: TestClient, sample_results: dict[str, Any]) -> None:
        """Test summary endpoint with all results."""
        response = client.post(
            "/api/v2/verifications/summary",
            json={"results": sample_results, "run_name": None},
        )

        assert response.status_code == 200
        data = response.json()

        # Response follows envelope pattern with success and summary fields
        assert data["success"] is True
        summary = data["summary"]
        assert summary["num_results"] == 3
        assert summary["num_completed"] == 3
        assert summary["num_questions"] == 2
        assert summary["num_models"] == 2
        assert summary["tokens"]["total_input"] == 370
        assert summary["tokens"]["total_output"] == 150

    def test_summary_endpoint_filtered_by_run_name(self, client: TestClient, sample_results: dict[str, Any]) -> None:
        """Test summary endpoint with run_name filter."""
        response = client.post(
            "/api/v2/verifications/summary",
            json={"results": sample_results, "run_name": "test-run-1"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["summary"]["num_results"] == 3

    def test_summary_endpoint_no_results(self, client: TestClient) -> None:
        """Test summary endpoint with empty results."""
        response = client.post(
            "/api/v2/verifications/summary",
            json={"results": {}, "run_name": None},
        )

        assert response.status_code == 400
        assert "No valid results found" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.api
class TestCompareModelsEndpoints:
    """Test model comparison endpoints."""

    def test_compare_models_endpoint(self, client: TestClient, sample_results: dict[str, Any]) -> None:
        """Test model comparison endpoint."""
        response = client.post(
            "/api/v2/verifications/compare",
            json={
                "results": sample_results,
                "models": [
                    {"answering_model": "claude-haiku-4-5", "interface": "langchain"},
                    {"answering_model": "claude-sonnet-4", "interface": "langchain"},
                ],
                "parsing_model": "langchain:claude-haiku-4-5",
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Response follows envelope pattern with success field
        assert data["success"] is True
        assert "model_summaries" in data
        assert len(data["model_summaries"]) == 2

    def test_compare_models_no_models(self, client: TestClient, sample_results: dict[str, Any]) -> None:
        """Test model comparison with no models specified."""
        response = client.post(
            "/api/v2/verifications/compare",
            json={
                "results": sample_results,
                "models": [],
                "parsing_model": "langchain:gpt-4o-mini",
            },
        )

        assert response.status_code == 400
        assert "At least one model must be specified" in response.json()["detail"]

    def test_compare_models_no_results(self, client: TestClient) -> None:
        """Test model comparison with empty results."""
        response = client.post(
            "/api/v2/verifications/compare",
            json={
                "results": {},
                "models": [{"answering_model": "gpt-4o-mini", "interface": "langchain", "mcp_config": "[]"}],
                "parsing_model": "langchain:gpt-4o-mini",
            },
        )

        assert response.status_code == 400
        assert "No valid results found" in response.json()["detail"]
