"""
Test suite for summary statistics endpoints.

Tests:
- POST /api/verification/summary
- POST /api/verification/compare-models
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from karenina_server.server import create_fastapi_app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    webapp_dir = Path(__file__).parent.parent.parent / "karenina-gui"
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
                "answering_model": "gpt-4o-mini",
                "parsing_model": "gpt-4o-mini",
                "answering_replicate": 1,
                "parsing_replicate": 1,
                "completed_without_errors": True,
                "error": None,
                "execution_time": 1.5,
                "timestamp": "2024-01-01T12:00:00Z",
                "run_name": "test-run-1",
                "mcp_config": "[]",
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
                "answering_model": "gpt-4o-mini",
                "parsing_model": "gpt-4o-mini",
                "answering_replicate": 1,
                "parsing_replicate": 1,
                "completed_without_errors": True,
                "error": None,
                "execution_time": 1.2,
                "timestamp": "2024-01-01T12:01:00Z",
                "run_name": "test-run-1",
                "mcp_config": "[]",
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
                "answering_model": "claude-3-5-sonnet-20241022",
                "parsing_model": "gpt-4o-mini",
                "answering_replicate": 1,
                "parsing_replicate": 1,
                "completed_without_errors": True,
                "error": None,
                "execution_time": 2.1,
                "timestamp": "2024-01-01T12:02:00Z",
                "run_name": "test-run-1",
                "mcp_config": "[]",
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


def test_summary_endpoint_all_results(client: TestClient, sample_results: dict[str, dict[str, object]]) -> None:
    """Test summary endpoint with all results."""
    response = client.post(
        "/api/verification/summary",
        json={"results": sample_results, "run_name": None},
    )

    assert response.status_code == 200
    data = response.json()

    # Check basic counts
    assert data["num_results"] == 3
    assert data["num_completed"] == 3
    assert data["num_questions"] == 2  # q1 and q2
    assert data["num_models"] == 2  # gpt-4o-mini and claude-3-5-sonnet
    assert data["num_with_template"] == 3

    # Check token usage
    assert data["tokens"]["total_input"] == 370  # 100 + 120 + 150
    assert data["tokens"]["total_output"] == 150  # 50 + 40 + 60

    # Check template pass rates
    assert data["template_pass_overall"]["total"] == 3
    assert data["template_pass_overall"]["passed"] == 2
    assert data["template_pass_overall"]["pass_pct"] == pytest.approx(66.67, rel=0.01)


def test_summary_endpoint_filtered_by_run_name(
    client: TestClient, sample_results: dict[str, dict[str, object]]
) -> None:
    """Test summary endpoint with run_name filter."""
    response = client.post(
        "/api/verification/summary",
        json={"results": sample_results, "run_name": "test-run-1"},
    )

    assert response.status_code == 200
    data = response.json()

    # Should get all 3 results since they all have run_name="test-run-1"
    assert data["num_results"] == 3


def test_summary_endpoint_no_results(client: TestClient) -> None:
    """Test summary endpoint with empty results."""
    response = client.post(
        "/api/verification/summary",
        json={"results": {}, "run_name": None},
    )

    assert response.status_code == 400
    assert "No valid results found" in response.json()["detail"]


def test_compare_models_endpoint(client: TestClient, sample_results: dict[str, dict[str, object]]) -> None:
    """Test model comparison endpoint."""
    response = client.post(
        "/api/verification/compare-models",
        json={
            "results": sample_results,
            "models": [
                {"answering_model": "gpt-4o-mini", "mcp_config": "[]"},
                {"answering_model": "claude-3-5-sonnet-20241022", "mcp_config": "[]"},
            ],
            "parsing_model": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Check model summaries
    assert "model_summaries" in data
    assert len(data["model_summaries"]) == 2
    assert "gpt-4o-mini|[]" in data["model_summaries"]
    assert "claude-3-5-sonnet-20241022|[]" in data["model_summaries"]

    # Check gpt-4o-mini summary
    gpt_summary = data["model_summaries"]["gpt-4o-mini|[]"]
    assert gpt_summary["num_results"] == 2
    assert gpt_summary["template_pass_overall"]["passed"] == 1  # Only q1 passed

    # Check claude summary
    claude_summary = data["model_summaries"]["claude-3-5-sonnet-20241022|[]"]
    assert claude_summary["num_results"] == 1
    assert claude_summary["template_pass_overall"]["passed"] == 1

    # Check heatmap data
    assert "heatmap_data" in data
    assert len(data["heatmap_data"]) == 2  # 2 unique questions

    # Find q1 in heatmap
    q1_row = next((row for row in data["heatmap_data"] if row["question_id"] == "q1"), None)
    assert q1_row is not None
    assert "gpt-4o-mini|[]" in q1_row["results_by_model"]
    assert "claude-3-5-sonnet-20241022|[]" in q1_row["results_by_model"]
    assert q1_row["results_by_model"]["gpt-4o-mini|[]"]["passed"] is True
    assert q1_row["results_by_model"]["claude-3-5-sonnet-20241022|[]"]["passed"] is True


def test_compare_models_no_models(client: TestClient, sample_results: dict[str, dict[str, object]]) -> None:
    """Test model comparison with no models specified."""
    response = client.post(
        "/api/verification/compare-models",
        json={
            "results": sample_results,
            "models": [],
            "parsing_model": "gpt-4o-mini",
        },
    )

    assert response.status_code == 400
    assert "At least one model must be specified" in response.json()["detail"]


def test_compare_models_no_results(client: TestClient) -> None:
    """Test model comparison with empty results."""
    response = client.post(
        "/api/verification/compare-models",
        json={
            "results": {},
            "models": [{"answering_model": "gpt-4o-mini", "mcp_config": "[]"}],
            "parsing_model": "gpt-4o-mini",
        },
    )

    assert response.status_code == 400
    assert "No valid results found" in response.json()["detail"]
