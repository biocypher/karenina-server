"""Integration tests for template builder API endpoints.

Uses the ``client`` fixture from the integration conftest.py, which
provides a FastAPI TestClient backed by the shared ``app`` fixture.
"""

import pytest

VERIFIED_TEMPLATE = """
from karenina.schemas.entities import BaseAnswer, VerifiedField, ExactMatch, BooleanMatch

class Answer(BaseAnswer):
    target: str = VerifiedField(
        description="Protein target",
        ground_truth="BCL2",
        verify_with=ExactMatch(normalize=["lowercase", "strip"]),
    )
    is_approved: bool = VerifiedField(
        description="FDA approved",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
"""

CLASSIC_TEMPLATE = """
from pydantic import Field
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    target: str = Field(description="target")

    def ground_truth(self):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.lower() == self.correct["target"].lower()
"""


@pytest.mark.integration
@pytest.mark.api
class TestTemplateParse:
    def test_parse_verified_template(self, client):
        response = client.post(
            "/api/v2/templates/builder/parse",
            json={"code": VERIFIED_TEMPLATE},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["mode"] == "verified"
        assert data["spec"] is not None
        assert len(data["spec"]["fields"]) == 2

    def test_parse_classic_template(self, client):
        response = client.post(
            "/api/v2/templates/builder/parse",
            json={"code": CLASSIC_TEMPLATE},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "classic"
        assert data["spec"] is None

    def test_parse_invalid_code(self, client):
        response = client.post(
            "/api/v2/templates/builder/parse",
            json={"code": "this is not python"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] is not None


@pytest.mark.integration
@pytest.mark.api
class TestTemplateGenerate:
    def test_generate_from_spec(self, client):
        spec = {
            "fields": [
                {
                    "name": "target",
                    "type": "str",
                    "description": "Protein target",
                    "ground_truth": "BCL2",
                    "verify_with": {"type": "ExactMatch"},
                },
            ],
        }
        response = client.post(
            "/api/v2/templates/builder/generate",
            json={"spec": spec},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "VerifiedField(" in data["code"]

    def test_generate_invalid_spec(self, client):
        response = client.post(
            "/api/v2/templates/builder/generate",
            json={"spec": {"fields": "invalid"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


@pytest.mark.integration
@pytest.mark.api
class TestTemplateValidate:
    def test_validate_valid_template(self, client):
        response = client.post(
            "/api/v2/templates/builder/validate",
            json={"code": VERIFIED_TEMPLATE},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["ground_truth_check"] is True
        assert data["verify_check"] is True

    def test_validate_syntax_error(self, client):
        response = client.post(
            "/api/v2/templates/builder/validate",
            json={"code": "class Answer(BaseAnswer): def"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0


@pytest.mark.integration
@pytest.mark.api
class TestPrimitivesList:
    def test_list_primitives(self, client):
        response = client.get("/api/v2/templates/builder/primitives")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        names = [p["name"] for p in data["primitives"]]
        assert "ExactMatch" in names
        assert "BooleanMatch" in names
        assert "TraceRegex" in names
