"""E2E tests for rubric database save/load roundtrip.

Tests the complete flow of saving and loading rubrics through the database,
verifying that both global and question-specific rubrics survive the roundtrip.
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest
from karenina.benchmark import Benchmark
from karenina.schemas.domain import LLMRubricTrait, Question, RegexTrait, Rubric
from karenina.storage import DBConfig, init_database, load_benchmark, save_benchmark


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for E2E tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def db_config(temp_db_path):
    """Create database config and initialize database."""
    config = DBConfig(storage_url=f"sqlite:///{temp_db_path}")
    init_database(config)
    return config


@pytest.fixture
def sample_questions():
    """Create sample questions for testing."""
    return [
        Question(question="What is 2+2?", raw_answer="The answer is 4."),
        Question(question="What is the capital of France?", raw_answer="Paris is the capital of France."),
        Question(question="Is water wet?", raw_answer="Yes, water is wet."),
    ]


@pytest.fixture
def sample_global_rubric():
    """Create a sample global rubric with multiple trait types."""
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="accuracy",
                description="Is the response factually accurate?",
                kind="boolean",
            ),
            LLMRubricTrait(
                name="completeness",
                description="Rate the completeness of the response",
                kind="score",
                min_score=1,
                max_score=5,
            ),
        ],
        regex_traits=[
            RegexTrait(
                name="no_profanity",
                description="Response should not contain profanity",
                pattern=r"\b(damn|hell)\b",
                case_sensitive=False,
                invert_result=True,
            ),
        ],
        callable_traits=[],
        metric_traits=[],
    )


@pytest.fixture
def sample_question_rubric():
    """Create a sample question-specific rubric."""
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="test_question_trait",
                description="A test trait for this specific question",
                kind="boolean",
            )
        ],
        regex_traits=[],
        callable_traits=[],
        metric_traits=[],
    )


@pytest.mark.e2e
class TestGlobalRubricRoundtrip:
    """E2E tests for global rubric save/load cycle."""

    def test_global_rubric_survives_roundtrip(self, db_config, sample_questions, sample_global_rubric):
        """Test that global rubric survives save/load cycle."""
        # Create benchmark
        benchmark = Benchmark.create(
            name="test_global_rubric_benchmark",
            description="Test benchmark for global rubric roundtrip",
            version="1.0.0",
            creator="E2E Test",
        )

        # Add questions
        for q in sample_questions:
            benchmark.add_question(question=q, answer_template="# Template placeholder", finished=True)

        # Set global rubric
        benchmark.set_global_rubric(sample_global_rubric)

        # Save to database
        save_benchmark(benchmark, db_config)

        # Load benchmark back
        loaded_benchmark = load_benchmark("test_global_rubric_benchmark", db_config)

        # Verify global rubric
        loaded_rubric = loaded_benchmark.get_global_rubric()
        assert loaded_rubric is not None

        # Verify LLM traits
        assert len(loaded_rubric.llm_traits) == len(sample_global_rubric.llm_traits)
        assert loaded_rubric.llm_traits[0].name == "accuracy"
        assert loaded_rubric.llm_traits[0].kind == "boolean"
        assert loaded_rubric.llm_traits[1].name == "completeness"
        assert loaded_rubric.llm_traits[1].kind == "score"
        assert loaded_rubric.llm_traits[1].min_score == 1
        assert loaded_rubric.llm_traits[1].max_score == 5

        # Verify regex traits
        assert len(loaded_rubric.regex_traits) == len(sample_global_rubric.regex_traits)
        assert loaded_rubric.regex_traits[0].name == "no_profanity"
        assert loaded_rubric.regex_traits[0].invert_result is True

    def test_global_rubric_stored_in_metadata(self, db_config, temp_db_path, sample_questions, sample_global_rubric):
        """Test that global rubric is properly stored in metadata_json."""
        # Create and save benchmark
        benchmark = Benchmark.create(
            name="test_metadata_storage",
            description="Test",
            version="1.0.0",
            creator="E2E Test",
        )
        for q in sample_questions:
            benchmark.add_question(question=q, answer_template="# Template", finished=True)
        benchmark.set_global_rubric(sample_global_rubric)
        save_benchmark(benchmark, db_config)

        # Verify with raw SQL
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT metadata_json FROM benchmarks WHERE name = ?", ("test_metadata_storage",))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        metadata = json.loads(row[0])
        assert "global_rubric" in metadata

        gr = metadata["global_rubric"]
        assert "traits" in gr or "llm_traits" in gr  # Supports both formats


@pytest.mark.e2e
class TestQuestionRubricRoundtrip:
    """E2E tests for question-specific rubric save/load cycle."""

    def test_question_rubric_survives_roundtrip(self, db_config, sample_questions, sample_question_rubric):
        """Test that question-specific rubric survives save/load cycle."""
        # Create benchmark
        benchmark = Benchmark.create(
            name="test_question_rubric_benchmark",
            description="Test benchmark for question rubric roundtrip",
            version="1.0.0",
            creator="E2E Test",
        )

        # Add questions
        for q in sample_questions:
            benchmark.add_question(question=q, answer_template="# Template", finished=True)

        # Set question-specific rubric on first question
        question_ids = benchmark.get_question_ids()
        first_q_id = question_ids[0]
        benchmark.set_question_rubric(first_q_id, sample_question_rubric)

        # Save to database
        save_benchmark(benchmark, db_config)

        # Load benchmark back
        loaded_benchmark = load_benchmark("test_question_rubric_benchmark", db_config)

        # Verify question rubric
        loaded_q_ids = loaded_benchmark.get_question_ids()
        first_loaded_q_id = loaded_q_ids[0]
        q_data = loaded_benchmark.get_question(first_loaded_q_id)

        loaded_q_rubric = q_data.get("question_rubric")
        assert loaded_q_rubric is not None

        # Rubric may be stored as dict or Rubric object
        if isinstance(loaded_q_rubric, dict):
            llm_traits = loaded_q_rubric.get("llm_traits", [])
            assert len(llm_traits) == 1
            # Traits may be dicts or objects
            trait = llm_traits[0]
            if isinstance(trait, dict):
                assert trait["name"] == "test_question_trait"
            else:
                assert trait.name == "test_question_trait"
        elif hasattr(loaded_q_rubric, "llm_traits"):
            assert len(loaded_q_rubric.llm_traits) == 1
            assert loaded_q_rubric.llm_traits[0].name == "test_question_trait"


@pytest.mark.e2e
class TestCombinedRubricRoundtrip:
    """E2E tests for combined global and question rubric scenarios."""

    def test_both_rubrics_survive_roundtrip(
        self, db_config, sample_questions, sample_global_rubric, sample_question_rubric
    ):
        """Test that both global and question rubrics survive roundtrip."""
        # Create benchmark
        benchmark = Benchmark.create(
            name="test_combined_rubric_benchmark",
            description="Test benchmark for combined rubric roundtrip",
            version="1.0.0",
            creator="E2E Test",
        )

        # Add questions
        for q in sample_questions:
            benchmark.add_question(question=q, answer_template="# Template", finished=True)

        # Set global rubric
        benchmark.set_global_rubric(sample_global_rubric)

        # Set question-specific rubric on first question
        question_ids = benchmark.get_question_ids()
        first_q_id = question_ids[0]
        benchmark.set_question_rubric(first_q_id, sample_question_rubric)

        # Save to database
        save_benchmark(benchmark, db_config)

        # Load benchmark back
        loaded_benchmark = load_benchmark("test_combined_rubric_benchmark", db_config)

        # Verify global rubric
        loaded_global = loaded_benchmark.get_global_rubric()
        assert loaded_global is not None
        assert len(loaded_global.llm_traits) == 2
        assert len(loaded_global.regex_traits) == 1

        # Verify question rubric
        loaded_q_ids = loaded_benchmark.get_question_ids()
        first_loaded_q_id = loaded_q_ids[0]
        q_data = loaded_benchmark.get_question(first_loaded_q_id)
        loaded_q_rubric = q_data.get("question_rubric")
        assert loaded_q_rubric is not None

    def test_questions_without_rubric_have_none(self, db_config, sample_questions, sample_question_rubric):
        """Test that questions without rubric have None after roundtrip."""
        # Create benchmark
        benchmark = Benchmark.create(
            name="test_partial_rubric_benchmark",
            description="Test",
            version="1.0.0",
            creator="E2E Test",
        )

        # Add questions
        for q in sample_questions:
            benchmark.add_question(question=q, answer_template="# Template", finished=True)

        # Only set question rubric on first question
        question_ids = benchmark.get_question_ids()
        benchmark.set_question_rubric(question_ids[0], sample_question_rubric)

        # Save and load
        save_benchmark(benchmark, db_config)
        loaded_benchmark = load_benchmark("test_partial_rubric_benchmark", db_config)

        # First question should have rubric
        q1_data = loaded_benchmark.get_question(loaded_benchmark.get_question_ids()[0])
        assert q1_data.get("question_rubric") is not None

        # Second question should NOT have rubric
        q2_data = loaded_benchmark.get_question(loaded_benchmark.get_question_ids()[1])
        assert q2_data.get("question_rubric") is None


@pytest.mark.e2e
class TestRubricSerialization:
    """E2E tests for rubric API serialization format."""

    def test_rubric_serializes_for_api(self, db_config, sample_questions, sample_global_rubric):
        """Test that loaded rubric serializes correctly for API responses."""
        # Create, save, and load benchmark
        benchmark = Benchmark.create(
            name="test_api_serialization",
            description="Test",
            version="1.0.0",
            creator="E2E Test",
        )
        for q in sample_questions:
            benchmark.add_question(question=q, answer_template="# Template", finished=True)
        benchmark.set_global_rubric(sample_global_rubric)
        save_benchmark(benchmark, db_config)

        loaded_benchmark = load_benchmark("test_api_serialization", db_config)
        loaded_rubric = loaded_benchmark.get_global_rubric()

        # Serialize to API format (same as _serialize_rubric_to_dict in handlers)
        serialized = {
            "llm_traits": [t.model_dump() for t in loaded_rubric.llm_traits],
            "regex_traits": [t.model_dump() for t in loaded_rubric.regex_traits],
            "callable_traits": [t.model_dump() for t in loaded_rubric.callable_traits],
            "metric_traits": [t.model_dump() for t in loaded_rubric.metric_traits],
        }

        # Verify format
        assert "llm_traits" in serialized
        assert "regex_traits" in serialized
        assert len(serialized["llm_traits"]) == 2
        assert len(serialized["regex_traits"]) == 1

        # Verify trait structure
        accuracy_trait = serialized["llm_traits"][0]
        assert "name" in accuracy_trait
        assert "description" in accuracy_trait
        assert "kind" in accuracy_trait
        assert accuracy_trait["name"] == "accuracy"
        assert accuracy_trait["kind"] == "boolean"
