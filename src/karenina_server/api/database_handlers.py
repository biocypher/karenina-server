"""Database management API handlers."""

import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException

try:
    from karenina.storage import DBConfig, get_benchmark_summary, init_database, load_benchmark, save_benchmark

    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False


def _extract_creator_name(creator: Any) -> str:
    """
    Extract creator name from either a string or Person dict.

    Args:
        creator: Either a string or a dict with '@type': 'Person' and 'name' key

    Returns:
        The creator name as a string, or 'Unknown' if unable to extract
    """
    if creator is None:
        return "Unknown"
    if isinstance(creator, str):
        return creator
    if isinstance(creator, dict) and creator.get("@type") == "Person":
        name = creator.get("name", "Unknown")
        return str(name)  # Ensure it's a string
    return "Unknown"


def register_database_routes(
    app: Any,
    DatabaseConnectRequest: Any,
    DatabaseConnectResponse: Any,
    BenchmarkListResponse: Any,
    BenchmarkLoadRequest: Any,
    BenchmarkLoadResponse: Any,
    BenchmarkCreateRequest: Any,
    BenchmarkCreateResponse: Any,
    BenchmarkSaveRequest: Any,
    BenchmarkSaveResponse: Any,
    DuplicateResolutionRequest: Any,
    DuplicateResolutionResponse: Any,
    ListDatabasesResponse: Any,
) -> None:
    """Register database management routes."""

    @app.post("/api/database/connect", response_model=DatabaseConnectResponse)  # type: ignore[misc]
    async def connect_database_endpoint(request: DatabaseConnectRequest) -> DatabaseConnectResponse:
        """Connect to or create a database."""
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            # Create database configuration
            db_config = DBConfig(storage_url=request.storage_url)

            # Initialize database if requested
            if request.create_if_missing:
                init_database(db_config)

            # Get database statistics to verify connection
            try:
                summaries = get_benchmark_summary(db_config)
                benchmark_count = len(summaries)
            except Exception:
                benchmark_count = 0

            return DatabaseConnectResponse(
                success=True,
                storage_url=request.storage_url,
                benchmark_count=benchmark_count,
                message=f"Successfully connected to database. Found {benchmark_count} benchmarks.",
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error connecting to database: {e!s}") from e

    @app.get("/api/database/benchmarks", response_model=BenchmarkListResponse)  # type: ignore[misc]
    async def list_benchmarks_endpoint(storage_url: str) -> BenchmarkListResponse:
        """List all benchmarks in the database."""
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            db_config = DBConfig(storage_url=storage_url)
            summaries = get_benchmark_summary(db_config)

            # Extract benchmark information
            benchmarks = [
                {
                    "id": str(summary["benchmark_id"]),
                    "name": summary["benchmark_name"],
                    "total_questions": summary["total_questions"],
                    "finished_count": summary.get("finished_count", 0),
                    "unfinished_count": summary.get("unfinished_count", 0),
                    "last_modified": summary.get("updated_at", summary.get("created_at", "")),
                }
                for summary in summaries
            ]

            return BenchmarkListResponse(success=True, benchmarks=benchmarks, count=len(benchmarks))

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing benchmarks: {e!s}") from e

    @app.post("/api/database/load-benchmark", response_model=BenchmarkLoadResponse)  # type: ignore[misc]
    async def load_benchmark_endpoint(request: BenchmarkLoadRequest) -> BenchmarkLoadResponse:
        """Load a benchmark from the database."""
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            from datetime import datetime

            from karenina.storage import BenchmarkModel, BenchmarkQuestionModel, get_session

            db_config = DBConfig(storage_url=request.storage_url)

            # Load benchmark
            benchmark, loaded_config = load_benchmark(request.benchmark_name, db_config, load_config=True)

            # Query database for updated_at timestamps
            updated_at_map = {}
            with get_session(db_config) as session:
                # Get benchmark ID
                from sqlalchemy import select

                benchmark_model = session.execute(
                    select(BenchmarkModel).where(BenchmarkModel.name == request.benchmark_name)
                ).scalar_one_or_none()

                if benchmark_model:
                    # Get all benchmark_questions for this benchmark
                    bq_models = (
                        session.execute(
                            select(BenchmarkQuestionModel).where(
                                BenchmarkQuestionModel.benchmark_id == benchmark_model.id
                            )
                        )
                        .scalars()
                        .all()
                    )

                    # Build map of question_id -> updated_at
                    for bq in bq_models:
                        if bq.updated_at:
                            updated_at_map[bq.question_id] = (
                                bq.updated_at.isoformat() if hasattr(bq.updated_at, "isoformat") else str(bq.updated_at)
                            )

            # Convert to checkpoint format (similar to FileManager handleCheckpointUpload)
            questions_data = {}
            for q_data in benchmark.get_all_questions():
                question_id = q_data["id"]
                questions_data[question_id] = {
                    "id": question_id,
                    "question": q_data["question"],
                    "raw_answer": q_data["raw_answer"],
                    "answer_template": q_data.get("answer_template"),
                    "finished": q_data.get("finished", False),
                    "keywords": q_data.get("keywords", []),  # Frontend CheckpointItem expects "keywords"
                    "last_modified": updated_at_map.get(question_id, datetime.now().isoformat()),
                }

            # Create checkpoint-like response
            # Convert created_at to string if it's a datetime object
            created_at_str = ""
            if hasattr(benchmark, "created_at") and benchmark.created_at:
                if hasattr(benchmark.created_at, "isoformat"):
                    created_at_str = benchmark.created_at.isoformat()
                else:
                    created_at_str = str(benchmark.created_at)

            checkpoint_data = {
                "dataset_metadata": {
                    "name": benchmark.name,
                    "description": benchmark.description or "",
                    "version": benchmark.version,
                    "creator": benchmark.creator or "",
                    "created_at": created_at_str,
                },
                "questions": questions_data,
                "global_rubric": benchmark.global_rubric if hasattr(benchmark, "global_rubric") else None,
            }

            return BenchmarkLoadResponse(
                success=True,
                benchmark_name=request.benchmark_name,
                checkpoint_data=checkpoint_data,
                storage_url=request.storage_url,
                message=f"Successfully loaded benchmark '{request.benchmark_name}' from database",
            )

        except ValueError as e:
            raise HTTPException(status_code=404, detail=f"Benchmark not found: {e!s}") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading benchmark: {e!s}") from e

    @app.post("/api/database/create-benchmark", response_model=BenchmarkCreateResponse)  # type: ignore[misc]
    async def create_benchmark_endpoint(request: BenchmarkCreateRequest) -> BenchmarkCreateResponse:
        """Create a new empty benchmark in the database."""
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            # Import Benchmark class here to avoid circular imports
            from karenina.benchmark.benchmark import Benchmark

            db_config = DBConfig(storage_url=request.storage_url)

            # Check if benchmark with this name already exists
            try:
                summaries = get_benchmark_summary(db_config, benchmark_name=request.name)
                if summaries:
                    raise HTTPException(status_code=400, detail=f"Benchmark '{request.name}' already exists")
            except Exception:
                # Database may not exist yet, which is fine
                pass

            # Create empty benchmark
            benchmark = Benchmark.create(
                name=request.name,
                description=request.description or "",
                version=request.version or "1.0.0",
                creator=request.creator or "Unknown",
            )

            # Save to database
            save_benchmark(benchmark, db_config)

            # Return in same format as load-benchmark
            checkpoint_data = {
                "dataset_metadata": {
                    "name": benchmark.name,
                    "description": benchmark.description,
                    "version": benchmark.version,
                    "creator": benchmark.creator,
                    "created_at": "",  # Will be set by database
                },
                "questions": {},  # Empty benchmark
                "global_rubric": None,
            }

            return BenchmarkCreateResponse(
                success=True,
                benchmark_name=request.name,
                checkpoint_data=checkpoint_data,
                storage_url=request.storage_url,
                message=f"Successfully created benchmark '{request.name}' in database",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating benchmark: {e!s}") from e

    @app.post("/api/database/save-benchmark", response_model=BenchmarkSaveResponse)  # type: ignore[misc]
    async def save_benchmark_endpoint(request: BenchmarkSaveRequest) -> BenchmarkSaveResponse:
        """Save current checkpoint data to the database."""
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            # Import required classes
            from karenina.benchmark.benchmark import Benchmark
            from karenina.schemas import LLMRubricTrait, Question, Rubric

            db_config = DBConfig(storage_url=request.storage_url)

            # Get dataset metadata from checkpoint
            metadata = request.checkpoint_data.get("dataset_metadata", {})

            # Create benchmark instance
            benchmark = Benchmark.create(
                name=request.benchmark_name,
                description=metadata.get("description", ""),
                version=metadata.get("version", "1.0.0"),
                creator=_extract_creator_name(metadata.get("creator")),
            )

            # Add questions from checkpoint
            questions_data = request.checkpoint_data.get("questions", {})
            for question_id, q_data in questions_data.items():
                # Create Question object
                question = Question(
                    question=q_data["question"],
                    raw_answer=q_data["raw_answer"],
                    tags=q_data.get("keywords", []),  # Frontend CheckpointItem sends "keywords"
                    few_shot_examples=q_data.get("few_shot_examples"),
                )

                # Add question to benchmark with the original question_id
                benchmark.add_question(
                    question=question,
                    answer_template=q_data.get("answer_template", ""),
                    question_id=question_id,
                    finished=q_data.get("finished", False),
                    few_shot_examples=q_data.get("few_shot_examples"),
                )

                # Add question-specific rubric if present
                if q_data.get("question_rubric"):
                    from karenina.schemas import CallableTrait, MetricRubricTrait, RegexTrait

                    rubric_data = q_data["question_rubric"]
                    traits = []
                    regex_traits = []
                    callable_traits = []
                    metric_traits = []

                    # Process LLM-based traits
                    for trait_data in rubric_data.get("traits", []):
                        kind = trait_data.get("kind", "score")
                        # Map frontend trait types to backend TraitKind
                        if kind in ("binary", "Binary"):
                            kind = "boolean"
                        elif kind in ("score", "Score"):
                            kind = "score"

                        trait = LLMRubricTrait(
                            name=trait_data["name"],
                            description=trait_data.get("description"),
                            kind=kind,
                            min_score=trait_data.get("min_score", 1) if kind == "score" else None,
                            max_score=trait_data.get("max_score", 5) if kind == "score" else None,
                        )
                        traits.append(trait)

                    # Process regex traits
                    for regex_trait_data in rubric_data.get("regex_traits", []):
                        regex_trait = RegexTrait(
                            name=regex_trait_data["name"],
                            description=regex_trait_data.get("description"),
                            pattern=regex_trait_data.get("pattern", ""),
                            case_sensitive=regex_trait_data.get("case_sensitive", True),
                            invert_result=regex_trait_data.get("invert_result", False),
                        )
                        regex_traits.append(regex_trait)

                    # Process callable traits
                    for callable_trait_data in rubric_data.get("callable_traits", []):
                        callable_trait = CallableTrait(
                            name=callable_trait_data["name"],
                            description=callable_trait_data.get("description"),
                            callable_code=callable_trait_data.get("callable_code", b""),
                            kind=callable_trait_data.get("kind", "boolean"),
                            min_score=callable_trait_data.get("min_score"),
                            max_score=callable_trait_data.get("max_score"),
                            invert_result=callable_trait_data.get("invert_result", False),
                        )
                        callable_traits.append(callable_trait)

                    # Process metric traits
                    for metric_trait_data in rubric_data.get("metric_traits", []):
                        metric_trait = MetricRubricTrait(
                            name=metric_trait_data["name"],
                            description=metric_trait_data.get("description"),
                            evaluation_mode=metric_trait_data.get("evaluation_mode", "tp_only"),
                            metrics=metric_trait_data.get("metrics", []),
                            tp_instructions=metric_trait_data.get("tp_instructions", []),
                            tn_instructions=metric_trait_data.get("tn_instructions", []),
                            repeated_extraction=metric_trait_data.get("repeated_extraction", True),
                        )
                        metric_traits.append(metric_trait)

                    if traits or regex_traits or callable_traits or metric_traits:
                        rubric = Rubric(
                            traits=traits,
                            regex_traits=regex_traits,
                            callable_traits=callable_traits,
                            metric_traits=metric_traits,
                        )
                        benchmark.set_question_rubric(question_id, rubric)

            # Add global rubric if present
            if request.checkpoint_data.get("global_rubric"):
                from karenina.schemas import CallableTrait, RegexTrait

                global_rubric_data = request.checkpoint_data["global_rubric"]
                traits = []
                regex_traits = []
                callable_traits = []

                # Process LLM-based traits
                for trait_data in global_rubric_data.get("traits", []):
                    kind = trait_data.get("kind", "score")
                    # Map frontend trait types to backend TraitKind
                    if kind in ("binary", "Binary"):
                        kind = "boolean"
                    elif kind in ("score", "Score"):
                        kind = "score"

                    trait = LLMRubricTrait(
                        name=trait_data["name"],
                        description=trait_data.get("description"),
                        kind=kind,
                        min_score=trait_data.get("min_score", 1) if kind == "score" else None,
                        max_score=trait_data.get("max_score", 5) if kind == "score" else None,
                    )
                    traits.append(trait)

                # Process regex traits
                for regex_trait_data in global_rubric_data.get("regex_traits", []):
                    regex_trait = RegexTrait(
                        name=regex_trait_data["name"],
                        description=regex_trait_data.get("description"),
                        pattern=regex_trait_data.get("pattern", ""),
                        case_sensitive=regex_trait_data.get("case_sensitive", True),
                        invert_result=regex_trait_data.get("invert_result", False),
                    )
                    regex_traits.append(regex_trait)

                # Process callable traits
                for callable_trait_data in global_rubric_data.get("callable_traits", []):
                    callable_trait = CallableTrait(
                        name=callable_trait_data["name"],
                        description=callable_trait_data.get("description"),
                        callable_code=callable_trait_data.get("callable_code", b""),
                        kind=callable_trait_data.get("kind", "boolean"),
                        min_score=callable_trait_data.get("min_score"),
                        max_score=callable_trait_data.get("max_score"),
                        invert_result=callable_trait_data.get("invert_result", False),
                    )
                    callable_traits.append(callable_trait)

                if traits or regex_traits or callable_traits:
                    benchmark.global_rubric = Rubric(
                        traits=traits, regex_traits=regex_traits, callable_traits=callable_traits
                    )

            # Save to database (will overwrite if exists, or detect duplicates if requested)
            result = save_benchmark(benchmark, db_config, detect_duplicates_only=request.detect_duplicates)
            # Handle conditional return type: tuple if detect_duplicates=True, Benchmark if False
            duplicates_found = result[1] if isinstance(result, tuple) else None

            # Get updated timestamp from database
            from datetime import UTC, datetime

            last_modified = datetime.now(UTC).isoformat()

            # If detecting duplicates and duplicates were found, return them
            if request.detect_duplicates and duplicates_found:
                return BenchmarkSaveResponse(
                    success=True,
                    message=f"Found {len(duplicates_found)} duplicate question(s)",
                    last_modified=None,
                    duplicates=duplicates_found,
                )

            # Normal save response
            return BenchmarkSaveResponse(
                success=True,
                message=f"Benchmark '{request.benchmark_name}' saved successfully",
                last_modified=last_modified,
                duplicates=None,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving benchmark: {e!s}") from e

    @app.post("/api/database/resolve-duplicates", response_model=DuplicateResolutionResponse)  # type: ignore[misc]
    async def resolve_duplicates_endpoint(request: DuplicateResolutionRequest) -> DuplicateResolutionResponse:
        """Resolve duplicate questions by applying user's choices (keep_old vs keep_new)."""
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            # Import required classes
            from karenina.benchmark.benchmark import Benchmark
            from karenina.schemas import LLMRubricTrait, Question, Rubric

            db_config = DBConfig(storage_url=request.storage_url)

            # Get dataset metadata from checkpoint
            metadata = request.checkpoint_data.get("dataset_metadata", {})

            # Create benchmark instance
            benchmark = Benchmark.create(
                name=request.benchmark_name,
                description=metadata.get("description", ""),
                version=metadata.get("version", "1.0.0"),
                creator=_extract_creator_name(metadata.get("creator")),
            )

            # Track resolution counts
            kept_old_count = 0
            kept_new_count = 0

            # Add questions from checkpoint based on user resolutions
            questions_data = request.checkpoint_data.get("questions", {})
            for question_id, q_data in questions_data.items():
                # Check resolution for this question
                resolution = request.resolutions.get(question_id, "keep_new")  # Default to keep_new

                if resolution == "keep_old":
                    # Skip this question - keeping the existing version in DB
                    kept_old_count += 1
                    continue
                elif resolution == "keep_new":
                    # Add this question - will update the existing version
                    kept_new_count += 1

                # Create Question object
                question = Question(
                    question=q_data["question"],
                    raw_answer=q_data["raw_answer"],
                    tags=q_data.get("keywords", []),  # Frontend CheckpointItem sends "keywords"
                    few_shot_examples=q_data.get("few_shot_examples"),
                )

                # Add question to benchmark with the original question_id
                benchmark.add_question(
                    question=question,
                    answer_template=q_data.get("answer_template", ""),
                    question_id=question_id,
                    finished=q_data.get("finished", False),
                    few_shot_examples=q_data.get("few_shot_examples"),
                )

                # Add question-specific rubric if present
                if q_data.get("question_rubric"):
                    from karenina.schemas import CallableTrait, MetricRubricTrait, RegexTrait

                    rubric_data = q_data["question_rubric"]
                    traits = []
                    regex_traits = []
                    callable_traits = []
                    metric_traits = []

                    # Process LLM-based traits
                    for trait_data in rubric_data.get("traits", []):
                        kind = trait_data.get("kind", "score")
                        # Map frontend trait types to backend TraitKind
                        if kind in ("binary", "Binary"):
                            kind = "boolean"
                        elif kind in ("score", "Score"):
                            kind = "score"

                        trait = LLMRubricTrait(
                            name=trait_data["name"],
                            description=trait_data.get("description"),
                            kind=kind,
                            min_score=trait_data.get("min_score", 1) if kind == "score" else None,
                            max_score=trait_data.get("max_score", 5) if kind == "score" else None,
                        )
                        traits.append(trait)

                    # Process regex traits
                    for regex_trait_data in rubric_data.get("regex_traits", []):
                        regex_trait = RegexTrait(
                            name=regex_trait_data["name"],
                            description=regex_trait_data.get("description"),
                            pattern=regex_trait_data.get("pattern", ""),
                            case_sensitive=regex_trait_data.get("case_sensitive", True),
                            invert_result=regex_trait_data.get("invert_result", False),
                        )
                        regex_traits.append(regex_trait)

                    # Process callable traits
                    for callable_trait_data in rubric_data.get("callable_traits", []):
                        callable_trait = CallableTrait(
                            name=callable_trait_data["name"],
                            description=callable_trait_data.get("description"),
                            callable_code=callable_trait_data.get("callable_code", b""),
                            kind=callable_trait_data.get("kind", "boolean"),
                            min_score=callable_trait_data.get("min_score"),
                            max_score=callable_trait_data.get("max_score"),
                            invert_result=callable_trait_data.get("invert_result", False),
                        )
                        callable_traits.append(callable_trait)

                    # Process metric traits
                    for metric_trait_data in rubric_data.get("metric_traits", []):
                        metric_trait = MetricRubricTrait(
                            name=metric_trait_data["name"],
                            description=metric_trait_data.get("description"),
                            evaluation_mode=metric_trait_data.get("evaluation_mode", "tp_only"),
                            metrics=metric_trait_data.get("metrics", []),
                            tp_instructions=metric_trait_data.get("tp_instructions", []),
                            tn_instructions=metric_trait_data.get("tn_instructions", []),
                            repeated_extraction=metric_trait_data.get("repeated_extraction", True),
                        )
                        metric_traits.append(metric_trait)

                    if traits or regex_traits or callable_traits or metric_traits:
                        rubric = Rubric(
                            traits=traits,
                            regex_traits=regex_traits,
                            callable_traits=callable_traits,
                            metric_traits=metric_traits,
                        )
                        benchmark.set_question_rubric(question_id, rubric)

            # Add global rubric if present
            if request.checkpoint_data.get("global_rubric"):
                from karenina.schemas import CallableTrait, RegexTrait

                global_rubric_data = request.checkpoint_data["global_rubric"]
                traits = []
                regex_traits = []
                callable_traits = []

                # Process LLM-based traits
                for trait_data in global_rubric_data.get("traits", []):
                    kind = trait_data.get("kind", "score")
                    # Map frontend trait types to backend TraitKind
                    if kind in ("binary", "Binary"):
                        kind = "boolean"
                    elif kind in ("score", "Score"):
                        kind = "score"

                    trait = LLMRubricTrait(
                        name=trait_data["name"],
                        description=trait_data.get("description"),
                        kind=kind,
                        min_score=trait_data.get("min_score", 1) if kind == "score" else None,
                        max_score=trait_data.get("max_score", 5) if kind == "score" else None,
                    )
                    traits.append(trait)

                # Process regex traits
                for regex_trait_data in global_rubric_data.get("regex_traits", []):
                    regex_trait = RegexTrait(
                        name=regex_trait_data["name"],
                        description=regex_trait_data.get("description"),
                        pattern=regex_trait_data.get("pattern", ""),
                        case_sensitive=regex_trait_data.get("case_sensitive", True),
                        invert_result=regex_trait_data.get("invert_result", False),
                    )
                    regex_traits.append(regex_trait)

                # Process callable traits
                for callable_trait_data in global_rubric_data.get("callable_traits", []):
                    callable_trait = CallableTrait(
                        name=callable_trait_data["name"],
                        description=callable_trait_data.get("description"),
                        callable_code=callable_trait_data.get("callable_code", b""),
                        kind=callable_trait_data.get("kind", "boolean"),
                        min_score=callable_trait_data.get("min_score"),
                        max_score=callable_trait_data.get("max_score"),
                        invert_result=callable_trait_data.get("invert_result", False),
                    )
                    callable_traits.append(callable_trait)

                if traits or regex_traits or callable_traits:
                    benchmark.global_rubric = Rubric(
                        traits=traits, regex_traits=regex_traits, callable_traits=callable_traits
                    )

            # Save to database (normal save, not detect-only)
            save_benchmark(benchmark, db_config, detect_duplicates_only=False)

            # Get updated timestamp from database
            from datetime import UTC, datetime

            last_modified = datetime.now(UTC).isoformat()

            return DuplicateResolutionResponse(
                success=True,
                message=f"Successfully resolved duplicates: kept {kept_old_count} old, {kept_new_count} new",
                last_modified=last_modified,
                kept_old_count=kept_old_count,
                kept_new_count=kept_new_count,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error resolving duplicates: {e!s}") from e

    @app.post("/api/database/init")  # type: ignore[misc]
    async def init_database_endpoint(request: dict[str, Any]) -> dict[str, Any]:
        """Initialize a new database."""
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            storage_url = request.get("storage_url")
            if not storage_url:
                raise HTTPException(status_code=400, detail="storage_url is required")

            db_config = DBConfig(storage_url=storage_url)
            init_database(db_config)

            return {"success": True, "storage_url": storage_url, "message": "Database initialized successfully"}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing database: {e!s}") from e

    @app.get("/api/database/list-databases", response_model=ListDatabasesResponse)  # type: ignore[misc]
    async def list_databases_endpoint() -> ListDatabasesResponse:
        """List all .db files in the DB_PATH directory."""
        try:
            # Get DB_PATH from environment, default to current working directory
            db_path = os.environ.get("DB_PATH")
            is_default = db_path is None
            db_directory = Path(db_path if db_path else os.getcwd())

            # Ensure directory exists
            if not db_directory.exists():
                raise HTTPException(status_code=404, detail=f"Database directory not found: {db_directory.absolute()}")

            # Find all .db files
            databases = []
            for db_file in db_directory.glob("*.db"):
                if db_file.is_file():
                    try:
                        size = db_file.stat().st_size
                    except Exception:
                        size = None

                    databases.append({"name": db_file.name, "path": str(db_file.absolute()), "size": size})

            # Sort by name
            databases.sort(key=lambda x: x["name"] or "")

            return ListDatabasesResponse(
                success=True,
                databases=databases,
                db_directory=str(db_directory.absolute()),
                is_default_directory=is_default,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing databases: {e!s}") from e
