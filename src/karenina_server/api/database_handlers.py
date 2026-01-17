"""Database management API handlers."""

import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException

try:
    from karenina.schemas import Rubric
    from karenina.storage import (
        DBConfig,
        ImportMetadataModel,
        VerificationRunModel,
        get_benchmark_summary,
        get_session,
        import_verification_results,
        init_database,
        load_benchmark,
        load_verification_results,
        save_benchmark,
    )

    from ..utils.rubric_utils import build_rubric_from_dict

    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    Rubric = None
    build_rubric_from_dict = None  # type: ignore[assignment]


# Trait keys for consistent iteration over rubric trait types
TRAIT_KEYS = ("llm_traits", "regex_traits", "callable_traits", "metric_traits")


def _serialize_trait(trait: Any) -> dict[str, Any]:
    """Serialize a single trait to dict format.

    Args:
        trait: A Pydantic model with model_dump() or already a dict.

    Returns:
        Dict representation of the trait.
    """
    if hasattr(trait, "model_dump"):
        return trait.model_dump()  # type: ignore[no-any-return]
    # trait is already a dict
    return dict(trait) if not isinstance(trait, dict) else trait


def _serialize_traits_from_object(rubric: Any, key: str) -> list[dict[str, Any]]:
    """Serialize traits from a Rubric object attribute.

    Args:
        rubric: A Rubric object with trait attributes.
        key: The trait key (e.g., 'llm_traits').

    Returns:
        List of serialized trait dicts.
    """
    traits = getattr(rubric, key, None) or []
    return [t.model_dump() for t in traits]


def _serialize_traits_from_dict(rubric: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Serialize traits from a rubric dict.

    Args:
        rubric: A dict with trait lists.
        key: The trait key (e.g., 'llm_traits').

    Returns:
        List of serialized trait dicts.
    """
    return [_serialize_trait(t) for t in rubric.get(key, [])]


def _serialize_rubric_to_dict(rubric: Any) -> dict[str, Any] | None:
    """Serialize a Rubric object or rubric dict to API-compatible dict format.

    This function handles both Rubric objects and dicts with trait lists,
    converting Pydantic models to dicts using model_dump().

    Args:
        rubric: A Rubric object, a dict with trait lists, or None

    Returns:
        Dict with traits serialized (llm_traits, regex_traits, callable_traits, metric_traits),
        or None if rubric is None/empty
    """
    if rubric is None:
        return None

    # Handle Rubric object (has llm_traits, regex_traits, etc. as lists of Pydantic models)
    if hasattr(rubric, "llm_traits"):
        result = {key: _serialize_traits_from_object(rubric, key) for key in TRAIT_KEYS}
    # Handle dict with trait object lists (from extract_questions_from_benchmark)
    elif isinstance(rubric, dict):
        result = {key: _serialize_traits_from_dict(rubric, key) for key in TRAIT_KEYS}
    else:
        return None

    # Return None if no traits at all
    if not any(result.values()):
        return None

    return result


def _normalize_creator_name(creator: str | dict[str, Any] | None) -> str:
    """Normalize creator value to a string, with 'Unknown' as fallback.

    Handles various input formats that may come from JSON-LD metadata:
    - None -> "Unknown"
    - String -> returned as-is
    - Person dict with 'name' key -> extracts the name
    - Any other format -> "Unknown"

    Args:
        creator: The creator value from metadata, which may be a string,
            a JSON-LD Person object dict, or None.

    Returns:
        A normalized string representation of the creator name.
        Returns "Unknown" if the creator cannot be determined.
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
    DeleteDatabaseRequest: Any,
    DeleteDatabaseResponse: Any,
    DeleteBenchmarkRequest: Any,
    DeleteBenchmarkResponse: Any,
    ImportResultsRequest: Any,
    ImportResultsResponse: Any,
    ListVerificationRunsRequest: Any,
    ListVerificationRunsResponse: Any,
    LoadVerificationResultsRequest: Any,
    LoadVerificationResultsResponse: Any,
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
                question_entry: dict[str, Any] = {
                    "id": question_id,
                    "question": q_data["question"],
                    "raw_answer": q_data["raw_answer"],
                    "answer_template": q_data.get("answer_template"),
                    "finished": q_data.get("finished", False),
                    "keywords": q_data.get("keywords", []),  # Frontend CheckpointItem expects "keywords"
                    "last_modified": updated_at_map.get(question_id, datetime.now().isoformat()),
                }

                # Include question-specific rubric if present
                question_rubric = q_data.get("question_rubric")
                if question_rubric:
                    serialized_rubric = _serialize_rubric_to_dict(question_rubric)
                    if serialized_rubric:
                        question_entry["question_rubric"] = serialized_rubric

                questions_data[question_id] = question_entry

            # Create checkpoint-like response
            # Convert created_at to string if it's a datetime object
            created_at_str = ""
            if hasattr(benchmark, "created_at") and benchmark.created_at:
                if hasattr(benchmark.created_at, "isoformat"):
                    created_at_str = benchmark.created_at.isoformat()
                else:
                    created_at_str = str(benchmark.created_at)

            # Serialize global rubric
            global_rubric = None
            loaded_global_rubric = benchmark.get_global_rubric()
            if loaded_global_rubric:
                global_rubric = _serialize_rubric_to_dict(loaded_global_rubric)

            checkpoint_data = {
                "dataset_metadata": {
                    "name": benchmark.name,
                    "description": benchmark.description or "",
                    "version": benchmark.version,
                    "creator": benchmark.creator or "",
                    "created_at": created_at_str,
                },
                "questions": questions_data,
                "global_rubric": global_rubric,
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
            from karenina.schemas import Question

            db_config = DBConfig(storage_url=request.storage_url)

            # Get dataset metadata from checkpoint
            metadata = request.checkpoint_data.get("dataset_metadata", {})

            # Create benchmark instance
            benchmark = Benchmark.create(
                name=request.benchmark_name,
                description=metadata.get("description", ""),
                version=metadata.get("version", "1.0.0"),
                creator=_normalize_creator_name(metadata.get("creator")),
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
                    rubric = build_rubric_from_dict(q_data["question_rubric"])
                    if rubric:
                        benchmark.set_question_rubric(question_id, rubric)

            # Add global rubric if present
            if request.checkpoint_data.get("global_rubric"):
                global_rubric = build_rubric_from_dict(request.checkpoint_data["global_rubric"])
                if global_rubric:
                    benchmark.set_global_rubric(global_rubric)

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
            from karenina.schemas import Question

            db_config = DBConfig(storage_url=request.storage_url)

            # Get dataset metadata from checkpoint
            metadata = request.checkpoint_data.get("dataset_metadata", {})

            # Create benchmark instance
            benchmark = Benchmark.create(
                name=request.benchmark_name,
                description=metadata.get("description", ""),
                version=metadata.get("version", "1.0.0"),
                creator=_normalize_creator_name(metadata.get("creator")),
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
                    rubric = build_rubric_from_dict(q_data["question_rubric"])
                    if rubric:
                        benchmark.set_question_rubric(question_id, rubric)

            # Add global rubric if present
            if request.checkpoint_data.get("global_rubric"):
                global_rubric = build_rubric_from_dict(request.checkpoint_data["global_rubric"])
                if global_rubric:
                    benchmark.set_global_rubric(global_rubric)

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

    @app.delete("/api/database/delete", response_model=DeleteDatabaseResponse)  # type: ignore[misc]
    async def delete_database_endpoint(request: DeleteDatabaseRequest) -> DeleteDatabaseResponse:
        """Delete a SQLite database file.

        Only works for SQLite databases. The database file must be in the DB_PATH directory
        (or current working directory if DB_PATH is not set).

        Safety measures:
        - Only SQLite databases can be deleted
        - Database must be within the allowed directory
        - Closes any active connections before deletion
        """
        try:
            storage_url = request.storage_url

            # Validate it's a SQLite URL
            if not storage_url.startswith("sqlite:///"):
                raise HTTPException(
                    status_code=400,
                    detail="Only SQLite databases can be deleted. URL must start with 'sqlite:///'",
                )

            # Extract the file path from the URL
            db_path = Path(storage_url.replace("sqlite:///", ""))

            # Validate the file exists
            if not db_path.exists():
                raise HTTPException(status_code=404, detail=f"Database file not found: {db_path}")

            if not db_path.is_file():
                raise HTTPException(status_code=400, detail=f"Path is not a file: {db_path}")

            # Validate the file is within the allowed directory
            db_directory = os.environ.get("DB_PATH")
            allowed_dir = Path(db_directory if db_directory else os.getcwd()).resolve()
            db_path_resolved = db_path.resolve()

            if not str(db_path_resolved).startswith(str(allowed_dir)):
                raise HTTPException(
                    status_code=403,
                    detail=f"Cannot delete database outside of allowed directory: {allowed_dir}",
                )

            # Close any active connections to this database
            if STORAGE_AVAILABLE:
                try:
                    from karenina.storage import close_engine

                    db_config = DBConfig(storage_url=storage_url)
                    close_engine(db_config)
                except Exception:
                    pass  # Ignore errors closing connections

            # Delete the file
            db_name = db_path.name
            db_path.unlink()

            return DeleteDatabaseResponse(
                success=True,
                message=f"Successfully deleted database: {db_name}",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting database: {e!s}") from e

    @app.delete("/api/database/delete-benchmark", response_model=DeleteBenchmarkResponse)  # type: ignore[misc]
    async def delete_benchmark_endpoint(request: DeleteBenchmarkRequest) -> DeleteBenchmarkResponse:
        """Delete a benchmark and all its associated data.

        This will delete:
        - The benchmark record
        - All questions associated with the benchmark
        - All verification runs and results associated with the benchmark
        """
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            from karenina.storage import BenchmarkModel, BenchmarkQuestionModel, VerificationRunModel
            from sqlalchemy import func, select

            db_config = DBConfig(storage_url=request.storage_url)

            with get_session(db_config) as session:
                # Find the benchmark
                benchmark = session.execute(
                    select(BenchmarkModel).where(BenchmarkModel.name == request.benchmark_name)
                ).scalar_one_or_none()

                if not benchmark:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Benchmark '{request.benchmark_name}' not found",
                    )

                # Count associated data for the response
                question_count = (
                    session.execute(
                        select(func.count(BenchmarkQuestionModel.id)).where(
                            BenchmarkQuestionModel.benchmark_id == benchmark.id
                        )
                    ).scalar()
                    or 0
                )

                run_count = (
                    session.execute(
                        select(func.count(VerificationRunModel.id)).where(
                            VerificationRunModel.benchmark_id == benchmark.id
                        )
                    ).scalar()
                    or 0
                )

                # Delete the benchmark (cascade will delete questions and runs)
                session.delete(benchmark)
                session.commit()

                return DeleteBenchmarkResponse(
                    success=True,
                    message=f"Successfully deleted benchmark '{request.benchmark_name}'",
                    deleted_questions=question_count,
                    deleted_runs=run_count,
                )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting benchmark: {e!s}") from e

    # =========================================================================
    # Verification Results Import/Export Endpoints
    # =========================================================================

    @app.post("/api/database/import-results", response_model=ImportResultsResponse)  # type: ignore[misc]
    async def import_results_endpoint(request: ImportResultsRequest) -> ImportResultsResponse:
        """Import verification results from JSON export format."""
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            db_config = DBConfig(storage_url=request.storage_url)

            # Import the results
            run_id, imported_count, skipped_count = import_verification_results(
                json_data=request.json_data,
                db_config=db_config,
                benchmark_name=request.benchmark_name,
                run_name=request.run_name,
            )

            return ImportResultsResponse(
                success=True,
                run_id=run_id,
                imported_count=imported_count,
                message=f"Successfully imported {imported_count} verification results ({skipped_count} skipped)",
            )

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error importing results: {e!s}") from e

    @app.post("/api/database/verification-runs", response_model=ListVerificationRunsResponse)  # type: ignore[misc]
    async def list_verification_runs_endpoint(request: ListVerificationRunsRequest) -> ListVerificationRunsResponse:
        """List all verification runs in the database."""
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            from sqlalchemy import select

            db_config = DBConfig(storage_url=request.storage_url)

            runs = []
            with get_session(db_config) as session:
                # Build query
                query = select(VerificationRunModel)

                # Filter by benchmark if specified
                if request.benchmark_name:
                    from karenina.storage import BenchmarkModel

                    benchmark = session.execute(
                        select(BenchmarkModel).where(BenchmarkModel.name == request.benchmark_name)
                    ).scalar_one_or_none()
                    if benchmark:
                        query = query.where(VerificationRunModel.benchmark_id == benchmark.id)
                    else:
                        # No benchmark found, return empty list
                        return ListVerificationRunsResponse(success=True, runs=[], count=0)

                # Execute query
                run_models = session.execute(query.order_by(VerificationRunModel.created_at.desc())).scalars().all()

                # Check which runs have import metadata
                imported_run_ids = set()
                import_results = session.execute(select(ImportMetadataModel.run_id)).scalars().all()
                imported_run_ids = set(import_results)

                for run in run_models:
                    # Get benchmark name
                    from karenina.storage import BenchmarkModel

                    benchmark = session.get(BenchmarkModel, run.benchmark_id)
                    benchmark_name = benchmark.name if benchmark else "Unknown"

                    runs.append(
                        {
                            "id": run.id,
                            "run_name": run.run_name,
                            "benchmark_id": run.benchmark_id,
                            "benchmark_name": benchmark_name,
                            "status": run.status,
                            "total_questions": run.total_questions or 0,
                            "processed_count": run.processed_count or 0,
                            "successful_count": run.successful_count or 0,
                            "failed_count": run.failed_count or 0,
                            "start_time": run.start_time.isoformat() if run.start_time else None,
                            "end_time": run.end_time.isoformat() if run.end_time else None,
                            "created_at": run.created_at.isoformat() if run.created_at else "",
                            "is_imported": run.id in imported_run_ids,
                        }
                    )

            return ListVerificationRunsResponse(
                success=True,
                runs=runs,
                count=len(runs),
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing verification runs: {e!s}") from e

    @app.post("/api/database/load-results", response_model=LoadVerificationResultsResponse)  # type: ignore[misc]
    async def load_verification_results_endpoint(
        request: LoadVerificationResultsRequest,
    ) -> LoadVerificationResultsResponse:
        """Load verification results with filtering options."""
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            db_config = DBConfig(storage_url=request.storage_url)

            # Load results using the storage function with as_dict=False for list format
            results = load_verification_results(
                db_config=db_config,
                run_id=request.run_id,
                benchmark_name=request.benchmark_name,
                question_id=request.question_id,
                answering_model=request.answering_model,
                limit=request.limit,
                as_dict=False,
            )

            # Convert to summary format
            result_summaries = []
            for result in results:
                result_summaries.append(
                    {
                        "id": result.get("id", 0),
                        "run_id": result.get("run_id", ""),
                        "question_id": result.get("metadata", {}).get("question_id", ""),
                        "question_text": result.get("metadata", {}).get("question_text", ""),
                        "answering_model": result.get("metadata", {}).get("answering_model", ""),
                        "parsing_model": result.get("metadata", {}).get("parsing_model", ""),
                        "completed_without_errors": result.get("metadata", {}).get("completed_without_errors", False),
                        "template_verify_result": result.get("template", {}).get("verify_result")
                        if result.get("template")
                        else None,
                        "execution_time": result.get("metadata", {}).get("execution_time", 0.0),
                        "timestamp": result.get("metadata", {}).get("timestamp", ""),
                    }
                )

            return LoadVerificationResultsResponse(
                success=True,
                results=result_summaries,
                count=len(result_summaries),
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading verification results: {e!s}") from e
