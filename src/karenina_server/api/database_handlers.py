"""Database management API handlers."""

from typing import Any

from fastapi import HTTPException

try:
    from karenina.storage import DBConfig, get_benchmark_summary, init_database, load_benchmark, save_benchmark

    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False


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
            db_config = DBConfig(storage_url=request.storage_url)

            # Load benchmark
            benchmark, loaded_config = load_benchmark(request.benchmark_name, db_config, load_config=True)

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
                    "tags": q_data.get("tags", []),
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
            from karenina.schemas.question_class import Question
            from karenina.schemas.rubric_class import Rubric, RubricTrait

            db_config = DBConfig(storage_url=request.storage_url)

            # Get dataset metadata from checkpoint
            metadata = request.checkpoint_data.get("dataset_metadata", {})

            # Create benchmark instance
            benchmark = Benchmark.create(
                name=request.benchmark_name,
                description=metadata.get("description", ""),
                version=metadata.get("version", "1.0.0"),
                creator=metadata.get("creator", "Unknown"),
            )

            # Add questions from checkpoint
            questions_data = request.checkpoint_data.get("questions", {})
            for question_id, q_data in questions_data.items():
                # Create Question object
                question = Question(
                    question=q_data["question"],
                    raw_answer=q_data["raw_answer"],
                    tags=q_data.get("tags", []),
                    few_shot_examples=q_data.get("few_shot_examples"),
                )

                # Add question to benchmark
                benchmark.add_question(
                    question=question,
                    answer_template=q_data.get("answer_template", ""),
                    finished=q_data.get("finished", False),
                    few_shot_examples=q_data.get("few_shot_examples"),
                )

                # Add question-specific rubric if present
                if q_data.get("question_rubric"):
                    rubric_data = q_data["question_rubric"]
                    traits = []
                    for trait_data in rubric_data.get("traits", []):
                        kind = trait_data.get("kind", "score")
                        trait = RubricTrait(
                            name=trait_data["name"],
                            description=trait_data.get("description"),
                            kind=kind,
                            min_score=trait_data.get("min_score", 1) if kind == "score" else None,
                            max_score=trait_data.get("max_score", 5) if kind == "score" else None,
                        )
                        traits.append(trait)

                    if traits:
                        rubric = Rubric(traits=traits)
                        benchmark.set_question_rubric(question_id, rubric)

            # Add global rubric if present
            if request.checkpoint_data.get("global_rubric"):
                global_rubric_data = request.checkpoint_data["global_rubric"]
                traits = []
                for trait_data in global_rubric_data.get("traits", []):
                    kind = trait_data.get("kind", "score")
                    trait = RubricTrait(
                        name=trait_data["name"],
                        description=trait_data.get("description"),
                        kind=kind,
                        min_score=trait_data.get("min_score", 1) if kind == "score" else None,
                        max_score=trait_data.get("max_score", 5) if kind == "score" else None,
                    )
                    traits.append(trait)

                if traits:
                    benchmark.global_rubric = Rubric(traits=traits)

            # Save to database (will overwrite if exists)
            save_benchmark(benchmark, db_config)

            # Get updated timestamp from database
            from datetime import UTC, datetime

            last_modified = datetime.now(UTC).isoformat()

            return BenchmarkSaveResponse(
                success=True,
                message=f"Benchmark '{request.benchmark_name}' saved successfully",
                last_modified=last_modified,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving benchmark: {e!s}") from e

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
