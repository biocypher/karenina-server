"""Database management API handlers."""

from typing import Any

from fastapi import HTTPException

try:
    from karenina.storage import DBConfig, get_benchmark_summary, init_database, load_benchmark

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
                    "id": summary["benchmark_id"],
                    "name": summary["benchmark_name"],
                    "total_questions": summary["total_questions"],
                    "finished_count": summary.get("finished_count", 0),
                    "unfinished_count": summary.get("unfinished_count", 0),
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
            checkpoint_data = {
                "dataset_metadata": {
                    "name": benchmark.name,
                    "description": benchmark.description or "",
                    "version": benchmark.version,
                    "creator": benchmark.creator or "",
                    "created_at": benchmark.created_at.isoformat() if hasattr(benchmark, "created_at") else "",
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

    @app.post("/api/database/init")  # type: ignore[misc]
    async def init_database_endpoint(storage_url: str) -> dict[str, Any]:
        """Initialize a new database."""
        if not STORAGE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Storage functionality not available")

        try:
            db_config = DBConfig(storage_url=storage_url)
            init_database(db_config)

            return {"success": True, "storage_url": storage_url, "message": "Database initialized successfully"}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing database: {e!s}") from e
