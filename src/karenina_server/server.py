"""Server module for serving the Karenina webapp."""

import http.server
import os
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()  # Load .env from project root

# FastAPI imports
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, SecretStr

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseModel = None  # type: ignore[misc,assignment]

# Import LLM functionality from the karenina package
try:
    import karenina.infrastructure.llm  # noqa: F401 - Test if LLM module is available

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import Question Extractor functionality
try:
    import karenina.domain.questions.extractor  # noqa: F401 - Test if extractor module is available

    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False


# Pydantic models for Question Extractor API
if FASTAPI_AVAILABLE and BaseModel is not None:
    # Type definitions
    QuestionData = dict  # Dictionary mapping question IDs to question data

    class FilePreviewResponse(BaseModel):
        success: bool
        total_rows: int | None = None
        columns: list[str] | None = None
        preview_rows: int | None = None
        data: list[dict[str, Any]] | None = None
        error: str | None = None

    class KeywordColumnConfig(BaseModel):
        column: str
        separator: str

    class ExtractQuestionsRequest(BaseModel):
        file_id: str
        question_column: str
        answer_column: str
        sheet_name: str | None = None
        # Optional metadata column mappings
        author_name_column: str | None = None
        author_email_column: str | None = None
        author_affiliation_column: str | None = None
        url_column: str | None = None
        # New format: multiple keyword columns with individual separators
        keywords_columns: list[dict[str, str]] | None = None
        # Deprecated: kept for backward compatibility
        keywords_column: str | None = None
        keywords_separator: str = ","

    class ExtractQuestionsResponse(BaseModel):
        success: bool
        questions_count: int | None = None
        questions_data: dict[str, Any] | None = None
        error: str | None = None

    # Template Generation API Models
    class TemplateGenerationConfig(BaseModel):
        model_provider: str
        model_name: str
        temperature: float = 0.1
        interface: str = "langchain"
        endpoint_base_url: str | None = None
        endpoint_api_key: SecretStr | None = None

    class TemplateGenerationRequest(BaseModel):
        questions: dict[str, Any]
        config: TemplateGenerationConfig
        force_regenerate: bool = False

    class TemplateGenerationResponse(BaseModel):
        job_id: str
        status: str
        message: str

    class TemplateGenerationStatusResponse(BaseModel):
        job_id: str
        status: str
        percentage: float
        current_question: str
        processed_count: int
        total_count: int
        duration_seconds: float | None = None
        last_task_duration: float | None = None
        error: str | None = None
        result: dict[str, Any] | None = None
        in_progress_questions: list[str] = []

    # MCP Validation API Models
    class MCPTool(BaseModel):
        name: str
        description: str | None = None

    class MCPValidationRequest(BaseModel):
        server_name: str
        server_url: str

    class MCPValidationResponse(BaseModel):
        success: bool
        tools: list[MCPTool] | None = None
        error: str | None = None

    # Database Management API Models
    class DatabaseConnectRequest(BaseModel):
        storage_url: str
        create_if_missing: bool = True

    class DatabaseConnectResponse(BaseModel):
        success: bool
        storage_url: str
        benchmark_count: int
        message: str
        error: str | None = None

    class BenchmarkInfo(BaseModel):
        id: str
        name: str
        total_questions: int
        finished_count: int
        unfinished_count: int
        last_modified: str | None = None

    class BenchmarkListResponse(BaseModel):
        success: bool
        benchmarks: list[BenchmarkInfo]
        count: int
        error: str | None = None

    class BenchmarkLoadRequest(BaseModel):
        storage_url: str
        benchmark_name: str

    class BenchmarkLoadResponse(BaseModel):
        success: bool
        benchmark_name: str
        checkpoint_data: dict[str, Any]
        storage_url: str
        message: str
        error: str | None = None

    class BenchmarkCreateRequest(BaseModel):
        storage_url: str
        name: str
        description: str | None = None
        version: str | None = None
        creator: str | None = None

    class BenchmarkCreateResponse(BaseModel):
        success: bool
        benchmark_name: str
        checkpoint_data: dict[str, Any]
        storage_url: str
        message: str
        error: str | None = None

    class DuplicateQuestionInfo(BaseModel):
        question_id: str
        question_text: str
        old_version: dict[str, Any]  # Full question data from database
        new_version: dict[str, Any]  # Full question data from current checkpoint

    class BenchmarkSaveRequest(BaseModel):
        storage_url: str
        benchmark_name: str
        checkpoint_data: dict[str, Any]
        detect_duplicates: bool = False  # If True, only detect duplicates without saving

    class BenchmarkSaveResponse(BaseModel):
        success: bool
        message: str
        last_modified: str | None = None
        duplicates: list[DuplicateQuestionInfo] | None = None  # Present when duplicates detected
        error: str | None = None

    class DuplicateResolutionRequest(BaseModel):
        storage_url: str
        benchmark_name: str
        checkpoint_data: dict[str, Any]
        resolutions: dict[str, str]  # Map of question_id -> "keep_old" | "keep_new"

    class DuplicateResolutionResponse(BaseModel):
        success: bool
        message: str
        last_modified: str
        kept_old_count: int
        kept_new_count: int
        error: str | None = None

    class DatabaseInfo(BaseModel):
        name: str
        path: str
        size: int | None = None

    class ListDatabasesResponse(BaseModel):
        success: bool
        databases: list[DatabaseInfo]
        db_directory: str
        is_default_directory: bool
        error: str | None = None

    # Verification Results Import/Export Models
    class ImportResultsRequest(BaseModel):
        storage_url: str
        json_data: dict[str, Any]
        benchmark_name: str
        run_name: str | None = None

    class ImportResultsResponse(BaseModel):
        success: bool
        run_id: str
        imported_count: int
        message: str
        error: str | None = None

    class VerificationRunInfo(BaseModel):
        id: str
        run_name: str | None
        benchmark_id: str
        benchmark_name: str
        status: str
        total_questions: int
        processed_count: int
        successful_count: int
        failed_count: int
        start_time: str | None
        end_time: str | None
        created_at: str
        is_imported: bool = False

    class ListVerificationRunsRequest(BaseModel):
        storage_url: str
        benchmark_name: str | None = None

    class ListVerificationRunsResponse(BaseModel):
        success: bool
        runs: list[VerificationRunInfo]
        count: int
        error: str | None = None

    class LoadVerificationResultsRequest(BaseModel):
        storage_url: str
        run_id: str | None = None
        benchmark_name: str | None = None
        question_id: str | None = None
        answering_model: str | None = None
        limit: int | None = None

    class VerificationResultSummary(BaseModel):
        id: int
        run_id: str
        question_id: str
        question_text: str
        answering_model: str
        parsing_model: str
        completed_without_errors: bool
        template_verify_result: Any
        execution_time: float
        timestamp: str

    class LoadVerificationResultsResponse(BaseModel):
        success: bool
        results: list[VerificationResultSummary]
        count: int
        error: str | None = None

else:
    # Fallback classes for when FastAPI is not available
    FilePreviewResponse = None  # type: ignore[misc,assignment]
    ExtractQuestionsRequest = None  # type: ignore[misc,assignment]
    ExtractQuestionsResponse = None  # type: ignore[misc,assignment]
    TemplateGenerationRequest = None  # type: ignore[misc,assignment]
    TemplateGenerationResponse = None  # type: ignore[misc,assignment]
    TemplateGenerationStatusResponse = None  # type: ignore[misc,assignment]
    MCPTool = None  # type: ignore[misc,assignment]
    MCPValidationRequest = None  # type: ignore[misc,assignment]
    MCPValidationResponse = None  # type: ignore[misc,assignment]
    DatabaseConnectRequest = None  # type: ignore[misc,assignment]
    DatabaseConnectResponse = None  # type: ignore[misc,assignment]
    BenchmarkInfo = None  # type: ignore[misc,assignment]
    BenchmarkListResponse = None  # type: ignore[misc,assignment]
    BenchmarkLoadRequest = None  # type: ignore[misc,assignment]
    BenchmarkLoadResponse = None  # type: ignore[misc,assignment]
    BenchmarkCreateRequest = None  # type: ignore[misc,assignment]
    BenchmarkCreateResponse = None  # type: ignore[misc,assignment]
    BenchmarkSaveRequest = None  # type: ignore[misc,assignment]
    BenchmarkSaveResponse = None  # type: ignore[misc,assignment]
    DatabaseInfo = None  # type: ignore[misc,assignment]
    ListDatabasesResponse = None  # type: ignore[misc,assignment]
    ImportResultsRequest = None  # type: ignore[misc,assignment]
    ImportResultsResponse = None  # type: ignore[misc,assignment]
    VerificationRunInfo = None  # type: ignore[misc,assignment]
    ListVerificationRunsRequest = None  # type: ignore[misc,assignment]
    ListVerificationRunsResponse = None  # type: ignore[misc,assignment]
    LoadVerificationResultsRequest = None  # type: ignore[misc,assignment]
    VerificationResultSummary = None  # type: ignore[misc,assignment]
    LoadVerificationResultsResponse = None  # type: ignore[misc,assignment]


# Global verification service instance
verification_service = None


class KareninaHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for serving the Karenina webapp."""

    def __init__(self, *args: Any, webapp_dir: Path, **kwargs: Any) -> None:
        self.webapp_dir = webapp_dir
        super().__init__(*args, directory=str(webapp_dir), **kwargs)

    def end_headers(self) -> None:
        """Add CORS headers and other necessary headers."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        """Handle OPTIONS requests for CORS."""
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests with SPA routing support."""
        # For SPA routing, serve index.html for non-asset requests
        if (
            not self.path.startswith("/assets/")
            and not self.path.endswith(".js")
            and not self.path.endswith(".css")
            and not self.path.endswith(".json")
            and not self.path.endswith(".ico")
            and self.path != "/"
            and not Path(self.webapp_dir / self.path.lstrip("/")).exists()
        ):
            self.path = "/index.html"

        super().do_GET()


def find_webapp_directory(webapp_path: str | None = None) -> Path:
    """Find the webapp directory relative to the package."""
    # If explicit path provided, use it
    if webapp_path:
        webapp_dir = Path(webapp_path)
        if not webapp_dir.exists():
            raise FileNotFoundError(f"Specified webapp directory not found: {webapp_path}")
        return webapp_dir

    # Check environment variable
    env_path = os.environ.get("KARENINA_WEBAPP_DIR")
    if env_path:
        webapp_dir = Path(env_path)
        if not webapp_dir.exists():
            raise FileNotFoundError(f"Environment KARENINA_WEBAPP_DIR directory not found: {env_path}")
        return webapp_dir

    # Try to find webapp directory relative to this file
    current_dir = Path(__file__).parent
    webapp_dir = current_dir.parent / "webapp"

    if not webapp_dir.exists():
        # Try alternative locations
        alternatives = [
            current_dir / "webapp",
            current_dir.parent.parent / "webapp",
        ]

        for alt in alternatives:
            if alt.exists():
                webapp_dir = alt
                break
        else:
            raise FileNotFoundError(
                f"Webapp directory not found. Searched in: {webapp_dir} and alternatives: {alternatives}. Set KARENINA_WEBAPP_DIR environment variable or use --webapp-dir option."
            )

    return webapp_dir


def build_webapp(webapp_dir: Path, force_rebuild: bool = False) -> Path:
    """Build the webapp if needed and return the dist directory."""
    dist_dir = webapp_dir / "dist"
    package_json = webapp_dir / "package.json"

    # Check if we need to build
    if not force_rebuild and dist_dir.exists() and dist_dir.is_dir():
        # Check if dist is newer than src
        try:
            src_dir = webapp_dir / "src"
            if src_dir.exists():
                src_mtime = max(f.stat().st_mtime for f in src_dir.rglob("*") if f.is_file())
                dist_mtime = min(f.stat().st_mtime for f in dist_dir.rglob("*") if f.is_file())
                if dist_mtime > src_mtime:
                    print("✓ Webapp build is up to date")
                    return dist_dir
            else:
                # No src directory means this is a pre-built package installation
                print("✓ Using pre-built webapp assets")
                return dist_dir
        except (ValueError, StopIteration):
            pass  # Fall through to rebuild

    if not package_json.exists():
        raise FileNotFoundError(f"package.json not found in {webapp_dir}")

    print("🔧 Building webapp...")

    # Check if node_modules exists, if not run npm install
    node_modules = webapp_dir / "node_modules"
    if not node_modules.exists():
        print("📦 Installing dependencies...")
        try:
            subprocess.run(["npm", "install"], cwd=webapp_dir, check=True, capture_output=True, text=True)
            print("✓ Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e.stderr}")
            raise
        except FileNotFoundError as e:
            raise RuntimeError(
                "npm not found. Please install Node.js and npm to build the webapp.\nVisit: https://nodejs.org/"
            ) from e

    # Build the webapp
    try:
        subprocess.run(["npm", "run", "build"], cwd=webapp_dir, check=True, capture_output=True, text=True)
        print("✓ Webapp built successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build webapp: {e.stderr}")
        raise

    if not dist_dir.exists():
        raise RuntimeError("Build completed but dist directory not found")

    return dist_dir


def start_development_server(webapp_dir: Path, host: str, port: int) -> None:
    """Start the development server using Vite."""
    print("🚀 Starting development server...")

    try:
        # Check if dependencies are installed
        node_modules = webapp_dir / "node_modules"
        if not node_modules.exists():
            print("📦 Installing dependencies...")
            subprocess.run(["npm", "install"], cwd=webapp_dir, check=True)

        # Start Vite dev server
        env = os.environ.copy()
        env["HOST"] = host
        env["PORT"] = str(port)

        subprocess.run(["npm", "run", "dev", "--", "--host", host, "--port", str(port)], cwd=webapp_dir, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start development server: {e}")
        raise
    except FileNotFoundError as e:
        raise RuntimeError(
            "npm not found. Please install Node.js and npm to run the development server.\nVisit: https://nodejs.org/"
        ) from e


def start_production_server(dist_dir: Path, host: str, port: int) -> None:
    """Start the production server serving the built webapp."""
    print(f"🌐 Starting production server at http://{host}:{port}")

    # Create a custom handler with the dist directory
    def handler_factory(*args: Any, **kwargs: Any) -> KareninaHTTPRequestHandler:
        return KareninaHTTPRequestHandler(*args, webapp_dir=dist_dir, **kwargs)

    try:
        with socketserver.TCPServer((host, port), handler_factory) as httpd:
            # Open browser after a short delay
            def open_browser() -> None:
                time.sleep(1)
                webbrowser.open(f"http://{host}:{port}")

            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()

            print(f"✓ Server running at http://{host}:{port}")
            print("Press Ctrl+C to stop the server")

            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 Server stopped")

    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Try a different port with --port")
        else:
            print(f"❌ Failed to start server: {e}")
        raise


def start_server(
    host: str = "localhost",
    port: int = 8080,
    dev: bool = False,
    use_fastapi: bool = True,
    webapp_dir: str | None = None,
) -> None:
    """Start the Karenina webapp server.

    Args:
        host: Host address to bind to
        port: Port to serve on
        dev: Whether to run in development mode
        use_fastapi: Whether to use FastAPI server (default) or simple HTTP server
        webapp_dir: Path to webapp directory (overrides environment variable and auto-detection)
    """
    try:
        webapp_path = find_webapp_directory(webapp_dir)
        print(f"📁 Found webapp directory: {webapp_path}")

        if dev:
            start_development_server(webapp_path, host, port)
        elif use_fastapi:
            start_fastapi_server(webapp_path, host, port)
        else:
            dist_dir = build_webapp(webapp_path)
            start_production_server(dist_dir, host, port)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


# Pydantic models and session management are now imported from the llm submodule


# call_model function has been moved to the llm submodule


def create_fastapi_app(webapp_dir: Path) -> FastAPI:
    """Create FastAPI application with API routes and static file serving."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not available. Please install FastAPI dependencies.")

    app = FastAPI(title="Karenina API", description="API for Karenina webapp with LLM integration", version="1.0.0")

    # Initialize global services
    global verification_service
    if verification_service is None:
        try:
            from karenina_server.services.verification_service import VerificationService

            verification_service = VerificationService()
        except ImportError:
            print("Warning: Verification service not available")

    # Register API routes from extracted handlers
    from .api.config_handlers import router as config_router
    from .api.database_handlers import register_database_routes
    from .api.file_handlers import register_file_routes
    from .api.generation_handlers import register_generation_routes
    from .api.health_handlers import router as health_router
    from .api.mcp_handlers import register_mcp_routes
    from .api.preset_handlers import router as preset_router
    from .api.rubric_handlers import router as rubric_router
    from .api.verification_handlers import register_verification_routes

    # Register all route handlers
    register_file_routes(app, FilePreviewResponse, ExtractQuestionsRequest, ExtractQuestionsResponse)
    register_verification_routes(app, verification_service)
    register_generation_routes(
        app, TemplateGenerationRequest, TemplateGenerationResponse, TemplateGenerationStatusResponse
    )
    register_mcp_routes(app, MCPValidationRequest, MCPValidationResponse)
    register_database_routes(
        app,
        DatabaseConnectRequest,
        DatabaseConnectResponse,
        BenchmarkListResponse,
        BenchmarkLoadRequest,
        BenchmarkLoadResponse,
        BenchmarkCreateRequest,
        BenchmarkCreateResponse,
        BenchmarkSaveRequest,
        BenchmarkSaveResponse,
        DuplicateResolutionRequest,
        DuplicateResolutionResponse,
        ListDatabasesResponse,
        ImportResultsRequest,
        ImportResultsResponse,
        ListVerificationRunsRequest,
        ListVerificationRunsResponse,
        LoadVerificationResultsRequest,
        LoadVerificationResultsResponse,
    )
    app.include_router(health_router, prefix="/api")
    app.include_router(rubric_router, prefix="/api")
    app.include_router(config_router, prefix="/api/config")
    app.include_router(preset_router, prefix="/api")

    # Set up event loop for broadcasters on startup
    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize event loops for progress broadcasters."""
        import asyncio

        loop = asyncio.get_running_loop()

        # Set event loop for verification service broadcaster
        if verification_service is not None:
            verification_service.broadcaster.set_event_loop(loop)

        # Set event loop for generation service broadcaster if it exists
        try:
            from karenina_server.services.generation_service import generation_service

            if generation_service is not None:
                generation_service.broadcaster.set_event_loop(loop)
        except ImportError:
            pass

    # Serve static files from the webapp dist directory
    dist_dir = webapp_dir / "dist"
    if dist_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="assets")

        @app.get("/{full_path:path}")
        async def serve_webapp(full_path: str) -> FileResponse:
            """Serve the webapp with SPA routing support."""
            # Serve specific files if they exist
            file_path = dist_dir / full_path
            if file_path.is_file():
                return FileResponse(file_path)

            # For SPA routing, serve index.html for all other routes
            index_path = dist_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)

            raise HTTPException(status_code=404, detail="File not found")

    return app


def start_fastapi_server(webapp_dir: Path, host: str, port: int) -> None:
    """Start the FastAPI server."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is not available. Please install FastAPI dependencies:\npip install fastapi uvicorn"
        )

    print(f"🚀 Starting FastAPI server at http://{host}:{port}")

    # Build webapp if needed
    dist_dir = build_webapp(webapp_dir)
    print(f"📁 Serving webapp from: {dist_dir}")

    # Create FastAPI app
    app = create_fastapi_app(webapp_dir)

    # Open browser after a short delay
    def open_browser() -> None:
        time.sleep(2)
        webbrowser.open(f"http://{host}:{port}")

    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    print(f"✓ FastAPI server running at http://{host}:{port}")
    print(f"📖 API documentation at http://{host}:{port}/docs")
    print("Press Ctrl+C to stop the server")

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n�� Server stopped")


# Create app instance for uvicorn
# This allows uvicorn to find the app when running: uvicorn karenina_server.server:app
webapp_dir = Path(__file__).parent.parent.parent / "karenina-gui" / "dist"
app = create_fastapi_app(webapp_dir)
