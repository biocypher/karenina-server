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

# FastAPI imports
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseModel = None  # type: ignore[misc,assignment]

# Import LLM functionality from the karenina package
try:
    import karenina.llm  # noqa: F401 - Test if LLM module is available
    from karenina.llm.interface import LANGCHAIN_AVAILABLE

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LANGCHAIN_AVAILABLE = False

# Import Question Extractor functionality
try:
    import karenina.questions.extractor  # noqa: F401 - Test if extractor module is available

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

    class ExtractQuestionsRequest(BaseModel):
        file_id: str
        question_column: str
        answer_column: str
        sheet_name: str | None = None

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

    class TemplateGenerationRequest(BaseModel):
        questions: dict[str, Any]
        config: TemplateGenerationConfig
        custom_system_prompt: str | None = None

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
        estimated_time_remaining: float | None = None
        error: str | None = None
        result: dict[str, Any] | None = None

else:
    # Fallback classes for when FastAPI is not available
    FilePreviewResponse = None  # type: ignore[misc,assignment]
    ExtractQuestionsRequest = None  # type: ignore[misc,assignment]
    ExtractQuestionsResponse = None  # type: ignore[misc,assignment]
    TemplateGenerationRequest = None  # type: ignore[misc,assignment]
    TemplateGenerationResponse = None  # type: ignore[misc,assignment]
    TemplateGenerationStatusResponse = None  # type: ignore[misc,assignment]


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
    from .api.chat_handlers import register_chat_routes
    from .api.file_handlers import register_file_routes
    from .api.generation_handlers import register_generation_routes
    from .api.rubric_handlers import router as rubric_router
    from .api.verification_handlers import register_verification_routes

    # Register all route handlers
    register_chat_routes(app)
    register_file_routes(app, FilePreviewResponse, ExtractQuestionsRequest, ExtractQuestionsResponse)
    register_verification_routes(app, verification_service)
    register_generation_routes(
        app, TemplateGenerationRequest, TemplateGenerationResponse, TemplateGenerationStatusResponse
    )
    app.include_router(rubric_router, prefix="/api")

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
