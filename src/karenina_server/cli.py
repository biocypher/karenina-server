"""CLI interface for Karenina package."""

import argparse
import sys


def serve_webapp(
    host: str = "localhost",
    port: int = 8080,
    dev: bool = False,
    use_fastapi: bool = True,
    webapp_dir: str | None = None,
    async_enabled: bool | None = None,
    async_chunk_size: int | None = None,
    async_max_workers: int | None = None,
) -> None:
    """Serve the Karenina webapp locally.

    Args:
        host: Host address to bind to (default: localhost)
        port: Port to serve on (default: 8080)
        dev: Whether to run in development mode (default: False)
        use_fastapi: Whether to use FastAPI server (default: True)
        webapp_dir: Path to webapp directory (optional)
        async_enabled: Whether async processing is enabled (default: from environment)
        async_chunk_size: Chunk size for async processing (default: from environment)
        async_max_workers: Maximum number of async workers (default: from environment)
    """
    try:
        import os

        from .server import start_server

        # Set async configuration environment variables if provided
        if async_enabled is not None:
            os.environ["KARENINA_ASYNC_ENABLED"] = str(async_enabled).lower()
        if async_chunk_size is not None:
            os.environ["KARENINA_ASYNC_CHUNK_SIZE"] = str(async_chunk_size)
        if async_max_workers is not None:
            os.environ["KARENINA_ASYNC_MAX_WORKERS"] = str(async_max_workers)

        start_server(host=host, port=port, dev=dev, use_fastapi=use_fastapi, webapp_dir=webapp_dir)
    except ImportError as e:
        print(f"Error: Missing dependencies for serving webapp: {e}")
        print("Please install the webapp dependencies:")
        print("  pip install otarbench[webapp]")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="karenina-server",
        description="Karenina - A benchmark library for question extraction and processing",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the Karenina webapp server")
    serve_parser.add_argument("--host", default="localhost", help="Host address to bind to (default: localhost)")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    serve_parser.add_argument("--dev", action="store_true", help="Run in development mode with auto-reload")
    serve_parser.add_argument("--simple", action="store_true", help="Use simple HTTP server instead of FastAPI")
    serve_parser.add_argument("--webapp-dir", help="Path to webapp directory (overrides environment variable)")

    # Async configuration arguments
    async_group = serve_parser.add_mutually_exclusive_group()
    async_group.add_argument(
        "--async", dest="async_enabled", action="store_true", help="Enable async processing (default)"
    )
    async_group.add_argument("--no-async", dest="async_enabled", action="store_false", help="Disable async processing")
    serve_parser.add_argument(
        "--async-chunk-size", type=int, metavar="N", help="Number of items to process in parallel chunks (default: 5)"
    )
    serve_parser.add_argument(
        "--async-max-workers",
        type=int,
        metavar="N",
        help="Maximum number of worker threads (default: min(32, num_items))",
    )
    serve_parser.set_defaults(async_enabled=None)  # Use environment default

    args = parser.parse_args()

    if args.command == "serve":
        serve_webapp(
            host=args.host,
            port=args.port,
            dev=args.dev,
            use_fastapi=not args.simple,
            webapp_dir=getattr(args, "webapp_dir", None),
            async_enabled=args.async_enabled,
            async_chunk_size=getattr(args, "async_chunk_size", None),
            async_max_workers=getattr(args, "async_max_workers", None),
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
