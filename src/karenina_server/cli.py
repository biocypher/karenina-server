"""CLI interface for Karenina package."""

import argparse
import sys


def serve_webapp(
    host: str = "localhost",
    port: int = 8080,
    dev: bool = False,
    use_fastapi: bool = True,
    webapp_dir: str | None = None,
) -> None:
    """Serve the Karenina webapp locally.

    Args:
        host: Host address to bind to (default: localhost)
        port: Port to serve on (default: 8080)
        dev: Whether to run in development mode (default: False)
        use_fastapi: Whether to use FastAPI server (default: True)
        webapp_dir: Path to webapp directory (optional)
    """
    try:
        from .server import start_server

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

    args = parser.parse_args()

    if args.command == "serve":
        serve_webapp(
            host=args.host,
            port=args.port,
            dev=args.dev,
            use_fastapi=not args.simple,
            webapp_dir=getattr(args, "webapp_dir", None),
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
