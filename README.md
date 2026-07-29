# Karenina Server

FastAPI-based REST API for the [Karenina](https://github.com/biocypher/karenina) LLM benchmarking system.

## Overview

**Karenina Server** is the backend component of the Karenina graphical user interface stack, exposing the core [karenina](https://github.com/biocypher/karenina) library functionality through a REST API.

**Part of the Karenina stack:**
- **[karenina](https://github.com/biocypher/karenina)** - Core Python library for LLM benchmarking (works standalone)
- **karenina-server** (this package) - FastAPI backend exposing the library as REST API
- **[karenina-gui](https://github.com/biocypher/karenina-gui)** - React/TypeScript web application

Together, these three packages enable no-code web-based access to the Karenina framework for domain experts and non-technical users, as well as third-party integrations via standardized REST endpoints.

**Note**: Packaged releases can bundle the built `karenina-gui` assets and serve the existing GUI from the same FastAPI process as the API.

### Key Features

- **REST API**: Complete HTTP API for all Karenina operations
- **File Upload**: Support for Excel, CSV, and TSV file uploads
- **Async Job Management**: Long-running operations with progress tracking
- **WebSocket Support**: Real-time updates for benchmark progress
- **Static File Serving**: Serves the frontend web application

## Installation & Setup

For those who want to run this package independently:

### Prerequisites
- Python 3.11+ with `uv`
- [karenina](https://github.com/biocypher/karenina) library installed

### Basic Setup

```bash
# Install with uv
uv pip install karenina-server

# Or with pip
pip install karenina-server
```

Note: This package automatically installs the core `karenina` library as a dependency.

### Running the Server

```bash
# Start server (default: localhost:8080)
karenina-server serve

# Custom host/port
karenina-server serve --host 0.0.0.0 --port 3000

# Development mode with auto-reload
karenina-server serve --dev
```

API will be available at `http://localhost:8080/api/`

Interactive API documentation: `http://localhost:8080/docs`

Alternative docs: `http://localhost:8080/redoc`

### Webapp Asset Packaging

`karenina-server` is the Python package that serves both the REST API and the built GUI assets. At package build time, `build_hooks.py` looks for a sibling `karenina-gui` checkout, runs its production build, and copies `dist/` into `src/karenina_server/webapp/dist/` so runtime installs do not need Node.js or npm.

When neither GUI source nor pre-built assets are available, the build falls back to a placeholder webapp and warns. This keeps source installs working (editable checkouts, `pip install git+...`, CI) at the cost of serving the API with no UI. Set `KARENINA_RELEASE_BUILD=1` for any artifact you intend to publish: it turns that fallback into an error, and also refuses to reuse a placeholder left in `webapp/dist` by an earlier build.

Release/build options:

```bash
# Release build from a workspace containing ../karenina-gui
KARENINA_RELEASE_BUILD=1 uv build

# Use an explicit GUI checkout
KARENINA_GUI_DIR=/path/to/karenina-gui uv build

# Reuse pre-built src/karenina_server/webapp/dist assets without invoking npm
KARENINA_SKIP_GUI_BUILD=1 uv build

# Source build with no GUI available: writes a placeholder webapp and warns
uv build
```

Runtime serving uses packaged assets by default, and still supports overrides for development:

```bash
KARENINA_WEBAPP_DIR=/path/to/karenina-gui npm run build  # in GUI repo, if needed
karenina-server serve --webapp-dir /path/to/karenina-gui
```

For end users, the intended public entry point is the core package extra:

```bash
pip install "karenina[webapp]"
karenina serve
```

Note that neither `karenina` nor `karenina-server` is published to PyPI yet, so this command does not resolve today. Until both are published, install from git following the instructions in the [karenina](https://github.com/biocypher/karenina) README.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.
