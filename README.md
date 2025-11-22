# Karenina Server

FastAPI-based REST API for the [Karenina](https://github.com/biocypher/karenina) LLM benchmarking system.

## Overview

**Karenina Server** is the backend component of the Karenina graphical user interface stack, exposing the core [karenina](https://github.com/biocypher/karenina) library functionality through a REST API.

**Part of the Karenina stack:**
- **[karenina](https://github.com/biocypher/karenina)** - Core Python library for LLM benchmarking (works standalone)
- **karenina-server** (this package) - FastAPI backend exposing the library as REST API
- **[karenina-gui](https://github.com/biocypher/karenina-gui)** - React/TypeScript web application

Together, these three packages enable no-code web-based access to the Karenina framework for domain experts and non-technical users, as well as third-party integrations via standardized REST endpoints.

**Note**: The full stack integration is currently a work in progress. Comprehensive instructions for spinning up the complete web-based system will be provided soon.

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.
