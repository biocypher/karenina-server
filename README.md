# Karenina Server

FastAPI server for the Karenina benchmarking system, providing a web API layer for interactive LLM benchmark management.

## Overview

Karenina Server is the web service component that provides:

- **REST API**: Complete HTTP API for all Karenina operations
- **File Upload**: Support for Excel, CSV, and TSV file uploads
- **Async Job Management**: Long-running operations with progress tracking
- **WebSocket Support**: Real-time updates for benchmark progress
- **Static File Serving**: Serves the frontend web application

## Installation

```bash
pip install karenina-server
```

For development:
```bash
pip install karenina-server[dev]
```

Note: This package automatically installs the core `karenina` library as a dependency.

## Quick Start

### Start the Server

```bash
# Start with default settings (localhost:8080)
karenina-server serve

# Custom host and port
karenina-server serve --host 0.0.0.0 --port 3000

# Development mode with auto-reload
karenina-server serve --dev

# Simple HTTP server (no API, static files only)
karenina-server serve --simple
```

### Using the API

Once the server is running, the API will be available at `http://localhost:8080/api/`.

#### Key Endpoints

- `POST /api/files/upload` - Upload question files
- `POST /api/generation/start` - Start answer template generation
- `GET /api/generation/status/{job_id}` - Check generation progress
- `POST /api/verification/start` - Start benchmark verification
- `GET /api/verification/status/{job_id}` - Check verification progress
- `POST /api/chat/message` - Send chat messages to LLMs

#### Example Usage

```python
import httpx

# Upload a file
with open("questions.xlsx", "rb") as f:
    response = httpx.post(
        "http://localhost:8080/api/files/upload",
        files={"file": f}
    )

# Start template generation
response = httpx.post(
    "http://localhost:8080/api/generation/start",
    json={
        "questions_data": {...},
        "config": {
            "provider": "openai",
            "model_name": "gpt-4",
            "api_key": "your-key"
        }
    }
)
job_id = response.json()["job_id"]

# Check progress
status = httpx.get(f"http://localhost:8080/api/generation/status/{job_id}")
```

## Configuration

### Environment Variables

Set these environment variables for default LLM configurations:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Google (Gemini)
export GOOGLE_API_KEY="your-google-key"

# Anthropic (Claude)
export ANTHROPIC_API_KEY="your-anthropic-key"

# OpenRouter
export OPENROUTER_API_KEY="your-openrouter-key"
```

### API Configuration

The server accepts various configuration options through request bodies:

```json
{
  "provider": "openai",
  "model_name": "gpt-4",
  "api_key": "your-key",
  "temperature": 0.1,
  "interface": "chat"
}
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/karenina-server.git
cd karenina-server

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=karenina_server

# Run specific test file
pytest tests/test_api.py
```

### API Documentation

When the server is running, interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/karenina_server
```

## Architecture

### Components

- **CLI Interface** (`cli.py`): Command-line entry point
- **FastAPI Server** (`server.py`): Main application and route setup
- **API Handlers** (`api/`): Modular endpoint implementations
- **Services** (`services/`): Async job management for long operations

### Service Architecture

The server uses a service-oriented architecture:

```
┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│  API Handlers   │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│    Services     │────│  Karenina Core  │
│ (Async Jobs)    │    │    Library      │
└─────────────────┘    └─────────────────┘
```

### Job Management

Long-running operations use async job management:

1. Client starts job via API
2. Server returns job ID immediately
3. Client polls job status endpoint
4. Server provides progress updates
5. Client retrieves results when complete

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install karenina-server

EXPOSE 8080
CMD ["karenina-server", "serve", "--host", "0.0.0.0"]
```

### Systemd Service

```ini
[Unit]
Description=Karenina Server
After=network.target

[Service]
Type=simple
User=karenina
WorkingDirectory=/opt/karenina
ExecStart=/usr/local/bin/karenina-server serve --host 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- **[karenina](https://github.com/yourusername/karenina)**: Core benchmarking library
- **[karenina-gui](https://github.com/yourusername/karenina-gui)**: React frontend application