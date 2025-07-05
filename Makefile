.PHONY: help install dev test test-cov test-changed lint format type-check dead-code clean build serve version release release-dry

help:
	@echo "Available commands:"
	@echo "  make install         Install the package"
	@echo "  make dev             Install the package in development mode with all extras"
	@echo "  make test            Run tests"
	@echo "  make test-cov        Run tests with coverage"
	@echo "  make test-changed    Run only tests affected by changes (testmon)"
	@echo "  make lint            Run linting"
	@echo "  make format          Format code"
	@echo "  make type-check      Run type checking"
	@echo "  make dead-code       Find dead code with vulture"
	@echo "  make serve           Start the server in development mode"
	@echo "  make clean           Clean build artifacts"
	@echo "  make build           Build distribution packages"
	@echo "  make version         Show next version based on commits"
	@echo "  make release-dry     Dry run semantic release (show what would happen)"
	@echo "  make release         Create semantic release (bump version, tag, build, publish)"

install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"
	pre-commit install

test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=karenina_server --cov-report=html --cov-report=term

test-changed:
	uv run pytest --testmon -v

lint:
	uv run ruff check src/karenina_server tests

format:
	uv run ruff format src/karenina_server tests

type-check:
	uv run mypy src/karenina_server

dead-code:
	uv run vulture src/karenina_server

serve:
	karenina-server serve --dev

serve-simple:
	karenina-server serve --simple

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

build:
	uv pip install build
	python -m build

check: lint type-check dead-code test

version:
	@echo "Analyzing commits for next version..."
	uv run semantic-release version --print

release-dry:
	@echo "Dry run - showing what would be released..."
	uv run semantic-release version --dry-run

release: check
	@echo "Creating semantic release for karenina-server..."
	@echo "This will:"
	@echo "  1. Analyze git commits using conventional commit format"
	@echo "  2. Determine the next version (patch/minor/major)"
	@echo "  3. Update version in files"
	@echo "  4. Create git tag"
	@echo "  5. Build distribution packages"
	@echo "  6. Upload to PyPI"
	@echo "  7. Create GitHub release"
	@echo ""
	@read -p "Continue? [y/N]: " confirm && [ "$$confirm" = "y" ] || exit 1
	uv run semantic-release version

all: clean install dev check
