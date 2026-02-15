# GitHub Actions CI

This project uses GitHub Actions for continuous integration.

## Workflows

### CI Pipeline (`.github/workflows/ci.yml`)

The CI pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual workflow dispatch

#### Jobs

1. **Code Quality & Linting**
   - Runs `flake8` for Python linting
   - Checks code formatting with `black`
   - Checks import sorting with `isort`

2. **Unit & Integration Tests**
   - Runs pytest tests in the `tests/` directory
   - Executes sentiment routing tests (if dependencies available)
   - Includes test coverage reporting

3. **Build Verification**
   - Verifies all major dependencies can be imported
   - Checks Python syntax of main files
   - Validates project structure

4. **Security Scan**
   - Runs `safety` to check for known security vulnerabilities in dependencies

5. **CI Summary**
   - Aggregates results from all jobs
   - Provides overall pass/fail status

## Running Tests Locally

### Install Development Dependencies

```bash
pip install pytest pytest-cov pytest-asyncio httpx flake8 black isort safety
```

### Run Linting

```bash
# Check for syntax errors and undefined names
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=myenv,venv,.git,__pycache__,.github

# Check code style
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics --exclude=myenv,venv,.git,__pycache__,.github

# Check formatting
black --check --line-length=120 --exclude='myenv|venv|\.git' .

# Check import sorting
isort --check-only --profile black --line-length=120 --skip myenv --skip venv .
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
python test_sentiment_routing.py
```

### Security Check

```bash
safety check
```

## Configuration Files

- **`.github/workflows/ci.yml`**: GitHub Actions workflow configuration
- **`pyproject.toml`**: Configuration for pytest, black, and isort
- **`requirements.txt`**: Python dependencies

## Notes

- Some tests require API keys (Groq, LangChain) and may be skipped in CI
- Tests requiring a running API server are marked as `continue-on-error` in CI
- The workflow uses dependency caching to speed up runs
