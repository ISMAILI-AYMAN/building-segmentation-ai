# Contributing to Building Segmentation AI

Thank you for your interest in contributing to the Building Segmentation AI project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/yourusername/building-segmentation-ai.git
cd building-segmentation-ai
```

### 2. Set Up Development Environment
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes
- Follow the coding standards below
- Add tests for new functionality
- Update documentation as needed

### 5. Test Your Changes
```bash
# Run tests
python -m pytest

# Check code style
black --check .
flake8 .
mypy .

# Test the application
python main.py --help
```

### 6. Commit and Push
```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### 7. Create a Pull Request
- Go to your fork on GitHub
- Click "New Pull Request"
- Fill out the PR template
- Submit the PR

## 📋 Coding Standards

### Python Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Keep functions under 50 lines
- Use descriptive variable names
- Add docstrings to functions and classes

### Code Formatting
```bash
# Format code with black
black .

# Sort imports
isort .
```

### Testing
- Write unit tests for new functionality
- Aim for >80% code coverage
- Use pytest for testing
- Mock external dependencies

### Documentation
- Update README.md if adding new features
- Add docstrings to new functions
- Update API documentation if changing endpoints

## 🏗️ Project Structure

### Core Components
- `api/` - FastAPI backend
- `frontend/` - Flask web interface
- `inference/` - Model inference engine
- `training/` - Training pipeline
- `scripts/` - Utility scripts
- `docker_config/` - Docker configurations

### Adding New Features

#### 1. Model Improvements
- Modify `inference/inference_engine.py`
- Add new model architectures in `training/models/`
- Update `requirements.txt` if adding new dependencies

#### 2. API Extensions
- Add new endpoints in `api/app.py`
- Create new Pydantic models in `api/models.py`
- Update API documentation

#### 3. Frontend Enhancements
- Add new templates in `frontend/templates/`
- Update static files in `frontend/static/`
- Modify `frontend/app.py` for new routes

#### 4. Docker Configuration
- Update Dockerfiles in `docker_config/`
- Modify docker-compose files as needed
- Test with `./deploy.sh dev`

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_inference.py

# Run with coverage
python -m pytest --cov=.

# Run integration tests
python -m pytest tests/test_integration.py
```

### Writing Tests
```python
# Example test structure
def test_inference_engine():
    """Test inference engine functionality."""
    engine = InferenceEngine()
    result = engine.process_image("test_image.jpg")
    assert result is not None
    assert "segmentation" in result
```

## 📝 Commit Message Guidelines

Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```
feat(api): add batch inference endpoint
fix(frontend): resolve image display issue
docs(readme): update installation instructions
test(inference): add unit tests for model loading
```

## 🚀 Release Process

### Versioning
We use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Creating a Release
1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create a release tag
4. Build and test Docker images
5. Update documentation

## 🐛 Bug Reports

When reporting bugs, please include:
- Operating system and version
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
- Screenshots if applicable

## 💡 Feature Requests

When requesting features, please include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Any relevant research or references

## 📞 Getting Help

- Create an issue on GitHub
- Check existing issues and discussions
- Review the documentation
- Join our community discussions

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to Building Segmentation AI! 🚀
