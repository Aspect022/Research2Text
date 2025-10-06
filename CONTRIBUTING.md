# Contributing to Research2Text

First off, thank you for considering contributing to Research2Text! 🎉 Your help is essential for making this tool even better for the research community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please be respectful, inclusive, and constructive in all interactions.

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Sample files or code** (if applicable)
- **Screenshots** (if relevant)

### 💡 Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- **Clear title and description**
- **Use case and motivation**
- **Detailed explanation** of the feature
- **Potential implementation approach**
- **Mockups or examples** (if applicable)

### 🔧 Code Contributions

Areas where contributions are especially welcome:

- **Performance improvements**
- **New embedding models support**
- **UI/UX enhancements**
- **Documentation improvements**
- **Test coverage expansion**
- **Bug fixes**
- **Integration with other tools**

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/yourusername/research2text.git
cd research2text
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if available)
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks (Recommended)

```bash
# Install pre-commit
pip install pre-commit

# Set up pre-commit hooks
pre-commit install
```

### 4. Set Up Ollama

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai/ for installation instructions

# Pull required models
ollama pull gpt-oss:120b-cloud
ollama pull mistral:7b  # Lightweight alternative
```

### 5. Verify Setup

```bash
# Test the application
streamlit run src/app_streamlit.py

# Run existing tests (if available)
python -m pytest tests/
```

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-new-embedding-model`
- `bugfix/fix-pdf-parsing-error`
- `docs/update-installation-guide`

### 2. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add comments for complex logic
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run any existing tests
python -m pytest tests/

# Test manually with the Streamlit app
streamlit run src/app_streamlit.py

# Test command-line interfaces
python src/ingest_pdf.py
python src/query_rag.py "test query"
```

### 4. Update Documentation

- Update README.md if needed
- Add docstrings to new functions
- Update type hints
- Add inline comments for complex code

### 5. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "Add support for custom embedding models

- Add configuration option for embedding model selection
- Update documentation with new model options
- Add error handling for unsupported models"
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- **Clear title and description**
- **Reference to related issues** (if any)
- **Description of changes made**
- **Testing performed**
- **Screenshots** (if UI changes)

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** where possible
- Write **docstrings** for functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Example Function

```python
def chunk_text_by_words(
    text: str, 
    chunk_size_words: int = 700, 
    overlap_words: int = 100
) -> List[str]:
    """
    Split text into overlapping chunks based on word count.
    
    Args:
        text: Input text to be chunked
        chunk_size_words: Maximum words per chunk
        overlap_words: Number of overlapping words between chunks
        
    Returns:
        List of text chunks with specified overlap
        
    Raises:
        ValueError: If chunk_size_words <= 0
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    
    # Implementation here...
    return chunks
```

### File Structure

```
src/
├── __init__.py
├── app_streamlit.py      # Main Streamlit app
├── index_documents.py    # Document indexing logic
├── ingest_pdf.py        # PDF processing
├── query_rag.py         # RAG query handling
└── utils.py             # Utility functions

tests/
├── __init__.py
├── test_utils.py
├── test_query_rag.py
└── fixtures/            # Test data

docs/
├── api_reference.md
├── user_guide.md
└── development.md
```

### Dependencies

- Keep dependencies minimal
- Pin versions in `requirements.txt`
- Document any new dependencies in PR description
- Prefer well-maintained, popular libraries

## Issue Guidelines

### Bug Reports

Use the bug report template and include:

```markdown
## Bug Description
Clear description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 11, macOS 13, Ubuntu 22.04]
- Python version: [e.g., 3.11.0]
- Package versions: [copy from pip list]

## Additional Context
Any other relevant information
```

### Feature Requests

Use the feature request template:

```markdown
## Feature Description
Clear description of the proposed feature

## Motivation
Why is this feature needed?

## Use Case
How would you use this feature?

## Implementation Ideas
Any thoughts on how this could be implemented?

## Alternatives
What alternatives have you considered?
```

## Community

### Getting Help

- 📖 **Documentation**: Check the README and docs/ folder
- 🐛 **Issues**: Search existing issues first
- 💬 **Discussions**: Use GitHub Discussions for questions
- 📧 **Email**: Contact maintainers directly for sensitive issues

### Recognition

Contributors will be recognized in:

- **README acknowledgments**
- **Release notes** for significant contributions
- **Contributors page** (if applicable)

## Development Tips

### Testing Locally

```bash
# Test with sample PDFs
mkdir -p data/raw_pdfs
# Add sample PDFs to data/raw_pdfs/

# Process and test
python src/ingest_pdf.py
python src/index_documents.py
python src/query_rag.py "What is the main contribution?"

# Test Streamlit UI
streamlit run src/app_streamlit.py
```

### Debugging

```python
# Add debug prints
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debugger
import pdb; pdb.set_trace()

# Print ChromaDB collection info
collection = client.get_collection("research_papers")
print(f"Collection count: {collection.count()}")
```

### Common Issues

1. **ChromaDB path issues**: Ensure paths are absolute
2. **Ollama connection**: Check if Ollama is running (`ollama list`)
3. **Memory issues**: Use smaller models for development
4. **PDF parsing**: Test with various PDF formats

## Release Process

For maintainers:

1. **Version bump** in appropriate files
2. **Update CHANGELOG.md**
3. **Create release notes**
4. **Tag release** with semantic versioning
5. **Update documentation**

---

Thank you for contributing to Research2Text! Your efforts help make research more accessible to everyone. 🚀
