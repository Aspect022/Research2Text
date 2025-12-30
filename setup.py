from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="research2text",
    version="1.0.0",
    author="Research2Text Contributors",
    author_email="your.email@example.com",
    description="AI-powered research assistant for processing academic papers with RAG capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/research2text",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/research2text/issues",
        "Source": "https://github.com/yourusername/research2text",
        "Documentation": "https://github.com/yourusername/research2text/wiki",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
        "Topic :: Education",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "pre-commit>=2.20",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "research2text=app_streamlit:main",
            "r2t-ingest=ingest_pdf:main",
            "r2t-index=index_documents:main",
            "r2t-query=query_rag:main",
        ],
    },
    include_package_data=True,
    package_data={
        "research2text": [
            "data/raw_pdfs/.gitkeep",
            "data/raw_texts/.gitkeep",
            "data/chroma_db/.gitkeep",
        ],
    },
    keywords=[
        "research",
        "ai",
        "rag",
        "pdf",
        "academic",
        "papers",
        "embeddings",
        "streamlit",
        "chromadb",
        "ollama",
        "natural language processing",
        "information retrieval",
    ],
    zip_safe=False,
)
