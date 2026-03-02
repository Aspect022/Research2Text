@echo off
title Research2Code-GenAI
echo.
echo  ============================================
echo   Research2Code-GenAI Launcher
echo  ============================================
echo.

:: Navigate to the project root
cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+.
    pause
    exit /b 1
)

:: Create data directories if they don't exist
if not exist "data\raw_pdfs" mkdir "data\raw_pdfs"
if not exist "data\raw_texts" mkdir "data\raw_texts"
if not exist "outputs" mkdir "outputs"

echo.
echo  Choose what to run:
echo.
echo    [1] Launch Streamlit UI  (recommended)
echo    [2] Run Pipeline Tests
echo    [3] Run Pipeline on a PDF (CLI)
echo    [4] Check Dependencies
echo    [5] Install Dependencies
echo.

set /p choice="  Enter choice (1-5): "

if "%choice%"=="1" goto streamlit
if "%choice%"=="2" goto tests
if "%choice%"=="3" goto cli
if "%choice%"=="4" goto checkdeps
if "%choice%"=="5" goto install
echo Invalid choice.
pause
exit /b 1

:streamlit
echo.
echo  Starting Streamlit UI...
echo  Open http://localhost:8501 in your browser
echo.
cd src
streamlit run app_streamlit.py --server.headless true
goto end

:tests
echo.
echo  Running Multi-Agent Pipeline Tests...
echo.
python tests\test_multiagent_pipeline.py
echo.
echo  Tests complete.
pause
goto end

:cli
echo.
set /p pdf_path="  Enter PDF path (or drag-and-drop): "
if "%pdf_path%"=="" (
    echo No PDF provided.
    pause
    goto end
)
echo.
echo  Running multi-agent pipeline on: %pdf_path%
echo.
cd src
python -c "from paper_to_code_multiagent import run_paper_to_code; run_paper_to_code('%pdf_path%', use_multiagent=True)"
echo.
echo  Pipeline complete. Check the outputs/ folder.
pause
goto end

:checkdeps
echo.
echo  Checking dependencies...
echo.
python -c "import streamlit; print(f'  [OK] streamlit {streamlit.__version__}')" 2>nul || echo  [MISSING] streamlit
python -c "import chromadb; print(f'  [OK] chromadb {chromadb.__version__}')" 2>nul || echo  [MISSING] chromadb
python -c "import pydantic; print(f'  [OK] pydantic {pydantic.__version__}')" 2>nul || echo  [MISSING] pydantic
python -c "import fitz; print(f'  [OK] PyMuPDF {fitz.version[0]}')" 2>nul || echo  [MISSING] PyMuPDF (pip install pymupdf)
python -c "import ollama; print('  [OK] ollama')" 2>nul || echo  [MISSING] ollama (pip install ollama)
python -c "import sympy; print(f'  [OK] sympy {sympy.__version__}')" 2>nul || echo  [MISSING] sympy
python -c "import torch; print(f'  [OK] torch {torch.__version__}')" 2>nul || echo  [OPTIONAL] torch (needed for code execution)
python -c "import magic_pdf; print('  [OK] MinerU')" 2>nul || echo  [OPTIONAL] MinerU (pip install mineru[all])
python -c "import olmocr; print('  [OK] olmOCR')" 2>nul || echo  [OPTIONAL] olmOCR (pip install olmocr[gpu])
echo.
pause
goto end

:install
echo.
echo  Installing core dependencies...
echo.
pip install streamlit chromadb pydantic pymupdf ollama sympy torch numpy
echo.
echo  Done. Run option [4] to verify.
pause
goto end

:end
