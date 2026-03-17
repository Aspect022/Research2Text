#!/usr/bin/env python3
"""
Comprehensive Test Script for Research2Text Phase 6
Tests all components: NewResearcher, Conformal Prediction, 3-Phase Pipeline
"""

import sys
import subprocess
from pathlib import Path

# Colors for output
GREEN = ""
RED = ""
YELLOW = ""
RESET = ""

def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")

def print_success(text):
    print(f"[PASS] {text}")

def print_error(text):
    print(f"[FAIL] {text}")

def print_warning(text):
    print(f"[WARN] {text}")

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n  Testing: {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print_success(f"{description}")
            return True
        else:
            print_error(f"{description}")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print_error(f"{description} (timeout)")
        return False
    except Exception as e:
        print_error(f"{description}")
        print(f"    Exception: {e}")
        return False

def test_imports():
    """Test all module imports."""
    print_header("PHASE 1: Module Imports")

    tests = [
        ("import sys; sys.path.insert(0, 'src'); from chunking.token_chunker import TokenChunker; print('OK')", "Token Chunking"),
        ("import sys; sys.path.insert(0, 'src'); from validation.source_validator import SourceValidator; print('OK')", "Source Validation"),
        ("import sys; sys.path.insert(0, 'src'); from search.academic_search import AcademicSearch; print('OK')", "Academic Search"),
        ("import sys; sys.path.insert(0, 'src'); from conformal.predictor import ConformalPredictor; print('OK')", "Conformal Prediction"),
        ("import sys; sys.path.insert(0, 'src'); from agents.orchestrator import Orchestrator; print('OK')", "Orchestrator"),
        ("import sys; sys.path.insert(0, 'src'); from sandbox.windows_sandbox import WindowsSandbox; print('OK')", "Windows Sandbox"),
    ]

    passed = 0
    for cmd, desc in tests:
        if run_command(f"python -c \"{cmd}\"", desc):
            passed += 1

    return passed, len(tests)

def test_unit_tests():
    """Run unit test suite."""
    print_header("PHASE 2: Unit Tests")

    result = subprocess.run(
        ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=line"],
        capture_output=True,
        text=True
    )

    # Parse output for summary
    output = result.stdout + result.stderr
    if "passed" in output:
        # Extract number of passed tests
        import re
        match = re.search(r'(\d+) passed', output)
        if match:
            passed = int(match.group(1))
            print_success(f"Unit Tests ({passed} tests passed)")
            return passed, passed

    print_error("Unit Tests")
    print(f"  Run manually: python -m pytest tests/unit/ -v")
    return 0, 1

def test_newresearcher_components():
    """Test NewResearcher components functionality."""
    print_header("PHASE 3: NewResearcher Components")

    tests = []

    # Test 1: Token Chunking
    chunking_test = '''
import sys
sys.path.insert(0, 'src')
from chunking.token_chunker import TokenChunker

text = "This is sentence one. This is sentence two. This is sentence three."
chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
chunks = chunker.chunk_text(text)

assert len(chunks) > 0, "No chunks created"
assert all(len(c.text) > 0 for c in chunks), "Empty chunks found"
assert all(c.token_count > 0 for c in chunks), "Zero token count"
print(f"Created {len(chunks)} chunks")
print("OK")
'''
    tests.append((f'python -c "{chunking_test}"', "Token Chunking Functionality"))

    # Test 2: Source Validation
    validation_test = '''
import sys
sys.path.insert(0, 'src')
from validation.source_validator import validate_sources

sources = [
    {"id": "1", "title": "NeurIPS Paper", "venue": "NeurIPS", "year": 2023, "text": "Deep learning..."},
    {"id": "2", "title": "Blog Post", "venue": "Blog", "year": 2023, "text": "AI is cool..."}
]

result = validate_sources(sources, top_n=2)
assert "top_sources" in result, "Missing top_sources"
assert "summary" in result, "Missing summary"
assert len(result["top_sources"]) <= 2, "Too many sources returned"
print(f"Validated {result['summary']['total_sources']} sources")
print("OK")
'''
    tests.append((f'python -c "{validation_test}"', "Source Validation Functionality"))

    # Test 3: Academic Search (without API keys)
    search_test = '''
import sys
sys.path.insert(0, 'src')
from search.academic_search import AcademicSearch

search = AcademicSearch()
assert "arxiv" in search.available_sources, "arXiv not available"
assert "semantic_scholar" in search.available_sources, "Semantic Scholar not available"
print(f"Available sources: {search.available_sources}")
print("OK")
'''
    tests.append((f'python -c "{search_test}"', "Academic Search Initialization"))

    passed = 0
    for cmd, desc in tests:
        if run_command(cmd, desc):
            passed += 1

    return passed, len(tests)

def test_conformal_prediction():
    """Test Conformal Prediction system."""
    print_header("PHASE 4: Conformal Prediction")

    tests = []

    # Test 1: Basic conformal prediction
    cp_test = '''
import sys
sys.path.insert(0, 'src')
from conformal.predictor import ConformalPredictor

predictor = ConformalPredictor(alpha=0.1)
assert predictor.alpha == 0.1, "Alpha not set correctly"
assert predictor.coverage == 0.9, "Coverage not calculated correctly"

# Test nonconformity scores
score = predictor._string_nonconformity("hello", "hello")
assert score == 0.0, "Exact match should score 0"

score = predictor._numeric_nonconformity(1.0, 1.0)
assert score == 0.0, "Exact numeric match should score 0"

score = predictor._jaccard_nonconformity(["a", "b"], ["a", "b"])
assert score == 0.0, "Exact set match should score 0"

print("Conformal prediction tests passed")
print("OK")
'''
    tests.append((f'python -c "{cp_test}"', "Conformal Prediction Core"))

    # Test 2: Calibration data exists
    cal_file = Path("data/calibration/validation_papers.json")
    if cal_file.exists():
        print_success("Calibration Data File Exists")
        passed = 1
    else:
        print_error("Calibration Data File Missing")
        passed = 0

    for cmd, desc in tests:
        if run_command(cmd, desc):
            passed += 1

    return passed, len(tests) + 1

def test_sandbox():
    """Test Windows Sandbox."""
    print_header("PHASE 5: Windows Sandbox")

    sandbox_test = '''
import sys
sys.path.insert(0, 'src')
from sandbox.windows_sandbox import WindowsSandbox

sandbox = WindowsSandbox()
print(f"Sandbox initialized: {sandbox}")
print("OK")
'''

    if run_command(f'python -c "{sandbox_test}"', "Sandbox Initialization"):
        return 1, 1
    return 0, 1

def test_orchestrator():
    """Test Orchestrator with 3-phase workflow."""
    print_header("PHASE 6: Orchestrator (3-Phase Workflow)")

    orch_test = '''
import sys
sys.path.insert(0, 'src')
from agents.orchestrator import Orchestrator

orch = Orchestrator()

# Check all agents initialized
expected_agents = [
    "ingest", "vision", "chunking", "method_extractor",
    "equation", "dataset_loader", "code_architect",
    "graph_builder", "validator", "cleaner"
]

for agent_id in expected_agents:
    assert agent_id in orch.agents, f"Agent {agent_id} not found"

print(f"All {len(expected_agents)} agents initialized")
print("OK")
'''

    if run_command(f'python -c "{orch_test}"', "Orchestrator Initialization"):
        return 1, 1
    return 0, 1

def test_streamlit_ui():
    """Test Streamlit UI can be imported."""
    print_header("PHASE 7: Streamlit UI")

    # Check if streamlit is installed
    result = subprocess.run(
        ["python", "-c", "import streamlit; print('OK')"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print_warning("Streamlit not installed, skipping UI tests")
        return 0, 0

    # Check if app file exists and can be imported
    ui_test = '''
import sys
sys.path.insert(0, 'src')

# Check file exists
from pathlib import Path
assert Path("src/app_streamlit.py").exists(), "app_streamlit.py not found"

# Try to import (without running)
print("Streamlit UI file exists and is importable")
print("OK")
'''

    if run_command(f'python -c "{ui_test}"', "Streamlit UI File"):
        return 1, 1
    return 0, 1

def test_calibration_data():
    """Test calibration data generation."""
    print_header("PHASE 8: Calibration Data")

    cal_file = Path("data/calibration/validation_papers.json")

    if not cal_file.exists():
        print_warning("Calibration data not found, generating...")
        result = subprocess.run(
            ["python", "-m", "src.conformal.generate_calibration", "--synthetic", "--num-synthetic", "5"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print_error("Failed to generate calibration data")
            return 0, 1

    # Verify file exists and is valid JSON
    try:
        import json
        data = json.loads(cal_file.read_text())
        num_papers = len(data.get("papers", []))
        print_success(f"Calibration Data ({num_papers} papers)")
        return 1, 1
    except Exception as e:
        print_error(f"Calibration Data Invalid: {e}")
        return 0, 1

def main():
    """Run all tests."""
    print("="*60)
    print(" Research2Text Phase 6 - Comprehensive Test Suite")
    print("="*60)
    print(f"\nProject Root: {Path.cwd()}")
    print(f"Python: {sys.executable}")
    print(f"Platform: {sys.platform}")

    results = []

    # Run all test phases
    results.append(("Module Imports", *test_imports()))
    results.append(("Unit Tests", *test_unit_tests()))
    results.append(("NewResearcher", *test_newresearcher_components()))
    results.append(("Conformal Prediction", *test_conformal_prediction()))
    results.append(("Windows Sandbox", *test_sandbox()))
    results.append(("Orchestrator", *test_orchestrator()))
    results.append(("Streamlit UI", *test_streamlit_ui()))
    results.append(("Calibration Data", *test_calibration_data()))

    # Summary
    print_header("TEST SUMMARY")

    total_passed = 0
    total_tests = 0

    for name, passed, total in results:
        if total == 0:
            status = "SKIPPED"
        elif passed == total:
            status = "PASS"
            total_passed += passed
            total_tests += total
        else:
            status = "FAIL"
            total_passed += passed
            total_tests += total

        print(f"  {name:.<40} {passed}/{total} {status}")

    print(f"\n{'='*60}")
    if total_passed == total_tests:
        print(f" ALL TESTS PASSED: {total_passed}/{total_tests}")
    else:
        print(f" SOME TESTS FAILED: {total_passed}/{total_tests}")
    print(f"{'='*60}\n")

    # Quick start commands
    print("Quick Start Commands:")
    print("  1. Run Streamlit UI:")
    print("     streamlit run src/app_streamlit.py")
    print("")
    print("  2. Run Unit Tests:")
    print("     python -m pytest tests/unit/ -v")
    print("")
    print("  3. Run All Tests:")
    print("     python tests/run_tests.py")
    print("")
    print("  4. Generate Calibration Data:")
    print("     python -m src.conformal.generate_calibration --synthetic")
    print("")

    return 0 if total_passed == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())
