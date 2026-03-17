"""Simple test runner for Research2Text."""

import sys
import subprocess
from pathlib import Path


def run_unit_tests():
    """Run unit tests."""
    print("=" * 60)
    print("Running Unit Tests")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
        capture_output=False,
        text=True
    )
    return result.returncode


def run_integration_tests():
    """Run integration tests."""
    print("\n" + "=" * 60)
    print("Running Integration Tests")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/integration/", "-v", "--tb=short"],
        capture_output=False,
        text=True
    )
    return result.returncode


def run_specific_test(test_path):
    """Run a specific test file."""
    print(f"\n{'=' * 60}")
    print(f"Running {test_path}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"],
        capture_output=False,
        text=True
    )
    return result.returncode


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running All Tests")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=False,
        text=True
    )
    return result.returncode


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Research2Text tests")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all", "specific"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Specific test file to run (with --type=specific)"
    )

    args = parser.parse_args()

    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    if args.type == "unit":
        exit_code = run_unit_tests()
    elif args.type == "integration":
        exit_code = run_integration_tests()
    elif args.type == "specific":
        if not args.file:
            print("Error: --file required with --type=specific")
            sys.exit(1)
        exit_code = run_specific_test(args.file)
    else:
        exit_code = run_all_tests()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
