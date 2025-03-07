#!/usr/bin/env python
"""
Script to run the CV matching tests locally.

Usage:
    python run_tests.py [--show-logs]
"""

import argparse
import logging
import os
import sys

import pytest


def main():
    """Run the CV matching tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run CV matching tests")
    parser.add_argument(
        "--show-logs", action="store_true", help="Show logs during test execution"
    )
    args = parser.parse_args()

    # Configure logging
    if not args.show_logs:
        # Suppress logs by default
        logging.getLogger().setLevel(logging.CRITICAL)

    # Add the current directory to the path so that imports work
    sys.path.insert(0, os.path.abspath("."))

    # Run the tests
    print("Running CV matching tests...")
    result = pytest.main(["-v", "tests/test_cv_matching.py"])

    # Return the exit code
    return result


if __name__ == "__main__":
    sys.exit(main())
