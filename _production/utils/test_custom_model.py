"""Test script for verifying custom model integration.

This script tests various functionalities of the custom model integration:
- Basic text generation
- Structured data extraction
- Error handling and retries
"""

import logging
import sys
from pathlib import Path

from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from _production.utils.custom_model import (
    get_custom_model_client,
    get_structured_model_client,
)
from _production.utils.instructor_wrapper import from_custom_model
from _production.utils.llm import (
    job_post_detection,
    job_post_parsing,
    match_cv_with_job,
)


# Test response models
class UserDetail(BaseModel):
    """User detail information model."""

    name: str
    age: int


class JobPost(BaseModel):
    """Job post detection model."""

    is_job_description: bool


def test_basic_generation() -> None:
    """Test basic text generation without structured output."""
    print("\nTesting basic text generation...")

    client = get_custom_model_client()

    try:
        # Test with a simple prompt
        response = client.generate(
            prompt="What is the capital of France?",
            temperature=0.7,
            max_tokens=64,
        )
        print(f"Response: {response}")

        # Test with different temperature
        high_temp_response = client.generate(
            prompt="Write a creative first line of a sci-fi story",
            temperature=1.0,
            max_tokens=64,
        )
        print(f"\nHigh temperature response: {high_temp_response}")

        # Test with different temperature
        low_temp_response = client.generate(
            prompt="Write a creative first line of a sci-fi story",
            temperature=0.1,
            max_tokens=64,
        )
        print(f"\nLow temperature response: {low_temp_response}")

    except Exception as e:
        print(f"Error in basic generation: {e}")
        raise


def test_structured_generation() -> None:
    """Test structured data extraction."""
    print("\nTesting structured data extraction...")

    structured_client = get_structured_model_client()

    try:
        # Extract structured data using pydantic model
        result = structured_client.structured_generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured information.",
                },
                {
                    "role": "user",
                    "content": "Extract structured data from this text: Jason is 30 years old",
                },
            ],
            response_model=UserDetail,
            temperature=0.7,
        )

        print(f"Structured result: {result}")
        print(f"Name: {result.name}")
        print(f"Age: {result.age}")

    except Exception as e:
        print(f"Error in structured generation: {e}")
        raise


def test_instructor_integration() -> None:
    """Test instructor-compatible client."""
    print("\nTesting instructor client integration...")

    client = from_custom_model()

    try:
        # Use instructor client to extract structured data
        result = client.messages.create(
            model="custom-model",  # Model name is ignored
            system="You are a helpful assistant that extracts structured information.",
            messages=[
                {"role": "user", "content": "Extract data from: Jason is 30 years old"}
            ],
            response_model=UserDetail,
            max_tokens=64,
            temperature=0.0,
        )

        print(f"Instructor result: {result}")
        print(f"Name: {result.name}")
        print(f"Age: {result.age}")

    except Exception as e:
        print(f"Error in instructor integration: {e}")
        raise


def test_job_post_functions() -> None:
    """Test job post detection and parsing functions."""
    print("\nTesting job post functions...")

    # Sample job post
    sample_job_post = """
    Senior Backend Engineer

    Company: TechCorp Solutions
    Location: Berlin, Germany (Hybrid)
    Department: Engineering

    About the Role:
    We're seeking an experienced Backend Engineer to join our platform team.

    Required Skills:
    - 5+ years Python development experience
    - Strong expertise in Django and FastAPI
    - Advanced PostgreSQL and Redis knowledge
    - Experience with Docker and Kubernetes
    - Microservices architecture design
    - AWS cloud services (ECS, Lambda)

    Benefits:
    - Competitive salary: €85K-120K
    - Flexible hybrid work arrangement
    - Professional development budget
    """

    try:
        # Test job post detection
        print("\nTesting job_post_detection...")
        is_job_post = job_post_detection(sample_job_post)
        print(f"Is job post: {is_job_post}")

        # Test job post parsing
        print("\nTesting job_post_parsing...")
        parsed_job = job_post_parsing(sample_job_post)
        print("Parsed job post:")
        if parsed_job:
            for key, value in parsed_job.items():
                print(f"{key}: {value}")
        else:
            print("Failed to parse job post")

    except Exception as e:
        print(f"Error in job post functions: {e}")
        raise


def test_cv_matching() -> None:
    """Test CV-to-job matching."""
    print("\nTesting CV matching...")

    # Sample CV
    sample_cv = """
    John Doe
    Senior Software Engineer

    Experience:
    - 7 years Python development
    - Lead backend architect for high-scale systems
    - Team leadership experience managing 5-person teams
    - Microservices architecture design and implementation

    Technical Skills:
    - Languages: Python, Go, TypeScript
    - Frameworks: Django, FastAPI, React
    - Databases: PostgreSQL, MongoDB, Redis
    - Cloud: AWS (ECS, Lambda, S3), Docker, Kubernetes
    - Tools: Git, CI/CD, Prometheus, Grafana

    Certifications:
    - AWS Solutions Architect Professional
    - Kubernetes CKA
    """

    # Sample job post
    sample_job_post = """
    Senior Backend Engineer

    Company: TechCorp Solutions
    Location: Berlin, Germany (Hybrid)
    Department: Engineering

    About the Role:
    We're seeking an experienced Backend Engineer to join our platform team.

    Required Skills:
    - 5+ years Python development experience
    - Strong expertise in Django and FastAPI
    - Advanced PostgreSQL and Redis knowledge
    - Experience with Docker and Kubernetes
    - Microservices architecture design
    - AWS cloud services (ECS, Lambda)

    Benefits:
    - Competitive salary: €85K-120K
    - Flexible hybrid work arrangement
    - Professional development budget
    """

    try:
        # Test CV matching
        match_score = match_cv_with_job(sample_cv, sample_job_post)
        print(f"CV match score: {match_score}")
    except Exception as e:
        print(f"Error in CV matching: {e}")
        raise


def main() -> None:
    """Run all tests."""
    try:
        test_basic_generation()
        test_structured_generation()
        test_instructor_integration()
        test_job_post_functions()
        test_cv_matching()

        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
