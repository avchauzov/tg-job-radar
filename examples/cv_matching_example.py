#!/usr/bin/env python
"""
Example script demonstrating how to use the CV matching functionality.

This script shows how to use the enhanced_cv_matching function to evaluate
a CV against a job post and get a match score.

Usage:
    python examples/cv_matching_example.py
"""

import logging
import os
import re
import sys
from typing import Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _production.utils.agents import enhanced_cv_matching

# Configure logging - set to INFO by default
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Set third-party loggers to a higher level to reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)


def extract_scores_from_text(text: str) -> dict[str, Any] | None:
    """
    Extract scores from a text response when JSON parsing fails.

    Args:
        text: The text response from the agent

    Returns:
        A dictionary with the extracted scores, or None if extraction fails
    """
    try:
        # Try to find experience match score
        experience_match = re.search(r"Experience Match:?\s*(\d+)(?:\/100)?", text)
        # Also try alternative format
        if not experience_match:
            experience_match = re.search(r"Experience Match \((\d+)(?:\/100)?\)", text)

        # Try to find skills match score
        skills_match = re.search(r"Skills Match:?\s*(\d+)(?:\/100)?", text)
        if not skills_match:
            skills_match = re.search(r"Skills Match \((\d+)(?:\/100)?\)", text)

        # Try to find soft skills match score
        soft_skills_match = re.search(r"Soft Skills Match:?\s*(\d+)(?:\/100)?", text)
        if not soft_skills_match:
            soft_skills_match = re.search(
                r"Soft Skills Match \((\d+)(?:\/100)?\)", text
            )

        # If we found all scores, return them
        if experience_match and skills_match and soft_skills_match:
            return {
                "experience_match": {"score": float(experience_match.group(1))},
                "skills_match": {"score": float(skills_match.group(1))},
                "soft_skills_match": {"score": float(soft_skills_match.group(1))},
            }

        # Try to find the final score directly
        final_score_match = re.search(
            r"Total (?:Weighted )?Score:?\s*(\d+(?:\.\d+)?)(?:\/100)?", text
        )
        if final_score_match:
            score = float(final_score_match.group(1))
            return {"final_score": score}

        return None
    except Exception as e:
        logging.debug(f"Failed to extract scores from text: {e}")
        return None


def cv_matching_with_fallback(cv_content: str, job_post: str) -> float | None:
    """
    Perform CV matching with automatic fallback to direct LLM approach if agent fails.

    Args:
        cv_content: The CV content
        job_post: The job post content

    Returns:
        A match score between 0 and 100, or None if calculation fails
    """
    # Temporarily reduce logging level for root logger to suppress errors
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.CRITICAL)

    try:
        # First try the agent-based approach
        match_score = enhanced_cv_matching(cv_content, job_post)

        # If it succeeds, return the result
        if match_score is not None:
            return match_score

        # If it fails, try the manual approach
        logging.info("Using direct LLM approach for CV matching...")
        return manual_cv_matching(cv_content, job_post)
    finally:
        # Restore original logging level
        root_logger.setLevel(original_level)


def manual_cv_matching(cv_content: str, job_post: str) -> float | None:
    """
    Direct LLM implementation that calculates the match score.

    Args:
        cv_content: The CV content
        job_post: The job post content

    Returns:
        A match score between 0 and 100, or None if calculation fails
    """
    from langchain_anthropic import ChatAnthropic

    from _production import LLM_BASE_MODEL

    try:
        # Create a direct LLM instance
        llm = ChatAnthropic(
            model_name=LLM_BASE_MODEL, temperature=0.0, timeout=120, stop=["```"]
        )

        # Make a direct request to the LLM
        response = llm.invoke(
            f"""You are a technical recruiter evaluating a candidate CV against a job post.

            Analyze the match between the CV and job post comprehensively, considering these aspects:

            1. Experience Match (40% weight in final score)
            2. Skills Match (45% weight in final score)
            3. Soft Skills Match (15% weight in final score)

            For each category, provide a score out of 100 and brief reasoning.

            At the end, calculate the final weighted score using these weights.

            CV:
            {cv_content}

            Job Post:
            {job_post}
            """
        )

        # Try to extract scores from the response
        if hasattr(response, "content") and isinstance(response.content, str):
            scores = extract_scores_from_text(response.content)
        else:
            logging.error("Unexpected response format from LLM")
            return None

        if scores:
            if "final_score" in scores:
                # If we extracted the final score directly, return it
                return int(scores["final_score"])
            elif all(
                key in scores
                for key in ["experience_match", "skills_match", "soft_skills_match"]
            ):
                # Calculate the weighted score
                experience_score = scores["experience_match"]["score"]
                skills_score = scores["skills_match"]["score"]
                soft_skills_score = scores["soft_skills_match"]["score"]

                final_score = int(
                    (skills_score * 0.45)
                    + (experience_score * 0.40)
                    + (soft_skills_score * 0.15)
                )
                return final_score

        return None
    except Exception as e:
        logging.error(f"Manual CV matching failed: {e}")
        return None


def main():
    """Run the CV matching example."""
    # Example CV content
    cv_content = """
    PROFESSIONAL SUMMARY
    Experienced Python developer with 5 years of experience in software development.
    Proficient in machine learning frameworks including TensorFlow and PyTorch.
    Strong background in data analysis and visualization.

    SKILLS
    - Python (5 years)
    - Machine Learning (3 years)
    - TensorFlow, PyTorch
    - Data Analysis
    - SQL, MongoDB
    - Git, Docker

    EXPERIENCE
    Senior Python Developer | ABC Tech | 2020-Present
    - Developed machine learning models for customer segmentation
    - Implemented data pipelines using Apache Airflow
    - Mentored junior developers and conducted code reviews

    Python Developer | XYZ Solutions | 2018-2020
    - Built RESTful APIs using Flask and FastAPI
    - Created data visualization dashboards with Plotly and Dash
    - Optimized database queries for improved performance

    EDUCATION
    Master of Science in Computer Science | University of Technology | 2018
    Bachelor of Science in Mathematics | State University | 2016

    CERTIFICATIONS
    - AWS Certified Developer Associate
    - TensorFlow Developer Certificate
    """

    # Example job post
    job_post = """
    Senior Python Developer

    We are looking for a Senior Python Developer with 3-5 years of experience to join our team.

    Requirements:
    - 3-5 years of experience in Python development
    - Experience with machine learning frameworks (TensorFlow, PyTorch)
    - Strong knowledge of data analysis and visualization
    - Experience with SQL and NoSQL databases
    - Familiarity with cloud platforms (AWS, GCP)

    Responsibilities:
    - Develop and maintain machine learning models
    - Implement data pipelines and ETL processes
    - Collaborate with data scientists and other developers
    - Mentor junior developers and conduct code reviews

    Education:
    - Bachelor's degree in Computer Science or related field required
    - Master's degree preferred

    We offer a competitive salary, flexible working hours, and a collaborative work environment.
    """

    # Run the CV matching with automatic fallback
    logging.info("Running CV matching...")
    match_score = cv_matching_with_fallback(cv_content, job_post)

    if match_score is not None:
        logging.info(f"Match score: {match_score}/100")

        # Interpret the score
        if match_score >= 90:
            logging.info(
                "Excellent match! This candidate exceeds the job requirements."
            )
        elif match_score >= 80:
            logging.info(
                "Strong match! This candidate meets most of the job requirements."
            )
        elif match_score >= 70:
            logging.info("Good match! This candidate meets the basic job requirements.")
        elif match_score >= 60:
            logging.info(
                "Fair match. This candidate meets some of the job requirements."
            )
        else:
            logging.info(
                "Poor match. This candidate may not be suitable for the position."
            )
    else:
        logging.error("Failed to calculate match score.")


if __name__ == "__main__":
    main()
