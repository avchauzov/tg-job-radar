"""
Helper classes and functions for LLM operations.

This module contains:
- Pydantic models for structured data
- Token counting utilities
- Common helper functions
"""

import logging
from typing import Literal

import tiktoken
from pydantic import BaseModel, Field

from _production import MIN_CV_LENGTH
from _production.utils.exceptions import LLMInputError

MAX_TEXT_LENGTH = 64000


def validate_text_input(
    text: str, field_name: str, max_length: int = MAX_TEXT_LENGTH
) -> None:
    """Validate text input.

    Args:
        text: Input text to validate
        field_name: Name of the field for error messages
        max_length: Maximum allowed length

    Raises:
        LLMInputError: If validation fails
    """
    if not text or not text.strip():
        raise LLMInputError(f"{field_name} cannot be empty")
    if len(text) > max_length:
        raise LLMInputError(
            f"{field_name} exceeds maximum length of {max_length} characters"
        )


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text using tiktoken.

    Uses o200k_base encoding which is optimized for gpt-4.1-nano model.
    """
    try:
        encoding = tiktoken.get_encoding("o200k_base")  # Optimized for gpt-4.1-nano
        return len(encoding.encode(text))
    except Exception as e:
        logging.warning(f"Error counting tokens: {e!s}")
        return len(text) // 4  # Fallback: rough estimate of 4 chars per token


class CleanJobPost(BaseModel):
    """
    Structured representation of a cleaned job post with standardized fields.

    Contains normalized values for job details including title, seniority, location,
    and other key attributes needed for matching and analysis.
    """

    job_title: str | None = Field(
        default=None,
        description="Standardized job title",
        examples=["Senior Software Engineer", "Data Scientist"],
    )
    seniority_level: str | None = Field(
        default=None,
        description="Standardized seniority level",
        examples=["Senior", "Mid-level", "Junior"],
    )
    location: str | None = Field(
        default=None,
        description="Standardized location format",
        examples=["Berlin, Germany", "Remote"],
    )
    salary_range: str | None = Field(
        default=None,
        description="Standardized salary range",
        examples=["€85K-120K", "$100K-150K"],
    )
    company_name: str | None = Field(
        default=None,
        description="Standardized company name",
        examples=["TechCorp Solutions", "Startup Inc."],
    )
    skills: str | None = Field(
        default=None,
        description="Combined technical and professional skills, ordered by importance",
    )
    short_description: str | None = Field(
        default=None,
        description="Concise summary of the job role and key responsibilities",
        examples=[
            "Senior Backend Engineer role focusing on microservices architecture and cloud infrastructure"
        ],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_title": "Senior Software Engineer",
                    "seniority_level": "Senior",
                    "location": "Berlin, Germany",
                    "salary_range": "€85K-120K",
                    "company_name": "TechCorp Solutions",
                    "skills": "Python, Django, FastAPI, PostgreSQL, Redis, Docker, Kubernetes",
                    "short_description": "Senior Backend Engineer role focusing on microservices architecture and cloud infrastructure",
                }
            ]
        }
    }


class CVSummary(BaseModel):
    """Structured representation of CV summary."""

    summary: str = Field(
        description="Concise summary of the CV while preserving technical details",
        min_length=MIN_CV_LENGTH,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Senior Software Engineer with 7+ years of experience in Python development. "
                    "Expertise in Django, FastAPI, and microservices architecture. "
                    "Strong background in AWS, Docker, and Kubernetes. "
                    "Team leadership experience managing 5-person teams."
                }
            ]
        }
    }


class JobPostRewrite(BaseModel):
    """Structured representation of rewritten job posting."""

    summary: str = Field(
        description="Clear and concise job posting while preserving essential technical information",
        min_length=MIN_CV_LENGTH,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Senior Backend Engineer\n\n"
                    "Key Requirements:\n"
                    "- 5+ years Python development experience\n"
                    "- Strong expertise in Django and FastAPI\n"
                    "- Advanced PostgreSQL and Redis knowledge\n"
                    "- Experience with Docker and Kubernetes\n"
                    "- Microservices architecture design\n"
                    "- AWS cloud services (ECS, Lambda)\n\n"
                    "Location: Berlin, Germany (Hybrid)\n"
                    "Salary: €85K-120K\n"
                    "Company: TechCorp Solutions"
                }
            ]
        }
    }


class CategoryScore(BaseModel):
    """Response model for category score."""

    hard_skills: int = Field(
        ge=0,
        le=40,
        description="Hard skills score (0-40)",
    )
    experience: int = Field(
        ge=0,
        le=40,
        description="Experience score (0-40)",
    )
    soft_skills: int = Field(
        ge=0,
        le=20,
        description="Soft skills and seniority score (0-20)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "hard_skills": 35,
                    "experience": 30,
                    "soft_skills": 15,
                }
            ]
        }
    }


class JobPost(BaseModel):
    """Model for job post detection response.

    This model represents the binary classification result for whether a text contains a job posting.
    A job posting must include a specific job title and at least two of the following:
    - Job responsibilities/requirements
    - Application instructions
    - Employment terms (salary, location, work type)
    - Company hiring information
    - Recruiter or hiring manager contacts
    """

    is_job_post: Literal[True, False] = Field(
        ...,
        description="Whether the text contains a job posting",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"is_job_post": True}, {"is_job_post": False}]
        }
    }


class SingleJobPost(BaseModel):
    """Response model for single job post detection."""

    is_single_post: bool = Field(
        description="Whether the text contains exactly one job posting",
        examples=[True, False],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"is_single_post": True}, {"is_single_post": False}]
        }
    }
