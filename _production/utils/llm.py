"""
LLM integration utilities for structured data extraction and analysis.

This module provides functions for interacting with LLM models:
- Job post detection and validation
- CV-to-job matching with detailed scoring
- Structured information extraction with retry logic
- Error handling and input validation

Supports OpenAI API based on configuration.
"""

import json
import logging
import time
from typing import Any, TypeVar

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from _production import (
    CV_COMPRESSION_RATIO,
    JOB_POST_COMPRESSION_RATIO,
    MAX_CONTEXT_TOKENS,
    MIN_CV_LENGTH,
    OPENAI,
)
from _production.utils.exceptions import (
    LLMError,
    LLMInputError,
    LLMRateLimitError,
    LLMResponseError,
)
from _production.utils.llm_helpers import (
    CategoryScore,
    CleanJobPost,
    CVSummary,
    JobPost,
    JobPostRewrite,
    SingleJobPost,
    count_tokens,
    validate_text_input,
)
from _production.utils.prompts import (
    CLEAN_JOB_POST_PROMPT,
    CV_JOB_MATCHING_PROMPT,
    CV_SUMMARY_PROMPT,
    JOB_POST_DETECTION_PROMPT,
    JOB_POST_REWRITE_PROMPT,
    SINGLE_JOB_POST_DETECTION_PROMPT,
)

T = TypeVar("T", bound=BaseModel)

MAX_TEXT_LENGTH = 64000

# Initialize OpenAI client
logging.info("Using OpenAI API for LLM operations")
OPENAI_CLIENT = openai.OpenAI(api_key=OPENAI["API_KEY"])


def _make_llm_call(
    messages: list[dict[str, str]],
    response_format: type[T],
    max_retries: int = 5,
    sleep_time: int = 10,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> T | None:
    """Make LLM API calls with retry logic.

    Args:
        messages: List of message dictionaries
        response_format: Pydantic model for response validation
        max_retries: Maximum number of retry attempts
        sleep_time: Initial sleep time between retries
        max_tokens: Maximum tokens to generate
        temperature: Temperature parameter

    Returns:
        Validated instance of response_format or None if all retries fail
    """
    start_time = time.time()
    call_type = response_format.__name__

    # Calculate size-based timeout
    total_tokens = sum(count_tokens(msg.get("content", "")) for msg in messages)
    size_based_timeout = min(
        300, max(60, total_tokens // 100)
    )  # 1-5 minutes based on size

    user_message = next(
        (msg["content"] for msg in messages if msg["role"] == "user"),
        "",
    )

    # Get model schema for validation
    model_schema = response_format.model_json_schema()

    for attempt in range(max_retries):
        try:
            logging.info(
                f"Starting LLM call for {call_type} (attempt {attempt + 1}/{max_retries})"
            )
            attempt_start = time.time()

            # Add exponential backoff between retries
            if attempt > 0:
                backoff_time = sleep_time * (2 ** (attempt - 1))
                logging.info(f"Waiting {backoff_time}s before retry...")
                time.sleep(backoff_time)

            # Prepare messages with schema
            system_msg: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": f"You are a helpful assistant that generates structured data according to this schema: {model_schema}",
            }
            user_msg: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": user_message,
            }
            all_messages: list[ChatCompletionMessageParam] = [system_msg, user_msg]

            response: ChatCompletion = OPENAI_CLIENT.chat.completions.create(
                model=OPENAI["MODEL"],
                messages=all_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=size_based_timeout,
            )

            content = response.choices[0].message.content
            if content is None:
                raise LLMError("OpenAI API returned empty response")

            # Parse the response into the requested model
            result = response_format.model_validate_json(content)

            attempt_duration = time.time() - attempt_start
            logging.info(
                f"LLM call for {call_type} completed in {attempt_duration:.2f}s (attempt {attempt + 1})"
            )

            if attempt > 0:  # Only log if we had to retry
                logging.info(f"Successful LLM call after {attempt + 1} attempts")

            total_duration = time.time() - start_time
            logging.info(
                f"Total processing time for {call_type}: {total_duration:.2f}s"
            )

            return result

        except openai.RateLimitError:
            if attempt < max_retries - 1:
                backoff_time = sleep_time * (2**attempt)
                logging.warning(
                    f"Rate limit hit, waiting {backoff_time}s before retry..."
                )
                time.sleep(backoff_time)
                continue
            raise LLMRateLimitError("Rate limit exceeded after max retries")

        except openai.APIError as error:
            if attempt < max_retries - 1:
                backoff_time = sleep_time * (2**attempt)
                logging.warning(
                    f"OpenAI API error: {error!s}, waiting {backoff_time}s before retry..."
                )
                time.sleep(backoff_time)
                continue
            raise LLMError(f"OpenAI API error: {error!s}") from error

        except Exception as error:
            if (
                hasattr(error, "__module__")
                and error.__module__ == "pydantic.error_wrappers"
            ):
                raise LLMResponseError(
                    f"Failed to validate response with model: {error!s}"
                ) from error
            logging.error(f"Unexpected error in LLM call: {error!s}")
            raise LLMError(f"Failed to get response from LLM: {error!s}") from error

    return None


def summarize_cv_content(cv_content: str) -> str | None:
    """
    Create a concise summary of the CV while preserving technical details.

    Args:
        cv_content: Raw CV content from Google Docs

    Returns:
        Optional[str]: Summarized CV content if successful, None if summarization fails
    """
    try:
        # Validate input length
        if len(cv_content) < MIN_CV_LENGTH:
            logging.warning(
                f"CV content too short: {len(cv_content)} chars (minimum: {MIN_CV_LENGTH})"
            )
            return None

        # Calculate target tokens based on compression ratio
        input_tokens = count_tokens(cv_content)
        prompt_tokens = 256  # Estimated tokens for system and user prompts

        # Calculate available tokens for response
        available_tokens = MAX_CONTEXT_TOKENS - prompt_tokens

        # Calculate target tokens with adaptive minimum
        target_tokens = min(input_tokens // CV_COMPRESSION_RATIO, available_tokens)

        # Ensure minimum token count based on input size
        min_tokens = min(1024, input_tokens // 2)  # Увеличено минимальное число токенов
        max_tokens = max(
            target_tokens, min_tokens, 2048
        )  # Увеличено максимальное число токенов

        logging.info(
            f"Token calculation - Input: {input_tokens}, Target: {target_tokens}, "
            f"Min: {min_tokens}, Max: {max_tokens}, Available: {available_tokens}"
        )

        # Prepare prompt for LLM with enhanced instructions
        prompt = f"""Please create a concise summary of this CV while preserving all technical details and skills.
        Follow these guidelines:
        1. Preserve all technical skills, programming languages, and frameworks
        2. Keep key achievements and responsibilities
        3. Maintain important certifications and education
        4. Remove redundant information and formatting
        5. Focus on quantifiable achievements where possible
        6. Keep the summary professional and objective

        Target length should be approximately {len(cv_content) // CV_COMPRESSION_RATIO} characters.

        CV content:
        {cv_content}
        """

        messages = [
            {
                "role": "system",
                "content": CV_SUMMARY_PROMPT,
            },
            {"role": "user", "content": prompt},
        ]

        # Call LLM using the structured approach with optimized parameters
        result = _make_llm_call(
            messages=messages,
            response_format=CVSummary,
            max_retries=3,
            sleep_time=10,
            max_tokens=int(max_tokens),  # Ensure max_tokens is an integer
            temperature=0.2,  # Reduced temperature for more consistent output
        )

        if not result:
            logging.error("Failed to generate CV summary")
            return None

        return result.summary

    except Exception as error:
        logging.error(f"CV summarization failed: {error!s}", exc_info=True)
        return None


def job_post_detection(
    post: str, max_retries: int = 3, sleep_time: int = 10
) -> bool | None:
    """Determines if the text contains any job postings.

    Args:
        post: Text to analyze for job posting content
        max_retries: Maximum number of retry attempts for LLM call
        sleep_time: Initial sleep time between retries in seconds

    Returns:
        Optional[bool]: True if job posting detected, False if not, None if error

    Raises:
        LLMInputError: If input is empty or too long
    """
    start_time = time.time()

    # Input validation
    if not post or not post.strip():
        raise LLMInputError("Empty or whitespace-only input provided")

    # Token length validation
    token_count = count_tokens(post)
    if token_count > MAX_CONTEXT_TOKENS:
        raise LLMInputError(
            f"Input too long ({token_count} tokens). Maximum allowed: {MAX_CONTEXT_TOKENS} tokens"
        )

    messages = [
        {
            "role": "system",
            "content": JOB_POST_DETECTION_PROMPT,
        },
        {
            "role": "user",
            "content": f"Text to classify:\n\n{post}",
        },
    ]

    result = _make_llm_call(messages, JobPost, max_retries, sleep_time)

    processing_time = time.time() - start_time
    logging.info(f"Job post detection completed in {processing_time:.2f}s")

    return result.is_job_post if result else None


def single_job_post_detection(post, max_retries=3, sleep_time=10):
    """Determines if the text contains exactly one job posting."""
    start_time = time.time()

    messages = [
        {
            "role": "system",
            "content": SINGLE_JOB_POST_DETECTION_PROMPT,
        },
        {
            "role": "user",
            "content": f"Does this text contain exactly one job posting?\n\n{post}",
        },
    ]

    result = _make_llm_call(messages, SingleJobPost, max_retries, sleep_time)

    processing_time = time.time() - start_time
    logging.info(f"Single job post detection completed in {processing_time:.2f}s")

    return result.is_single_post if result else None


def rewrite_job_post(post: str) -> str | None:
    """
    Rewrite a job posting to be clear and concise while preserving essential information.

    Args:
        post: Raw job posting text

    Returns:
        Optional[str]: Rewritten job posting if successful, None if rewriting fails
    """
    try:
        # Calculate target tokens based on compression ratio
        input_tokens = count_tokens(post)
        prompt_tokens = 256  # Estimated tokens for system and user prompts

        # Calculate available tokens for response
        available_tokens = MAX_CONTEXT_TOKENS - prompt_tokens

        # Calculate target tokens with adaptive minimum
        target_tokens = min(
            input_tokens // JOB_POST_COMPRESSION_RATIO, available_tokens
        )

        # Ensure minimum token count based on input size
        min_tokens = min(1024, input_tokens // 2)  # Увеличено минимальное число токенов
        max_tokens = max(
            target_tokens, min_tokens, 2048
        )  # Увеличено максимальное число токенов

        logging.info(
            f"Token calculation - Input: {input_tokens}, Target: {target_tokens}, "
            f"Min: {min_tokens}, Max: {max_tokens}, Available: {available_tokens}"
        )

        # Prepare prompt for LLM (требуем строго JSON)
        prompt = (
            f"Please rewrite this job posting to be clear and concise while preserving all essential information.\n"
            f"Follow these guidelines:\n"
            f"1. Keep all required skills and qualifications\n"
            f"2. Preserve salary range and benefits if mentioned\n"
            f"3. Maintain company name and location\n"
            f"4. Keep job title and seniority level\n"
            f"5. Remove redundant information and formatting\n"
            f"6. Focus on key responsibilities and requirements\n"
            f"\n"
            f'Ответь строго в формате JSON: {{"summary": "..."}}\n'
            f"Job posting:\n{post}\n"
        )

        messages = [
            {"role": "system", "content": JOB_POST_REWRITE_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Call LLM using the structured approach
        result = _make_llm_call(
            messages=messages,
            response_format=JobPostRewrite,  # Use dedicated JobPostRewrite model
            max_retries=3,
            sleep_time=10,
            max_tokens=int(max_tokens),  # Ensure max_tokens is an integer
            temperature=0.2,  # Reduced temperature for more consistent output
        )

        # Post-processing: проверяем, что summary есть и это строка
        if not result or not getattr(result, "summary", None):
            logging.error(
                "Failed to rewrite job post: LLM did not return valid JSON with 'summary' field. Возвращаю оригинальный пост."
            )
            return post

        return result.summary

    except Exception as error:
        logging.error(f"Job post rewriting failed: {error!s}", exc_info=True)
        return post


def match_cv_with_job(
    cv_text: str, post: str, max_retries: int = 5, sleep_time: int = 10
) -> float | None:
    """Evaluates match between CV and job post using a detailed scoring system.

    Args:
        cv_text: The candidate's CV text
        post: The job posting text
        max_retries: Maximum number of retry attempts
        sleep_time: Initial sleep time between retries

    Returns:
        float | None: The final matching score (0-100) or None if evaluation fails
    """
    start_time = time.time()

    try:
        validate_text_input(cv_text, "CV text")
        validate_text_input(post, "Job post")

        max_input_length = 32000
        cv_text = cv_text[:max_input_length]
        post = post[:max_input_length]

        messages = [
            {
                "role": "system",
                "content": CV_JOB_MATCHING_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
                CV:
                {cv_text}

                Job Post:
                {post}

                Provide your response in this exact JSON format:
                {{
                    "hard_skills": <score 0-40>,
                    "experience": <score 0-40>,
                    "soft_skills": <score 0-20>
                }}
                """,
            },
        ]

        result = _make_llm_call(
            messages,
            CategoryScore,
            max_retries,
            sleep_time,
            max_tokens=128,
            temperature=0.0,
        )

        if result:
            # Calculate total score
            total_score = result.hard_skills + result.experience + result.soft_skills
            logging.info(
                f"Scores - Hard skills: {result.hard_skills}, Experience: {result.experience}, "
                f"Soft skills: {result.soft_skills}, Total: {total_score}"
            )

            processing_time = time.time() - start_time
            logging.info(
                f"CV matching completed in {processing_time:.2f}s with score: {total_score}"
            )
            return total_score

        return 0  # Если не удалось получить результат, возвращаем 0

    except Exception as error:
        processing_time = time.time() - start_time
        logging.error(
            f"Error during CV matching in {processing_time:.2f}s: {error!s}",
            exc_info=True,
        )
        return 0  # Если ошибка, возвращаем 0


def job_post_parsing(
    post: str, compressed_post: str, max_retries: int = 3, sleep_time: int = 10
) -> dict[str, Any]:
    """Parse job posting into structured format. Always returns a dict with at least {'description': compressed_post} on error."""
    start_time = time.time()

    try:
        validate_text_input(post, "Job post")

        # Calculate tokens for structured data extraction
        input_tokens = count_tokens(post)
        prompt_tokens = 512  # Increased from 256 to 512 for more detailed prompts

        # Calculate available tokens for response
        available_tokens = MAX_CONTEXT_TOKENS - prompt_tokens

        # For structured data extraction, we need fewer tokens than input
        # since we're extracting specific fields
        target_tokens = min(
            input_tokens // 2, available_tokens
        )  # Use half of input tokens

        # Ensure minimum token count for structured data
        min_tokens = 256  # Increased from 128 to 256 for more detailed responses
        max_tokens = max(target_tokens, min_tokens)

        logging.info(
            f"Token calculation - Input: {input_tokens}, Target: {target_tokens}, "
            f"Min: {min_tokens}, Max: {max_tokens}, Available: {available_tokens}"
        )

        messages = [
            {
                "role": "system",
                "content": CLEAN_JOB_POST_PROMPT,
            },
            {
                "role": "user",
                "content": f"Parse this job posting into a structured format:\n\n{post}",
            },
        ]

        try:
            result = _make_llm_call(
                messages=messages,
                response_format=CleanJobPost,
                max_retries=max_retries,
                sleep_time=sleep_time,
                max_tokens=int(max_tokens),  # Ensure max_tokens is an integer
                temperature=0.0,  # Deterministic output
            )
            if not result:
                processing_time = time.time() - start_time
                logging.warning(
                    f"Job post parsing failed, using fallback. Returning only description. Time: {processing_time:.2f}s"
                )
                return {"description": compressed_post}

            # Get initial parsed response
            response = result.model_dump()

            # Log the successful response for debugging
            logging.debug(f"Successfully parsed structured data: {response}")

            # Ensure all required fields are present
            required_fields = [
                "job_title",
                "seniority_level",
                "location",
                "salary_range",
                "company_name",
                "skills",
                "short_description",
            ]
            for field in required_fields:
                if field not in response:
                    response[field] = None

            # Clean and validate the response
            response_filtered = {
                key: value.strip() if isinstance(value, str) else value
                for key, value in response.items()
                if value is not None
            }

            if not response_filtered:
                processing_time = time.time() - start_time
                logging.warning(
                    f"Job post parsing produced invalid result, using fallback. Returning only description. Time: {processing_time:.2f}s"
                )
                return {"description": compressed_post}

            processing_time = time.time() - start_time
            logging.info(
                f"Job post parsing completed in {processing_time:.2f}s for job: {response_filtered}"
            )

            # Always include description as compressed_post
            response_filtered["description"] = compressed_post
            return response_filtered

        except LLMResponseError as error:
            logging.warning(
                f"Failed to parse structured response: {error}. Using fallback description."
            )
            return {"description": compressed_post}

    except LLMInputError as error:
        processing_time = time.time() - start_time
        logging.error(
            f"Input validation failed during job parsing in {processing_time:.2f}s: {error!s}. Using fallback description."
        )
        return {"description": compressed_post}
    except Exception as error:
        processing_time = time.time() - start_time
        logging.error(
            f"Job post parsing failed in {processing_time:.2f}s: {error!s}. Using fallback description."
        )
        return {"description": compressed_post}


def clean_job_post_values(
    response: dict[str, Any], exclude_fields: list[str] | None = None
) -> dict[str, Any]:
    """Clean and standardize job post values, excluding specified fields from standardization.

    Args:
        response: Raw job post data dictionary
        exclude_fields: List of field names to exclude from standardization (e.g., ['description'])

    Returns:
        Dictionary with cleaned values, with excluded fields merged back in
    """
    start_time = time.time()
    exclude_fields = exclude_fields or []
    excluded = {k: v for k, v in response.items() if k in exclude_fields}
    to_clean = {k: v for k, v in response.items() if k not in exclude_fields}

    messages = [
        {
            "role": "system",
            "content": CLEAN_JOB_POST_PROMPT,
        },
        {
            "role": "user",
            "content": f"Please clean and standardize this job posting data: {json.dumps(to_clean)}",
        },
    ]

    try:
        result = _make_llm_call(
            messages=messages,
            response_format=CleanJobPost,
            max_retries=3,
            sleep_time=1,
            max_tokens=512,  # Increased from default to 512 for more detailed cleaning
            temperature=0.0,  # Deterministic output
        )
        if not result:
            processing_time = time.time() - start_time
            logging.error(
                f"Failed to get valid response from LLM in {processing_time:.2f}s"
            )
            raise LLMResponseError("Failed to get valid response from LLM")

        processing_time = time.time() - start_time
        logging.info(f"Job post cleaning completed in {processing_time:.2f}s")
        cleaned = result.model_dump()
        cleaned.update(excluded)
        return cleaned

    except Exception as error:
        processing_time = time.time() - start_time
        logging.error(
            f"Failed to clean job post values in {processing_time:.2f}s: {error!s}"
        )
        fallback = CleanJobPost(
            job_title=None,
            seniority_level=None,
            location=None,
            salary_range=None,
            company_name=None,
            skills=None,
            short_description=None,
        ).model_dump()
        fallback.update(excluded)
        return fallback


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Test data
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

    Nice to Have:
    - Go programming experience
    - MongoDB experience
    - Prometheus/Grafana monitoring
    - TypeScript/React for full-stack contributions
    - AWS certifications

    Benefits:
    - Competitive salary: €85K-120K
    - Flexible hybrid work arrangement
    - Professional development budget
    """

    print("\n=== Testing token counting ===")
    cv_tokens = count_tokens(sample_cv)
    job_tokens = count_tokens(sample_job_post)
    print(f"CV tokens: {cv_tokens}")
    print(f"Job post tokens: {job_tokens}")

    print("\n=== Testing job post detection ===")
    result = job_post_detection(sample_job_post)
    print(f"Is job post: {result}")

    print("\n=== Testing single job post detection ===")
    result = single_job_post_detection(sample_job_post)
    print(f"Is single job post: {result}")

    print("\n=== Testing match_cv_with_job ===")
    match_score = match_cv_with_job(sample_cv, sample_job_post)
    print(f"CV match score: {match_score}")

    print("\n=== Testing job_post_parsing ===")
    parsed_job = job_post_parsing(sample_job_post, sample_job_post)
    print("Parsed job post:")
    if parsed_job:
        for key, value in parsed_job.items():
            print(f"{key}: {value}")
    else:
        print("Failed to parse job post")

    print("\n=== Testing clean_job_post_values ===")
    sample_response = {
        "job_title": "Senior Backend Engineer",
        "seniority_level": "Senior",
        "location": "Berlin, Germany (Hybrid)",
        "salary_range": "€85K-120K",
        "company_name": "TechCorp Solutions",
        "description": "We're seeking an experienced Backend Engineer to join our platform team.",
        "skills": "Python, Django, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS, Microservices, Go, MongoDB, Prometheus, Grafana, TypeScript, React",
    }
    cleaned = clean_job_post_values(sample_response)
    print("Cleaned job post:")
    for key, value in cleaned.items():
        print(f"{key}: {value}")

    print("\n=== Testing summarize_cv_content ===")
    try:
        summary = summarize_cv_content(sample_cv)
        print("CV Summary:")
        print(summary if summary else "Failed to generate summary")
    except Exception as error:
        print(f"Error in summarize_cv_content: {error!s}")

    print("\n=== Testing rewrite_job_post ===")
    try:
        rewritten = rewrite_job_post(sample_job_post)
        print("Rewritten job post:")
        print(rewritten if rewritten else "Failed to rewrite job post")
    except Exception as error:
        print(f"Error in rewrite_job_post: {error!s}")

    print("\nAll tests completed!")
