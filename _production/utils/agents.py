"""Agent utilities for LLM-powered CV matching and response processing.

This module provides functions for creating and using LangChain agents to analyze
CVs against job postings, with robust error handling and response parsing capabilities.
It includes utilities for extracting structured data from LLM responses and calculating
match scores based on experience, skills, and soft skills criteria.
"""

import json
import logging
from typing import Any

from langchain.agents import AgentType, initialize_agent
from langchain.tools import StructuredTool
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from _production import LLM_BASE_MODEL
from _production.utils.exceptions import LLMError, LLMParsingError


def validate_input_text(text: str, input_name: str, min_length: int = 50) -> str:
    """
    Validate input text for LLM processing.

    Args:
        text: The text to validate
        input_name: Name of the input for error messages
        min_length: Minimum acceptable length for the text

    Returns:
        str: The validated text

    Raises:
        ValueError: If the text is invalid or too short
    """
    if not text:
        raise ValueError(f"{input_name} cannot be empty")

    text_str = str(text).strip()
    if len(text_str) < min_length:
        raise ValueError(
            f"{input_name} is too short (minimum {min_length} characters required)"
        )

    return text_str


def parse_llm_response(response: str | list) -> dict:
    """
    Safely parse LLM response into a dictionary with structured error handling.

    Args:
        response: The raw response from the LLM, either as a string or list

    Returns:
        dict: Parsed JSON response or error information dictionary

    Raises:
        LLMParsingError: When the response cannot be parsed into the expected format

    Examples:
        >>> parse_llm_response('{"score": 85, "reasoning": "Good match"}')
        {'score': 85, 'reasoning': 'Good match'}

        >>> parse_llm_response('No JSON here')
        Raises LLMParsingError
    """
    # Handle different response types
    if response is None:
        error_msg = "Received None response from LLM"
        logging.error(error_msg)
        raise LLMParsingError(error_msg)

    # Convert response to string if it's a list
    json_str = response[0] if isinstance(response, list) and response else response
    json_str = str(json_str).strip()

    # Check for batch processing errors
    if "No valid records in batch" in json_str:
        error_msg = "Batch processing error detected in LLM response"
        logging.error(f"{error_msg}: {json_str[:200]}...")
        raise LLMParsingError(error_msg, response=json_str[:1000])

    # Log the raw response for debugging
    logging.debug(f"Raw LLM response to parse: {json_str[:500]}...")

    # Try multiple extraction strategies

    # Strategy 1: Direct JSON parsing if it's already valid JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass  # Continue to next strategy

    # Strategy 2: Find JSON between curly braces
    start_idx = json_str.find("{")
    end_idx = json_str.rfind("}") + 1

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        try:
            extracted_json = json_str[start_idx:end_idx]
            # Remove any markdown code block markers
            extracted_json = extracted_json.replace("```json", "").replace("```", "")
            return json.loads(extracted_json)
        except json.JSONDecodeError:
            pass  # Continue to next strategy

    # Strategy 3: Look for JSON in code blocks
    import re

    json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", json_str)

    for block in json_blocks:
        try:
            cleaned_block = block.strip()
            if cleaned_block.startswith("{") and cleaned_block.endswith("}"):
                return json.loads(cleaned_block)
        except json.JSONDecodeError:
            continue

    # Strategy 4: Try to extract a structured response even without proper JSON formatting
    # Look for the expected fields in the response
    expected_fields = ["experience_match", "skills_match", "soft_skills_match"]

    # Check if all expected fields are in the response
    if all(field in json_str for field in expected_fields):
        try:
            # Try to construct a valid JSON manually
            result = {}
            for field in expected_fields:
                # Find the field and extract score
                score_match = re.search(rf"{field}[^\d]*(\d+)", json_str)
                if score_match:
                    score = int(score_match.group(1))
                    # Extract reasoning (text between this field and the next field or end)
                    field_pos = json_str.find(field)
                    next_field_pos = float("inf")
                    for next_field in expected_fields:
                        if next_field != field:
                            pos = json_str.find(next_field, field_pos + len(field))
                            if pos != -1 and pos < next_field_pos:
                                next_field_pos = pos

                    if next_field_pos == float("inf"):
                        next_field_pos = len(json_str)

                    # Extract text between score and next field as reasoning
                    reasoning_text = json_str[
                        field_pos + len(field) : next_field_pos
                    ].strip()
                    reasoning_text = re.sub(
                        r'.*"reasoning":\s*"(.*?)".*',
                        r"\1",
                        reasoning_text,
                        flags=re.DOTALL,
                    )
                    if not reasoning_text or len(reasoning_text) < 10:
                        reasoning_text = f"Extracted score for {field}"

                    result[field] = {"score": score, "reasoning": reasoning_text}

            if result and all(field in result for field in expected_fields):
                logging.warning(
                    f"Constructed JSON from unstructured response: {result}"
                )
                return result
        except Exception as e:
            logging.error(f"Failed to manually extract structured data: {e}")

    # If we got here, all strategies failed
    error_msg = "No JSON object found in response"
    logging.error(f"{error_msg}: {json_str[:200]}...")
    raise LLMParsingError(error_msg, response=json_str[:1000])


def create_cv_matching_agent(cv_content: str):
    """
    Create a LangChain agent for CV matching with improved response handling.

    Args:
        cv_content: The content of the candidate's CV

    Returns:
        Agent: A configured LangChain agent for CV matching
    """
    llm = ChatAnthropic(
        model_name=LLM_BASE_MODEL, temperature=0.0, timeout=120, stop=["```"]
    )

    # Define input schema for the structured tool
    class JobPostInput(BaseModel):
        job_post: str = Field(
            ..., description="The job posting to match against the CV"
        )

    def validate_input_text(text: str, input_name: str, min_length: int = 50) -> str:
        """
        Validate input text for LLM processing.

        Args:
            text: The text to validate
            input_name: Name of the input for error messages
            min_length: Minimum acceptable length for the text

        Returns:
            str: The validated text

        Raises:
            ValueError: If the text is invalid or too short
        """
        if not text:
            raise ValueError(f"{input_name} cannot be empty")

        text_str = str(text).strip()
        if len(text_str) < min_length:
            raise ValueError(
                f"{input_name} is too short (minimum {min_length} characters required)"
            )

        return text_str

    def process_job_post_input(job_post: Any) -> str:
        """
        Process and normalize job post input from various formats.

        Args:
            job_post: The job posting in various possible formats

        Returns:
            str: Normalized job post text
        """
        # Handle different input formats
        if isinstance(job_post, dict) and "job_post" in job_post:
            # Extract from structured input
            job_post_str = str(job_post["job_post"])
        elif isinstance(job_post, dict | list):
            # If it's a structured input, convert to string representation
            import json

            try:
                job_post_str = json.dumps(job_post)
            except (TypeError, ValueError):
                job_post_str = str(job_post)
        elif isinstance(job_post, list | tuple):
            # If it's a list of arguments, use only the first one as job post
            logging.warning(
                f"Received multiple arguments to comprehensive_cv_analysis: {job_post}"
            )
            job_post_str = str(job_post[0]) if job_post else ""
        else:
            # Regular string input
            job_post_str = str(job_post)

        return job_post_str

    def validate_cv_analysis_response(parsed_response: dict) -> None:
        """
        Validate the structure of a CV analysis response.

        Args:
            parsed_response: The parsed response to validate

        Raises:
            LLMParsingError: If the response structure is invalid
        """
        # Validate the expected structure exists
        required_fields = ["experience_match", "skills_match", "soft_skills_match"]
        missing_fields = [
            field for field in required_fields if field not in parsed_response
        ]

        if missing_fields:
            raise LLMParsingError(
                "Missing required fields in LLM response",
                field=", ".join(missing_fields),
            )

        # Validate each field has the expected structure
        for field in required_fields:
            if (
                not isinstance(parsed_response[field], dict)
                or "score" not in parsed_response[field]
            ):
                raise LLMParsingError(
                    "Invalid structure for field",
                    field=field,
                )

    def comprehensive_cv_analysis(job_post: Any) -> dict[str, dict[str, float]]:
        """
        Analyze all aspects of CV match in a single request with improved response formatting.

        Args:
            job_post: The job posting to match against. Can be a string or a structured input.

        Returns:
            dict: A structured dictionary with match scores and reasoning

        Raises:
            LLMParsingError: If the response cannot be parsed into the expected format
            LLMError: If other LLM-related errors occur
            ValueError: If input validation fails
        """
        try:
            # Process and normalize job post input
            job_post_str = process_job_post_input(job_post)

            # Validate the job post
            job_post_str = validate_input_text(job_post_str, "Job post")

            # Log the processed job post
            logging.debug(f"Processing job post: {job_post_str[:200]}...")

            response = llm.invoke(
                f"""Evaluate CV against job requirements. Provide scores (0-100) for:

                1. EXPERIENCE MATCH (40% weight):
                   - Years of Experience (50%)
                   - Domain Knowledge (30%)
                   - Project Scale (20%)

                2. SKILLS MATCH (45% weight):
                   - Technical Skills (40%)
                   - Education (25%)
                   - Tools/Technologies (20%)
                   - Certifications (15%)

                3. SOFT SKILLS MATCH (15% weight):
                   - Communication (30%)
                   - Team Collaboration (30%)
                   - Problem-Solving (20%)
                   - Cultural Fit (20%)

                Score Guidelines:
                95-100: Exceeds all requirements
                85-94: Meets all requirements
                75-84: Meets most requirements
                65-74: Meets basic requirements
                50-64: Meets some requirements
                0-49: Missing critical requirements

                RESPONSE FORMAT: JSON only with this structure:
                {{
                    "experience_match": {{
                        "score": 85,
                        "reasoning": "Detailed reasoning..."
                    }},
                    "skills_match": {{
                        "score": 90,
                        "reasoning": "Detailed reasoning..."
                    }},
                    "soft_skills_match": {{
                        "score": 88,
                        "reasoning": "Detailed reasoning..."
                    }}
                }}

                CV: {cv_content}

                Job Post: {job_post_str}"""
            )

            # Log the raw response for debugging
            logging.debug(f"Raw LLM response: {response.content[:500]}...")

            # Parse the response
            try:
                parsed_response = parse_llm_response(response.content)
            except LLMParsingError as error:
                # Add more context to the error
                raise LLMParsingError(
                    f"Failed to parse CV analysis response: {error.message}",
                    response=str(error.response) if error.response else None,
                ) from error

            # Validate the response structure
            validate_cv_analysis_response(parsed_response)

            return parsed_response

        except LLMParsingError:
            # Re-raise parsing errors without wrapping
            raise
        except ValueError as error:
            # Re-raise validation errors
            logging.error(f"Input validation error: {error}")
            raise
        except Exception as error:
            error_msg = f"Comprehensive CV analysis failed: {error!s}"
            logging.error(error_msg)
            raise LLMError(error_msg) from error

    # Create a structured tool that can handle complex inputs
    cv_analysis_tool = StructuredTool.from_function(
        func=comprehensive_cv_analysis,
        name="comprehensive_cv_analysis",
        description="Analyzes all aspects of CV match against job post in a single request",
        args_schema=JobPostInput,
        return_direct=False,
    )

    return initialize_agent(
        [cv_analysis_tool],
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,  # Add error handling for parsing issues
    )


def validate_cv_analysis_response(parsed_response: dict) -> None:
    """
    Validate the structure of a CV analysis response.

    Args:
        parsed_response: The parsed response to validate

    Raises:
        LLMParsingError: If the response structure is invalid
    """
    # Validate the expected structure exists
    required_fields = ["experience_match", "skills_match", "soft_skills_match"]
    missing_fields = [
        field for field in required_fields if field not in parsed_response
    ]

    if missing_fields:
        raise LLMParsingError(
            "Missing required fields in LLM response",
            field=", ".join(missing_fields),
        )

    # Validate each field has the expected structure
    for field in required_fields:
        if (
            not isinstance(parsed_response[field], dict)
            or "score" not in parsed_response[field]
        ):
            raise LLMParsingError(
                "Invalid structure for field",
                field=field,
            )


def enhanced_cv_matching(cv_content: str, job_post: str) -> float | None:
    """
    Enhanced CV matching using LangChain agent with improved error handling.

    Args:
        cv_content: The content of the candidate's CV
        job_post: The job posting to match against

    Returns:
        float: The calculated match score (0-100) or None if processing failed
    """
    try:
        # Validate inputs
        cv_content = validate_input_text(cv_content, "CV content", min_length=100)
        job_post = validate_input_text(job_post, "Job post", min_length=50)

        # Create agent only once and cache the result
        agent = create_cv_matching_agent(cv_content)

        # Single LLM call with comprehensive error handling
        try:
            result = agent.invoke(
                {
                    "input": f"""Analyze job post compatibility with CV.
                    Use comprehensive_cv_analysis tool with ONLY the job post as input.
                    After getting scores, calculate:
                    final_score = (experience_score * 0.40) + (skills_score * 0.45) + (soft_skills_score * 0.15)
                    Return ONLY a JSON object with the final_score field.
                    Job Post: {job_post}
                    """,
                    "chat_history": [],
                }
            )
        except Exception as llm_error:
            logging.error(f"LLM call failed: {llm_error}")
            return None

        # Early validation of result structure
        if not isinstance(result, dict) or "output" not in result:
            logging.error(f"Invalid agent output format: {result}")
            return None

        output_data = result["output"]

        # Extract scores from the comprehensive_cv_analysis observation
        try:
            # If we have a direct observation with scores
            if isinstance(output_data, dict) and "experience_match" in output_data:
                experience_score = float(output_data["experience_match"]["score"])
                skills_score = float(output_data["skills_match"]["score"])
                soft_skills_score = float(output_data["soft_skills_match"]["score"])

                final_score = round(
                    (experience_score * 0.40)
                    + (skills_score * 0.45)
                    + (soft_skills_score * 0.15)
                )
                return final_score

            # Try to find the observation in the agent's thought process
            if "Observation:" in str(output_data):
                observation_text = (
                    str(output_data).split("Observation:", 1)[1].split("Thought:", 1)[0]
                )
                try:
                    observation_data = parse_llm_response(observation_text)
                    experience_score = float(
                        observation_data["experience_match"]["score"]
                    )
                    skills_score = float(observation_data["skills_match"]["score"])
                    soft_skills_score = float(
                        observation_data["soft_skills_match"]["score"]
                    )

                    final_score = round(
                        (experience_score * 0.40)
                        + (skills_score * 0.45)
                        + (soft_skills_score * 0.15)
                    )
                    return final_score
                except (LLMParsingError, ValueError, KeyError) as error:
                    logging.error(f"Failed to parse observation data: {error}")
                    return None

            # Try to extract final score from text response
            import re

            # First try to extract from JSON structure
            if isinstance(output_data, dict):
                if "action_input" in output_data and isinstance(
                    output_data["action_input"], dict
                ):
                    if "final_score" in output_data["action_input"]:
                        return round(float(output_data["action_input"]["final_score"]))

            # Then try regex pattern for text-based responses
            score_match = re.search(r"final_score.*?=.*?(\d+\.?\d*)", str(output_data))
            if score_match:
                return round(float(score_match.group(1)))

            logging.error(f"Could not extract score from output: {output_data}")
            return None

        except (ValueError, TypeError, KeyError) as error:
            logging.error(f"Failed to calculate score: {error}")
            return None

    except ValueError as error:
        logging.error(f"Input validation error: {error}")
        return None
    except Exception as error:
        logging.error(f"Unexpected error: {error}")
        return None
