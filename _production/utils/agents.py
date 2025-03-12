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

        agent = create_cv_matching_agent(cv_content)

        result = agent.invoke(
            {
                "input": f"""Analyze job post compatibility with CV.

                Use comprehensive_cv_analysis tool with ONLY the job post as input.

                After getting scores, calculate:
                final_score = (experience_score * 0.40) + (skills_score * 0.45) + (soft_skills_score * 0.15)

                Return JSON with:
                {{
                    "experience_match": {{ "score": 85, "reasoning": "..." }},
                    "skills_match": {{ "score": 90, "reasoning": "..." }},
                    "soft_skills_match": {{ "score": 88, "reasoning": "..." }},
                    "final_score": 87,
                    "final_reasoning": "Overall assessment..."
                }}

                Job Post: {job_post}
                """,
                "chat_history": [],
            }
        )

        # Enhanced error handling for result parsing
        if not isinstance(result, dict) or "output" not in result:
            error_msg = "Unexpected agent output format"
            logging.error(f"{error_msg}: {result}")
            raise LLMParsingError(
                error_msg, response=str(result)[:1000] if result else None
            )

        try:
            # Extract the output from the agent result
            output_data = result["output"]

            # Check if the output is already a dictionary (parsed JSON)
            if isinstance(output_data, dict) and "experience_match" in output_data:
                parsed_output = output_data
            else:
                # Try to extract JSON if the output is a string
                if isinstance(output_data, str):
                    parsed_output = parse_llm_response(output_data)
                else:
                    # Convert to string if it's not already a string
                    parsed_output = parse_llm_response(str(output_data))

            # Check if there's a final_score already calculated
            if "final_score" in parsed_output and isinstance(
                parsed_output["final_score"], int | float
            ):
                final_score = float(parsed_output["final_score"])
                logging.info(f"Using pre-calculated final score: {final_score}")
                return final_score

            # Validate the response structure
            validate_cv_analysis_response(parsed_output)

            # Extract and convert scores to float
            try:
                experience_score = float(parsed_output["experience_match"]["score"])
                skills_score = float(parsed_output["skills_match"]["score"])
                soft_skills_score = float(parsed_output["soft_skills_match"]["score"])
            except (ValueError, TypeError) as error:
                error_msg = "Failed to convert scores to float"
                logging.error(f"{error_msg}: {error}")
                raise LLMParsingError(
                    error_msg, response=str(output_data)[:1000] if output_data else None
                ) from error

            # Calculate weighted final score
            final_score = int(
                (skills_score * 0.45)
                + (experience_score * 0.40)
                + (soft_skills_score * 0.15)
            )

            logging.info(f"Successfully calculated match score: {final_score}")
            return final_score

        except LLMParsingError:
            # Re-raise parsing errors without wrapping
            raise
        except (KeyError, ValueError, TypeError) as error:
            error_msg = f"Failed to parse agent output structure: {error!s}"
            logging.error(error_msg)
            raise LLMParsingError(
                error_msg,
                response=str(result["output"])[:1000] if "output" in result else None,
            ) from error

    except ValueError as error:
        # Handle input validation errors
        logging.error(f"Input validation error in CV matching: {error}")
        return None
    except (LLMParsingError, LLMError) as error:
        logging.error(f"CV matching error: {error!s}")
        return None
    except Exception as error:
        logging.error(f"Unexpected error in CV matching: {error!s}")
        return None
