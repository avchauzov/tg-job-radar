"""Agent utilities for LLM-powered CV matching and response processing.

This module provides functions for creating and using LangChain agents to analyze
CVs against job postings, with robust error handling and response parsing capabilities.
It includes utilities for extracting structured data from LLM responses and calculating
match scores based on experience, skills, and soft skills criteria.
"""

import json
import logging

from langchain.agents import AgentType, initialize_agent
from langchain.tools import StructuredTool
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from _production.utils.custom_model import get_custom_model_client
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
    if start_idx >= 0 and end_idx > start_idx:
        try:
            json_content = json_str[start_idx:end_idx]
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass  # Continue to next strategy

    # Strategy 3: Handle the case where the response contains 'Score: XX'
    score_pattern = r"(?i)score\s*[:=]\s*(\d+)"
    import re

    score_matches = re.findall(score_pattern, json_str)
    if score_matches:
        return {"score": int(score_matches[0])}

    # If all parsing attempts failed, raise an exception
    error_msg = "Could not extract JSON from LLM response"
    logging.error(f"{error_msg}: {json_str[:200]}...")
    raise LLMParsingError(error_msg, response=json_str[:1000])


# Custom LangChain-compatible LLM implementation
class CustomModelLLM(BaseLanguageModel):
    """A LangChain-compatible wrapper for our custom model client."""

    def __init__(self, temperature=0.0, timeout=120, stop=None):
        """Initialize with custom model parameters.

        Args:
            temperature: Temperature parameter for generation
            timeout: Request timeout in seconds
            stop: Optional stop sequences (not used by our model)
        """
        super().__init__()
        self.client = get_custom_model_client()
        self.temperature = temperature
        self.timeout = timeout
        self.stop = stop

    def invoke(self, prompt, **kwargs):
        """Generate a response to the given prompt.

        Args:
            prompt: The prompt text or LangChain message
            **kwargs: Additional parameters

        Returns:
            AIMessage containing the generated text
        """
        # Handle both string prompts and LangChain messages
        if isinstance(prompt, str):
            input_text = prompt
        elif hasattr(prompt, "content"):
            input_text = prompt.content
        else:
            # Handle lists of messages
            system_msg = ""
            user_msgs = []

            for msg in prompt:
                if isinstance(msg, SystemMessage):
                    system_msg = msg.content
                elif isinstance(msg, HumanMessage):
                    user_msgs.append(msg.content)

            # If we have both system and user messages, format them
            if system_msg and user_msgs:
                combined_user_msgs = "\n".join(user_msgs)
                input_text = f"{system_msg}\n\n{combined_user_msgs}"
            else:
                # Otherwise just join all content
                input_text = "\n".join(
                    msg.content for msg in prompt if hasattr(msg, "content")
                )

        # Get temperature from kwargs if provided
        temperature = kwargs.get("temperature", self.temperature)

        # Call our custom model
        try:
            response_text = self.client.generate(
                prompt=input_text,
                temperature=temperature,
                max_tokens=kwargs.get("max_tokens", 1024),
            )
            return AIMessage(content=response_text)
        except Exception as e:
            logging.error(f"Error in CustomModelLLM: {e!s}")
            raise LLMError(f"Failed to generate response: {e!s}")

    # Implement required abstract methods
    def generate_prompt(self, prompts, **kwargs):
        """Generate from a list of prompts."""
        from langchain_core.outputs import LLMResult

        generations = []
        for prompt in prompts:
            response = self.invoke(prompt, **kwargs)
            generations.append([{"text": response.content}])

        return LLMResult(generations=generations)

    async def agenerate_prompt(self, prompts, **kwargs):
        """Async version of generate_prompt."""
        # Just call the sync version for now
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate_prompt, prompts, **kwargs
        )

    def get_num_tokens(self, text):
        """Get the number of tokens in a string."""
        # Simple approximation: 1 token ≈ 4 chars
        return len(text) // 4

    @property
    def _llm_type(self):
        """Return the type of LLM."""
        return "custom_model"

    def predict(self, text, **kwargs):
        """Predict method for string inputs."""
        response = self.invoke(text, **kwargs)
        return response.content

    async def apredict(self, text, **kwargs):
        """Async predict method for string inputs."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, self.predict, text, **kwargs
        )

    def predict_messages(self, messages, **kwargs):
        """Predict method for message inputs."""
        response = self.invoke(messages, **kwargs)
        return response

    async def apredict_messages(self, messages, **kwargs):
        """Async predict method for message inputs."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, self.predict_messages, messages, **kwargs
        )


def create_cv_matching_agent(cv_content: str):
    """
    Create a LangChain agent for CV matching with improved response handling.

    Args:
        cv_content: The content of the candidate's CV

    Returns:
        Agent: A configured LangChain agent for CV matching
    """
    llm = CustomModelLLM(temperature=0.0, timeout=120, stop=["```"])

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

    def score_job_match(job_post: str) -> dict:
        """
        Calculate a match score between a CV and job posting with detailed analysis.

        Args:
            job_post: The job posting content

        Returns:
            dict: Match score and detailed analysis
        """
        try:
            job_post_validated = validate_input_text(job_post, "Job posting")
            cv_validated = cv_content  # Already validated in parent function

            # Prompt template encouraging detailed analysis
            prompt = f"""You are a technical recruiter evaluating if a candidate is suitable for a job position.

            Analyze the match between the CV and job posting in these areas:
            1. Technical Skills Match (programming languages, frameworks, tools)
            2. Experience Level Match (years of experience, seniority)
            3. Domain Knowledge Match (industry experience, specialized knowledge)
            4. Education & Certification Match
            5. Soft Skills & Cultural Fit

            For each area, provide a score out of 100 and brief justification.
            Then calculate a weighted final score using:
            - Technical Skills: 35%
            - Experience Level: 30%
            - Domain Knowledge: 20%
            - Education & Certification: 10%
            - Soft Skills: 5%

            CV:
            {cv_validated}

            Job Post:
            {job_post_validated}

            Provide your analysis as JSON with these fields:
            {{
                "technical_skills": {{
                    "score": <0-100>,
                    "justification": "<brief explanation>"
                }},
                "experience": {{
                    "score": <0-100>,
                    "justification": "<brief explanation>"
                }},
                "domain_knowledge": {{
                    "score": <0-100>,
                    "justification": "<brief explanation>"
                }},
                "education": {{
                    "score": <0-100>,
                    "justification": "<brief explanation>"
                }},
                "soft_skills": {{
                    "score": <0-100>,
                    "justification": "<brief explanation>"
                }},
                "final_score": <0-100>,
                "overall_assessment": "<brief overall assessment>"
            }}
            """

            response = llm.invoke(prompt)
            try:
                # Extract the response to avoid text wrapper types
                response_text = (
                    response.content
                    if hasattr(response, "content")
                    and isinstance(response.content, str)
                    else str(response)
                )

                result = parse_llm_response(response_text)

                # Calculate final score if not already included
                if "final_score" not in result:
                    weights = {
                        "technical_skills": 0.35,
                        "experience": 0.30,
                        "domain_knowledge": 0.20,
                        "education": 0.10,
                        "soft_skills": 0.05,
                    }
                    final_score = sum(
                        result.get(key, {}).get("score", 0) * weight
                        for key, weight in weights.items()
                    )
                    result["final_score"] = int(final_score)

                # Format the result for better readability
                logging.info(f"CV Match Score: {result.get('final_score', 0)}")
                return result

            except LLMParsingError as error:
                # Handle parsing errors with a fallback scoring
                logging.error(f"Error parsing response: {error}")
                error_response = {"final_score": 0, "error": str(error)}
                if hasattr(error, "response") and error.response:
                    error_response["raw_response"] = error.response[:500]
                return error_response

        except Exception as error:
            logging.error(f"Error in score_job_match: {error}")
            return {"final_score": 0, "error": str(error)}

    # Create a structured tool that performs the matching
    match_tool = StructuredTool.from_function(
        func=score_job_match,
        name="score_job_match",
        description="Calculate a match score between a CV and job posting with detailed analysis",
        args_schema=JobPostInput,
    )

    # Initialize the agent with the tool
    agent = initialize_agent(
        [match_tool],
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )

    return agent


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
                    Use score_job_match tool with ONLY the job post as input.
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

        # Extract scores from the score_job_match observation
        try:
            # If we have a direct observation with scores
            if isinstance(output_data, dict) and "experience_score" in output_data:
                experience_score = float(output_data["experience_score"])
                skills_score = float(output_data["skills_score"])
                soft_skills_score = float(output_data["soft_skills_score"])

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
                    experience_score = float(observation_data["experience_score"])
                    skills_score = float(observation_data["skills_score"])
                    soft_skills_score = float(observation_data["soft_skills_score"])

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
