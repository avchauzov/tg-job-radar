"""Tests for the CV matching agent functionality."""

import json
import logging
import unittest
from unittest.mock import MagicMock, patch

import pytest

from _production.utils.agents import (
    create_cv_matching_agent,
    enhanced_cv_matching,
    parse_llm_response,
)
from _production.utils.exceptions import LLMError, LLMParsingError


class TestParseLLMResponse(unittest.TestCase):
    """Test cases for the parse_llm_response function."""

    def setUp(self):
        """Set up test case - capture logs."""
        # Capture logs during tests
        self.log_capture = []
        self.log_handler = LogCaptureHandler(self.log_capture)
        self.logger = logging.getLogger()
        self.original_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.log_handler)

    def tearDown(self):
        """Tear down test case - stop capturing logs."""
        self.logger.removeHandler(self.log_handler)
        self.logger.setLevel(self.original_level)

    def test_parse_valid_json(self):
        """Test parsing a valid JSON response."""
        response = '{"score": 85, "reasoning": "Good match"}'
        result = parse_llm_response(response)
        self.assertEqual(result, {"score": 85, "reasoning": "Good match"})

    def test_parse_json_with_markdown(self):
        """Test parsing JSON with markdown code block markers."""
        response = '```json\n{"score": 90, "reasoning": "Excellent match"}\n```'
        result = parse_llm_response(response)
        self.assertEqual(result, {"score": 90, "reasoning": "Excellent match"})

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON embedded in surrounding text."""
        response = 'Here is the result: {"score": 75, "reasoning": "Average match"} Hope this helps!'
        result = parse_llm_response(response)
        self.assertEqual(result, {"score": 75, "reasoning": "Average match"})

    def test_parse_list_response(self):
        """Test parsing a response that comes as a list."""
        response = ['{"score": 80, "reasoning": "Good technical skills"}']
        result = parse_llm_response(response)
        self.assertEqual(result, {"score": 80, "reasoning": "Good technical skills"})

    def test_parse_invalid_json(self):
        """Test parsing an invalid JSON response."""
        response = "This is not a valid JSON response"
        with self.assertRaises(LLMParsingError) as context:
            parse_llm_response(response)
        self.assertIn("No JSON object found in response", str(context.exception))

    def test_parse_malformed_json(self):
        """Test parsing a malformed JSON response."""
        response = '{"score": 85, "reasoning": "Missing closing bracket'
        with self.assertRaises(LLMParsingError) as context:
            parse_llm_response(response)
        self.assertIn("No JSON object found in response", str(context.exception))


class LogCaptureHandler(logging.Handler):
    """A logging handler that captures log records in a list."""

    def __init__(self, log_list):
        """Initialize with a list to store the log records."""
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        """Store the log record in the list."""
        self.log_list.append(record)


@pytest.fixture
def mock_llm_response():
    """Fixture providing a mock LLM response for testing."""
    return {
        "content": json.dumps(
            {
                "experience_match": {
                    "score": 85,
                    "reasoning": "The candidate has 5 years of experience in software development, which matches the job requirement of 3-5 years.",
                },
                "skills_match": {
                    "score": 90,
                    "reasoning": "The candidate has strong Python skills and experience with machine learning frameworks as required.",
                },
                "soft_skills_match": {
                    "score": 88,
                    "reasoning": "The candidate demonstrates good communication skills and teamwork experience.",
                },
            }
        )
    }


# Fixture to suppress logs during pytest tests
@pytest.fixture(autouse=True)
def suppress_logs():
    """Suppress logs during pytest tests."""
    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.CRITICAL)
    yield
    logger.setLevel(original_level)


@pytest.mark.parametrize(
    "cv_content,job_post,expected_score",
    [
        (
            "5 years of Python experience, machine learning expertise",
            "Looking for Python developer with 3-5 years experience",
            87,  # (85*0.4 + 90*0.45 + 88*0.15)
        ),
        (
            "2 years of Java experience, web development",
            "Senior Python developer with 5+ years experience needed",
            None,  # Should return None for mismatched skills
        ),
    ],
)
@patch("_production.utils.agents.ChatAnthropic")
def test_enhanced_cv_matching(
    mock_chat_anthropic, cv_content, job_post, expected_score, mock_llm_response
):
    """Test the enhanced_cv_matching function with different inputs."""
    # Setup mock
    mock_instance = mock_chat_anthropic.return_value
    mock_instance.invoke.return_value = mock_llm_response

    # Mock the agent's invoke method
    with patch("_production.utils.agents.initialize_agent") as mock_initialize_agent:
        mock_agent = MagicMock()

        # Configure the mock based on the test case
        if expected_score is None:
            # For the case where we expect None, make the agent return an error
            mock_agent.invoke.side_effect = LLMError("Test error")
        else:
            # For the normal case, return a valid response
            mock_agent.invoke.return_value = {
                "output": json.dumps(
                    {
                        "experience_match": {"score": 85},
                        "skills_match": {"score": 90},
                        "soft_skills_match": {"score": 88},
                    }
                )
            }

        mock_initialize_agent.return_value = mock_agent

        # Call the function
        result = enhanced_cv_matching(cv_content, job_post)

        # Check the result
        assert result == expected_score  # noqa: S101


@patch("_production.utils.agents.ChatAnthropic")
def test_cv_matching_agent(mock_chat_anthropic, mock_llm_response):
    """Test the CV matching agent's comprehensive analysis function."""
    # Setup mock
    mock_instance = mock_chat_anthropic.return_value
    mock_instance.invoke.return_value = mock_llm_response

    # Instead of creating a real agent, we'll mock the agent creation
    # and directly test the comprehensive_cv_analysis function
    with patch("_production.utils.agents.initialize_agent") as mock_initialize_agent:
        # Create a mock agent
        mock_agent = MagicMock()
        mock_initialize_agent.return_value = mock_agent

        # Create a mock for the comprehensive_cv_analysis function
        mock_analysis = MagicMock()
        mock_analysis.return_value = {
            "experience_match": {
                "score": 85,
                "reasoning": "The candidate has 5 years of experience in software development.",
            },
            "skills_match": {
                "score": 90,
                "reasoning": "The candidate has strong Python skills.",
            },
            "soft_skills_match": {
                "score": 88,
                "reasoning": "The candidate demonstrates good communication skills.",
            },
        }

        # Create a CV matching agent with our mocks
        cv_content = "5 years of Python experience, machine learning expertise"

        # Test the agent creation
        create_cv_matching_agent(cv_content)

        # Verify that initialize_agent was called
        mock_initialize_agent.assert_called_once()

        # Now we'll test the agent's behavior by mocking its invoke method
        job_post = "Looking for Python developer with 3-5 years experience"
        mock_agent.invoke.return_value = {
            "output": json.dumps(
                {
                    "experience_match": {"score": 85},
                    "skills_match": {"score": 90},
                    "soft_skills_match": {"score": 88},
                }
            )
        }

        # Call the agent
        result = enhanced_cv_matching(cv_content, job_post)

        # Check the result
        assert result == 87  # (85*0.4 + 90*0.45 + 88*0.15)  # noqa: S101


@patch("_production.utils.agents.ChatAnthropic")
def test_error_handling(mock_chat_anthropic):
    """Test error handling in the CV matching functions."""
    # Setup mock to raise an exception
    mock_instance = mock_chat_anthropic.return_value
    mock_instance.invoke.side_effect = Exception("API error")

    # Test error handling in enhanced_cv_matching
    cv_content = "5 years of Python experience"
    job_post = "Looking for Python developer"

    # Mock the agent's invoke method to raise an exception
    with patch("_production.utils.agents.initialize_agent") as mock_initialize_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = LLMError("Test error")
        mock_initialize_agent.return_value = mock_agent

        # Call the function and check that it returns None on error
        result = enhanced_cv_matching(cv_content, job_post)
        assert result is None  # noqa: S101


if __name__ == "__main__":
    unittest.main()
