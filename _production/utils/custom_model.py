"""
Custom model integration for structured data extraction and generation.

This module provides client classes for connecting to a local LLM server with:
- Text generation with temperature control
- Structured data extraction using pydantic models
- Error handling and retry mechanisms
"""

import json
import logging
from typing import TypeVar

import requests
from pydantic import BaseModel

from _production.config.config import CUSTOM_MODEL_CONFIG
from _production.utils.exceptions import (
    LLMError,
    LLMResponseError,
)

T = TypeVar("T", bound=BaseModel)


class CustomModelClient:
    """Client for interacting with a custom local LLM server."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        default_temperature: float = 0.0,
        default_max_tokens: int = 1024,
    ):
        """Initialize the custom model client.

        Args:
            base_url: Base URL of the model server
            timeout: Request timeout in seconds
            default_temperature: Default generation temperature
            default_max_tokens: Default maximum tokens to generate
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._session = requests.Session()

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text completion for the given prompt.

        Args:
            prompt: Text prompt for generation
            temperature: Temperature parameter (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            LLMError: If the API request fails
            LLMResponseError: If the response cannot be parsed
        """
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        try:
            response = self._session.post(
                f"{self.base_url}/generate",
                json={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("generated_text", "")
        except requests.exceptions.Timeout:
            raise LLMError("Request to model server timed out") from None
        except requests.exceptions.RequestException as error:
            raise LLMError(f"Failed to connect to model server: {error!s}") from error
        except json.JSONDecodeError as error:
            raise LLMResponseError(
                "Failed to parse response from model server"
            ) from error
        except Exception as error:
            raise LLMError(
                f"Unexpected error when calling model server: {error!s}"
            ) from error


class StructuredModelClient(CustomModelClient):
    """Client for structured data extraction using the custom model server."""

    def structured_generate(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Generate structured data from LLM using a schema.

        Args:
            messages: List of message objects (system/user)
            response_model: Pydantic model to validate response
            temperature: Temperature parameter (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Instance of response_model

        Raises:
            LLMError: If the API request fails
            LLMResponseError: If the response cannot be parsed
        """
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        # Get model schema for validation
        # model_schema = response_model.model_json_schema()

        # Implementation with retry logic for timeout errors
        max_retries = 2
        base_timeout = self.timeout

        logging.info(f"response_model: {response_model}")

        for attempt in range(max_retries + 1):
            try:
                # Prepare payload for the structured_generate endpoint
                payload = {
                    "messages": messages,
                    "response_model": response_model.model_json_schema(),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                logging.info(f"payload: {payload}")

                # Use increasing timeout for each retry
                current_timeout = base_timeout * (attempt + 1)

                logging.info(f"current_timeout: {current_timeout}")

                logging.debug(
                    f"Making structured_generate request with timeout {current_timeout}s (attempt {attempt+1}/{max_retries+1})"
                )

                response = self._session.post(
                    f"{self.base_url}/structured_generate",
                    json=payload,
                    timeout=current_timeout,
                )

                logging.info(f"response: {response}")

                response.raise_for_status()
                data = response.json()

                logging.info(f"data: {data}")

                # Check if response indicates an error but contains partial JSON
                if isinstance(data, dict) and "error" in data and "message" in data:
                    error_msg = data.get("message", "")
                    if error_msg and error_msg.strip().startswith("{"):
                        logging.warning(
                            "Received error with partial JSON. Attempting to extract JSON from error message."
                        )
                        try:
                            # Try to parse the JSON from the error message
                            extracted_json = json.loads(error_msg.strip())
                            if (
                                isinstance(extracted_json, dict)
                                and "summary" in extracted_json
                            ):
                                logging.info(
                                    "Successfully extracted JSON from error message"
                                )
                                return response_model.model_validate(extracted_json)
                        except json.JSONDecodeError:
                            logging.warning("Failed to parse JSON from error message")
                            pass

                # Parse the response into the requested model
                return response_model.model_validate(data)

            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    logging.warning(
                        f"Request timed out after {current_timeout}s, retrying with longer timeout..."
                    )
                    continue
                else:
                    logging.error(
                        f"Request to model server timed out after {max_retries+1} attempts"
                    )
                    raise LLMError("Request to model server timed out") from None

            except requests.exceptions.RequestException as error:
                raise LLMError(
                    f"Failed to connect to model server: {error!s}"
                ) from error

            except json.JSONDecodeError as error:
                raise LLMResponseError(
                    "Failed to parse response from model server"
                ) from error

            except Exception as error:
                if (
                    hasattr(error, "__module__")
                    and error.__module__ == "pydantic.error_wrappers"
                ):
                    raise LLMResponseError(
                        f"Failed to validate response with model: {error!s}"
                    ) from error
                raise LLMError(
                    f"Unexpected error when calling model server: {error!s}"
                ) from error

        # This line should never be reached due to the exception in the final iteration
        # but it's needed to satisfy the return type checker
        raise LLMError(
            "Failed to get response from model server after all retries"
        ) from None


# Create client instances
def get_custom_model_client() -> CustomModelClient:
    """Get a singleton instance of the custom model client."""
    return CustomModelClient(
        base_url=CUSTOM_MODEL_CONFIG["LLM_INSTANCE_URL"],
        timeout=CUSTOM_MODEL_CONFIG["TIMEOUT"],
        default_temperature=CUSTOM_MODEL_CONFIG["DEFAULT_TEMPERATURE"],
        default_max_tokens=CUSTOM_MODEL_CONFIG["DEFAULT_MAX_TOKENS"],
    )


def get_structured_model_client() -> StructuredModelClient:
    """Get a singleton instance of the structured model client."""
    return StructuredModelClient(
        base_url=CUSTOM_MODEL_CONFIG["LLM_INSTANCE_URL"],
        timeout=CUSTOM_MODEL_CONFIG["TIMEOUT"],
        default_temperature=CUSTOM_MODEL_CONFIG["DEFAULT_TEMPERATURE"],
        default_max_tokens=CUSTOM_MODEL_CONFIG["DEFAULT_MAX_TOKENS"],
    )
