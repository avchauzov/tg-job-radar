"""
Instructor wrapper for the custom model client.

This module provides an instructor-compatible interface for the custom model client,
enabling structured data extraction with pydantic models using a similar API to
instructor's official clients.
"""

import logging
from typing import TypeVar

from pydantic import BaseModel

from _production.utils.custom_model import get_structured_model_client
from _production.utils.exceptions import LLMError, LLMResponseError

T = TypeVar("T", bound=BaseModel)


def from_custom_model(client=None):
    """Create an instructor-compatible client for structured data extraction.

    Args:
        client: Optional pre-configured custom model client

    Returns:
        InstructorClient: An instructor-compatible client instance
    """
    if client is None:
        client = get_structured_model_client()
    return InstructorClient(client)


class InstructorClient:
    """Instructor-compatible client for structured data extraction."""

    def __init__(self, client):
        """Initialize with a custom model client.

        Args:
            client: The base custom model client
        """
        self.client = client
        self.messages = Messages(self)

    def completions(self):
        """Placeholder for completions API compatibility."""
        return NotImplementedError("Completions API not supported")


class Messages:
    """Messages API wrapper for the custom model client."""

    def __init__(self, client):
        """Initialize with a custom model client.

        Args:
            client: The instructor client
        """
        self.client = client

    def create(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        response_model: type[T],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
    ) -> T:
        """Create a structured response using the custom model.

        Args:
            model: Model name (ignored, using the custom model server)
            system: System message content
            messages: List of message dictionaries
            response_model: Pydantic model for response validation
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter
            **kwargs: Additional parameters (ignored)

        Returns:
            Validated instance of the response model

        Raises:
            LLMError: If the API request fails
            LLMResponseError: If the response cannot be parsed
        """
        try:
            # Prepend the system message if not already in messages
            full_messages = [{"role": "system", "content": system}]
            for msg in messages:
                if msg.get("role") != "system":  # Don't duplicate system messages
                    full_messages.append(msg)

            # Call the structured_generate method
            return self.client.client.structured_generate(
                messages=full_messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            if isinstance(e, (LLMError, LLMResponseError)):
                raise
            logging.error(f"Error in instructor client: {e!s}")
            raise LLMError(f"Failed to create structured response: {e!s}")
