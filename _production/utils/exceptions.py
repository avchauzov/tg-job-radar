"""Custom exception hierarchy for structured error handling.

This module defines a comprehensive set of exceptions for the application,
with specialized classes for LLM-related errors including rate limiting,
response validation, input validation, and parsing failures.
"""


class BaseError(Exception):
    """Base exception for all custom errors."""

    pass


class LLMError(BaseError):
    """Base exception for LLM-related errors."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when hitting rate limits."""

    pass


class LLMResponseError(LLMError):
    """Raised when receiving invalid responses."""

    pass


class LLMInputError(LLMError):
    """Raised when input validation fails."""

    pass


class LLMParsingError(LLMResponseError):
    """
    Raised when parsing LLM responses fails.

    This exception is specifically for JSON parsing errors or when the
    expected structure is not found in the LLM response.

    Attributes:
        response: The raw response that failed to parse
        field: The specific field that caused the error (if applicable)
        message: A descriptive error message
    """

    def __init__(
        self, message: str, response: str | None = None, field: str | None = None
    ):
        """Initialize a new LLMParsingError.

        Args:
            message: A descriptive error message explaining what went wrong
            response: The raw LLM response that failed to parse
            field: The specific field that caused the parsing error
        """
        self.response = response
        self.field = field
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        """Return a string representation of the error.

        Returns:
            A formatted error message including the field name if available.
        """
        if self.field:
            return f"{self.message} (field: {self.field})"
        return self.message
