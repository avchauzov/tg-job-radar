class BaseError(Exception):
    """Base exception for all custom errors"""

    pass


class LLMError(BaseError):
    """Base exception for LLM-related errors"""

    pass


class LLMRateLimitError(LLMError):
    """Raised when hitting rate limits"""

    pass


class LLMResponseError(LLMError):
    """Raised when receiving invalid responses"""

    pass


class LLMInputError(LLMError):
    """Raised when input validation fails"""

    pass
