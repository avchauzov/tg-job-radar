# Custom Model Integration

This document explains how to use the custom LLM model integration instead of the Anthropic Claude API.

## Overview

The codebase has been updated to support using a local model server for all LLM operations, including:
- Job post detection and validation
- Structured data extraction from job posts
- CV-to-job matching with detailed scoring
- Text generation

## Server Requirements

The custom model server should:
1. Run on your local machine or a server accessible via network
2. Provide a `/generate` endpoint for text generation
3. Provide a `/structured_generate` endpoint for structured data extraction
4. Support the Claude-compatible message format

## Configuration

To enable the custom model, set these environment variables:

```bash
# Enable custom model (required)
USE_CUSTOM_MODEL=true

# Set custom model server URL (optional, defaults to production server)
CUSTOM_MODEL_URL=http://your-server-url:port
```

> **Note**: The system is preconfigured to use the production inference server at `http://54.254.18.117:8000` when no custom URL is specified.

## API Endpoints

The custom model server should implement these endpoints:

### 1. Text Generation Endpoint

**URL**: `/generate`

**Method**: POST

**Request Body**:
```json
{
  "prompt": "What is the capital of France?",
  "temperature": 0.7,
  "max_tokens": 64
}
```

**Response Format**:
```json
{
  "generated_text": "The capital of France is Paris."
}
```

### 2. Structured Data Extraction Endpoint

**URL**: `/structured_generate`

**Method**: POST

**Request Body**:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that extracts structured information."
    },
    {
      "role": "user",
      "content": "Extract data from: Jason is 30 years old"
    }
  ],
  "response_model": {
    "title": "UserDetail",
    "type": "object",
    "properties": {
      "name": {
        "title": "Name",
        "type": "string"
      },
      "age": {
        "title": "Age",
        "type": "integer"
      }
    },
    "required": ["name", "age"]
  },
  "temperature": 0.7,
  "max_tokens": 64
}
```

**Response Format**:
```json
{
  "name": "Jason",
  "age": 30
}
```

## Testing

To test the custom model integration:

```bash
# Set environment variables
export USE_CUSTOM_MODEL=true
export CUSTOM_MODEL_URL=http://localhost:8000

# Run the test script
python -m _production.utils.test_custom_model
```

## Fallback to Anthropic

If `USE_CUSTOM_MODEL` is not set to `true`, the codebase will fallback to using the Anthropic Claude API. In this case, you must set the `ANTHROPIC_API_KEY` environment variable.

## Implementation Details

The custom model integration uses:

1. `CustomModelClient`: A client for text generation using the `/generate` endpoint
2. `StructuredModelClient`: A client for structured data extraction using the `/structured_generate` endpoint
3. `InstructorClient`: An instructor-compatible wrapper that mimics the Anthropic API

These clients implement retry logic, error handling, and proper validation.
