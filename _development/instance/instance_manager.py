"""Instance management utilities for LLM server.

This module provides functions for managing the lifecycle of the LLM server instance:
- Starting the instance with retry logic
- Status checking with retries
"""

import logging
import time

import requests
from pydantic import BaseModel, Field


class InstanceStatus(BaseModel):
    """Response model for instance status.

    Matches the server's EC2StatusResponse model with the following fields:
    - instance_id: EC2 instance ID
    - state: Current instance state (pending, running, shutting-down, terminated, stopping, stopped)
    - public_dns: Public DNS name if available
    - system_status: System status check result (ok, impaired, insufficient-data, not-applicable, initializing)
    - instance_status: Instance status check result (ok, impaired, insufficient-data, not-applicable, initializing)
    - message: Operation result message
    """

    instance_id: str = Field(..., description="EC2 instance ID")
    state: str = Field(..., description="Current instance state")
    public_dns: str | None = Field(None, description="Public DNS name if available")
    system_status: str | None = Field(None, description="System status check result")
    instance_status: str | None = Field(
        None, description="Instance status check result"
    )
    message: str = Field(..., description="Operation result message")


class InstanceManager:
    """Manages the lifecycle of the LLM instance."""

    def __init__(self, base_url: str, timeout: int = 60):
        """Initialize the instance manager.

        Args:
            base_url: Base URL of the model server
            timeout: Request timeout in seconds
        """
        if not base_url:
            raise ValueError("Base URL cannot be empty")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        logging.info(f"Initialized InstanceManager with base URL: {self.base_url}")

    def _check_status(
        self,
        instance_id: str,
        region: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        max_retries: int = 10,
        sleep_time: int = 30,
    ) -> InstanceStatus:
        """Check instance status with retries.

        Args:
            instance_id: AWS EC2 instance ID
            region: AWS region
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            max_retries: Maximum number of retries
            sleep_time: Time to sleep between retries in seconds

        Returns:
            Instance status
        """
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/status"
                logging.info(
                    f"Checking instance status at {url} (attempt {attempt + 1}/{max_retries})"
                )

                # Prepare request data
                request_data = {
                    "instance_id": instance_id,
                    "region": region,
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key,
                }

                # Make request with explicit headers and method
                response = requests.post(
                    url,
                    json=request_data,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    timeout=self.timeout,
                    allow_redirects=False,
                )

                # Log request and response details
                logging.info("Request method: POST")
                logging.info(f"Request URL: {url}")
                logging.info(f"Request headers: {response.request.headers}")
                logging.info(f"Response status: {response.status_code}")
                logging.info(f"Response headers: {response.headers}")

                if response.status_code == 405:
                    logging.error(
                        "Received 405 Method Not Allowed. This indicates the request was not sent as POST."
                    )
                    raise requests.exceptions.RequestException(
                        "Request method was changed from POST to GET"
                    )

                response.raise_for_status()
                status = InstanceStatus.model_validate(response.json())

                if status.state == "running":
                    logging.info("Instance is running")
                    return status

                if attempt < max_retries - 1:
                    logging.info(
                        f"Instance not running yet (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                else:
                    logging.warning("Instance failed to start after all retries")
                    return status

            except requests.exceptions.RequestException as error:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"Failed to check status (attempt {attempt + 1}/{max_retries}): {error!s}. "
                        f"Retrying in {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                else:
                    logging.error(
                        f"Failed to check status after all retries: {error!s}"
                    )
                    return InstanceStatus(
                        instance_id=instance_id,
                        state="error",
                        public_dns=None,
                        system_status=None,
                        instance_status=None,
                        message=str(error),
                    )

        return InstanceStatus(
            instance_id=instance_id,
            state="error",
            public_dns=None,
            system_status=None,
            instance_status=None,
            message="Max retries exceeded",
        )

    def start_instance(
        self,
        instance_id: str,
        region: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
    ) -> InstanceStatus:
        """Start the LLM instance with retry logic.

        Args:
            instance_id: AWS EC2 instance ID
            region: AWS region
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key

        Returns:
            Instance status after starting
        """
        try:
            # Log the parameters (without sensitive data)
            logging.info("Starting instance with parameters:")
            logging.info(f"  instance_id: {instance_id}")
            logging.info(f"  region: {region}")
            logging.info(
                f"  aws_access_key_id: {'*' * len(aws_access_key_id) if aws_access_key_id else 'None'}"
            )
            logging.info(
                f"  aws_secret_access_key: {'*' * len(aws_secret_access_key) if aws_secret_access_key else 'None'}"
            )

            # Validate required parameters
            if not instance_id:
                raise ValueError("instance_id is required")
            if not region:
                raise ValueError("region is required")
            if not aws_access_key_id:
                raise ValueError("aws_access_key_id is required")
            if not aws_secret_access_key:
                raise ValueError("aws_secret_access_key is required")

            url = f"{self.base_url}/start"
            logging.info(f"Starting instance at {url}")

            # Use a longer timeout for the initial start request
            start_timeout = 300  # 5 minutes
            logging.info(
                f"Using extended timeout of {start_timeout} seconds for instance startup"
            )

            response = self._session.post(
                url,
                json={
                    "instance_id": instance_id,
                    "region": region,
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key,
                },
                timeout=start_timeout,
            )
            response.raise_for_status()

            logging.info(
                "Instance start request successful, waiting for instance to become operational..."
            )

            # Wait for instance to start with more retries and longer intervals
            return self._check_status(
                instance_id=instance_id,
                region=region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                max_retries=20,  # Increased from 10 to 20
                sleep_time=60,  # Increased from 30 to 60 seconds
            )

        except requests.exceptions.RequestException as error:
            logging.error(f"Failed to start instance: {error!s}")
            return InstanceStatus(
                instance_id=instance_id,
                state="error",
                public_dns=None,
                system_status=None,
                instance_status=None,
                message=str(error),
            )
