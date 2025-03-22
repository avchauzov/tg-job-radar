"""Utility module for InfluxDB operations.

This module provides functions for storing metrics and custom data in InfluxDB.
"""

import time
from typing import Any

from influxdb_client.client.write.point import Point

from _production.config.config import INFLUXDB, INFLUXDB_WRITE_API


def store_metrics(
    measurement: str,
    fields: dict[str, Any],
    tags: dict[str, str] | None = None,
    timestamp: int | None = None,
) -> None:
    """Store metrics in InfluxDB.

    Args:
        measurement: The measurement name (e.g., 'script_performance', 'scraping_stats')
        fields: Dictionary of field names and their values
        tags: Optional dictionary of tag names and their values
        timestamp: Optional timestamp in nanoseconds. If not provided, current time is used
    """
    if not INFLUXDB_WRITE_API:
        raise RuntimeError("InfluxDB client is not initialized")

    point = Point(measurement)

    # Add tags if provided
    if tags:
        for tag_name, tag_value in tags.items():
            point.tag(tag_name, tag_value)

    # Add fields
    for field_name, field_value in fields.items():
        point.field(field_name, field_value)

    # Set timestamp
    if timestamp is None:
        timestamp = time.time_ns()
    point.time(timestamp)

    try:
        INFLUXDB_WRITE_API.write(bucket=INFLUXDB["BUCKET"], record=point)
    except Exception as error:
        raise RuntimeError(f"Failed to write metrics to InfluxDB: {error}") from error
