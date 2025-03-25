# -*- coding: utf-8 -*-
"""Prepare some API keys by setting up envs"""

import base64
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from langfuse import Langfuse

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)

# Set up Langfuse authentication
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")  # Read from .env file
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")  # Read from .env file

if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
    logger.warning("Langfuse API keys not found in .env file. Telemetry will not be enabled.")
    langfuse = None
else:
    # Initialize Langfuse SDK
    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host="https://cloud.langfuse.com"
    )
    logger.info("Langfuse SDK initialized successfully.")

    # Set up OpenTelemetry configuration
    LANGFUSE_AUTH = base64.b64encode(
        f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
    ).decode()

    # Configure OpenTelemetry basic settings
    os.environ["OTEL_SERVICE_NAME"] = "smolagents-cs"
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = "deployment.environment=development"

    # Configure OpenTelemetry endpoint and authentication
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
        f"Authorization=Basic {LANGFUSE_AUTH},"
        "Content-Type=application/x-protobuf"
    )

    # Set up TracerProvider and SpanProcessor
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    # Initialize SmolagentsInstrumentor
    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
    logger.info("OpenTelemetry instrumentation enabled successfully.")

# Set up other environment variables
logger.info("Set up env variables from .env file.")

# Export langfuse instance for use in other modules
__all__ = ["langfuse"]
