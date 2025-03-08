# -*- coding: utf-8 -*-
"""Prepare some API keys by setting up envs"""

import logging
import json
import os

logger = logging.getLogger(__name__)
import os
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

key_pairs = json.load(open(f"{BASE_DIR}/../../../cs-deep-research/.api_keys.json", "r"))
for n, k in key_pairs.items():
    os.environ[n] = k
LANGFUSE_AUTH=base64.b64encode(f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()).decode()
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # EU data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

#
# from opentelemetry.sdk.trace import TracerProvider
# from openinference.instrumentation.smolagents import SmolagentsInstrumentor
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor
#
# trace_provider = TracerProvider()
# trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
#
# SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
print(f"SET UP {list(key_pairs.keys())}")
