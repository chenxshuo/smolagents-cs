# -*- coding: utf-8 -*-
""""""

import logging
logger = logging.getLogger(__name__)

# Load model directly
from smolagents import api_key_setup
from transformers import AutoTokenizer, AutoModelForCausalLM
model = "facebook/opt-125m"
# model = "Qwen/QwQ-32B"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)