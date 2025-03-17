# -*- coding: utf-8 -*-
""""""

import logging

from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)

# Load model directly
model = "facebook/opt-125m"
# model = "Qwen/QwQ-32B"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)
