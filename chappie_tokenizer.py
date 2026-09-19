"""
Chappie Standard Tokenizer Loader
==================================
Loads a HuggingFace standard tokenizer with automatic cache configuration.
"""

import os
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_standard_tokenizer(model_name: str = "meta-llama/Meta-Llama-3-8B"):
    """
    Load a standard HuggingFace tokenizer.

    Recommended options:
        - "meta-llama/Meta-Llama-3-8B"   : 128k vocab, 176+ languages
        - "google/gemma-3-4b-it"          : 262k vocab, 140+ languages
        - "xlm-roberta-base"              : 250k vocab, 100 languages (no token required)
        - "google/mt5-base"               : 250k vocab, 101 languages

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        AutoTokenizer instance ready for streaming tokenization.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loaded tokenizer: {model_name}")
    logger.info(f"Vocab size: {tokenizer.vocab_size:,}")
    logger.info(f"Model max length: {tokenizer.model_max_length}")

    return tokenizer