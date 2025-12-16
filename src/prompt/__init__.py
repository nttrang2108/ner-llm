"""Prompt templates for Vietnamese NER extraction"""

from .templates import (
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_chain_of_thought_prompt,
    build_custom_prompt,
    build_prompt,
)

__all__ = [
    'build_zero_shot_prompt',
    'build_few_shot_prompt',
    'build_chain_of_thought_prompt',
    'build_custom_prompt',
    'build_prompt',
]
