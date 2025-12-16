"""Utility functions for data handling"""

import json
from pathlib import Path
from typing import List, Dict, Tuple


def save_validation_and_examples(
    validation_set: List[Dict],
    few_shot_examples: List[Dict],
    output_dir: str = "outputs"
) -> Tuple[Path, Path]:
    """Save validation set and few-shot examples to JSON files

    Args:
        validation_set: Validation examples
        few_shot_examples: Few-shot examples
        output_dir: Output directory

    Returns:
        Tuple of (validation_path, examples_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save validation set
    val_path = output_path / "validation_set.json"
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(validation_set, f, ensure_ascii=False, indent=2)

    # Save few-shot examples
    examples_path = output_path / "few_shot_examples.json"
    with open(examples_path, 'w', encoding='utf-8') as f:
        json.dump(few_shot_examples, f, ensure_ascii=False, indent=2)

    return val_path, examples_path
