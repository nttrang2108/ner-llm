"""Sampling utilities for creating validation and few-shot examples"""

import random
from typing import Dict, List


def create_validation_set(
    dev_data: List[Dict],
    size: int = 30,
    strategy: str = 'diverse',
    seed: int = 42
) -> List[Dict]:
    """Create validation set with various sampling strategies

    Args:
        dev_data: Development set
        size: Number of examples
        strategy: 'random', 'diverse', or 'balanced'
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    if strategy == 'random':
        return random.sample(dev_data, min(size, len(dev_data)))

    elif strategy == 'diverse':
        # Sample based on entity diversity
        scored = []
        for item in dev_data:
            gt = item['ground_truth']
            entity_count = sum(len(v) for v in gt.values())
            entity_types = sum(1 for v in gt.values() if len(v) > 0)
            score = entity_count * entity_types
            scored.append((score, item))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [item for _, item in scored[:size]]

    elif strategy == 'balanced':
        # Balance across topics
        by_topic = {}
        for item in dev_data:
            topic = item.get('topic', 'unknown')
            by_topic.setdefault(topic, []).append(item)

        per_topic = size // len(by_topic)
        result = []
        for items in by_topic.values():
            result.extend(random.sample(items, min(per_topic, len(items))))

        return result[:size]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_few_shot_examples(
    train_data: List[Dict],
    num_examples: int = 3,
    quality_filter: bool = True,
    max_text_length: int = 800,
    seed: int = 42
) -> List[Dict]:
    """Create high-quality few-shot examples

    Args:
        train_data: Training set
        num_examples: Number of examples to create
        quality_filter: Filter for quality examples
        max_text_length: Maximum text length
        seed: Random seed
    """
    random.seed(seed)

    # Filter candidates
    candidates = []
    for item in train_data:
        text = item['text']
        gt = item['ground_truth']

        # Quality criteria
        entity_count = sum(len(v) for v in gt.values())
        has_all_types = all(len(v) > 0 for v in gt.values())
        length_ok = len(text) <= max_text_length

        if quality_filter:
            if entity_count >= 3 and has_all_types and length_ok:
                candidates.append(item)
        else:
            if entity_count > 0 and length_ok:
                candidates.append(item)

    # Sample from candidates
    selected = random.sample(candidates, min(num_examples, len(candidates)))

    # Format as few-shot examples
    examples = []
    for item in selected:
        examples.append({
            'input': item['text'],
            'output': item['ground_truth']
        })

    return examples
