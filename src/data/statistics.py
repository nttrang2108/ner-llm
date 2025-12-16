"""Dataset statistics and analysis utilities"""

from typing import Dict, List


def analyze_dataset_statistics(data: List[Dict]) -> Dict:
    """Analyze dataset statistics

    Args:
        data: List of NER examples

    Returns:
        Dictionary with statistics
    """
    if not data:
        return {
            'total_examples': 0,
            'entity_counts': {},
            'avg_entities_per_example': 0,
            'avg_text_length': 0,
            'examples_with_all_types': 0
        }

    entity_counts = {'person': 0, 'organizations': 0, 'address': 0}
    total_entities = 0
    total_text_length = 0
    examples_with_all = 0

    for item in data:
        gt = item['ground_truth']
        text = item.get('text', '')

        # Count entities
        for entity_type in entity_counts:
            count = len(gt.get(entity_type, []))
            entity_counts[entity_type] += count
            total_entities += count

        # Check if example has all types
        if all(len(gt.get(k, [])) > 0 for k in entity_counts):
            examples_with_all += 1

        # Text length
        total_text_length += len(text)

    return {
        'total_examples': len(data),
        'entity_counts': entity_counts,
        'total_entities': total_entities,
        'avg_entities_per_example': total_entities / len(data),
        'avg_text_length': total_text_length / len(data),
        'examples_with_all_types': examples_with_all
    }
