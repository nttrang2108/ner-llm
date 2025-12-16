"""Evaluation utilities for Vietnamese NER

This module provides comprehensive evaluation metrics and utilities for Named Entity Recognition,
including fuzzy matching, JSON parsing, and result comparison.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from difflib import SequenceMatcher


# ============================================================================
# Core Accuracy Calculation (Legacy - kept for backward compatibility)
# ============================================================================

def calculate_accuracy(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Calculate accuracy and detailed metrics per entity type
    
    Returns metrics as ratios (0-1), not percentages.
    """
    correct = 0
    total = len(ground_truth)
    entity_types = ['person', 'organizations', 'address']
    
    # Initialize per-type counters
    per_type_metrics = {
        entity_type: {'tp': 0, 'fp': 0, 'fn': 0, 'correct_samples': 0} 
        for entity_type in entity_types
    }
    
    # Overall counters
    overall_tp = overall_fp = overall_fn = 0
    
    # Calculate metrics for each prediction
    for pred, gt in zip(predictions, ground_truth):
        if _exact_match(pred, gt):
            correct += 1

        for entity_type in entity_types:
            pred_entities = set(_normalize_entities(pred.get(entity_type, [])))
            gt_entities = set(_normalize_entities(gt.get(entity_type, [])))
            
            # Fuzzy match entities
            matched = _fuzzy_match_entities(pred_entities, gt_entities)
            
            # Check if entity type is completely correct
            if len(matched) == len(gt_entities) and len(pred_entities) == len(gt_entities):
                per_type_metrics[entity_type]['correct_samples'] += 1
            
            # Calculate TP, FP, FN
            tp = len(matched)
            fn = len(gt_entities) - tp
            fp = len(pred_entities) - tp
            
            per_type_metrics[entity_type]['tp'] += tp
            per_type_metrics[entity_type]['fp'] += fp
            per_type_metrics[entity_type]['fn'] += fn
            
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
    
    # Calculate overall accuracy (0-1)
    accuracy = (correct / total) if total > 0 else 0

    # Calculate per-entity-type metrics (0-1)
    per_type_results = {}
    for entity_type in entity_types:
        metrics = per_type_metrics[entity_type]
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        accuracy_type = (metrics['correct_samples'] / total) if total > 0 else 0

        per_type_results[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy_type,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'correct_samples': metrics['correct_samples'],
            'total_samples': total,
            'precision_pct': f"{precision * 100:.1f}%",
            'recall_pct': f"{recall * 100:.1f}%",
            'f1_pct': f"{f1 * 100:.1f}%",
            'accuracy_pct': f"{accuracy_type * 100:.1f}%"
        }

    # Calculate overall entity-level metrics (0-1)
    overall_precision = (overall_tp / (overall_tp + overall_fp)) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = (overall_tp / (overall_tp + overall_fn)) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'percentage': f"{accuracy * 100:.1f}%",
        'per_entity_type': per_type_results,
        'overall_entity_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'tp': overall_tp,
            'fp': overall_fp,
            'fn': overall_fn,
            'precision_pct': f"{overall_precision * 100:.1f}%",
            'recall_pct': f"{overall_recall * 100:.1f}%",
            'f1_pct': f"{overall_f1 * 100:.1f}%"
        }
    }


def _exact_match(pred: Dict, gt: Dict, similarity_threshold: float = 0.8) -> bool:
    """Check if prediction exactly matches ground truth using fuzzy matching"""
    for key in ['person', 'organizations', 'address']:
        pred_entities = set(_normalize_entities(pred.get(key, [])))
        gt_entities = set(_normalize_entities(gt.get(key, [])))
        matched = _fuzzy_match_entities(pred_entities, gt_entities)
        if len(matched) != len(gt_entities):
            return False
    return True


def _normalize_entities(entities: list) -> list:
    """Convert entity list to strings, handling various types"""
    normalized = []
    for entity in entities:
        if isinstance(entity, str):
            normalized.append(entity)
        elif isinstance(entity, dict):
            # Extract text from dict entities
            if 'text' in entity:
                normalized.append(str(entity['text']))
            elif 'name' in entity:
                normalized.append(str(entity['name']))
            else:
                normalized.append(str(entity))
        elif entity is not None:
            normalized.append(str(entity))
    return normalized


def _fuzzy_match_entities(pred_set: set, gt_set: set) -> set:
    """Match entities using fuzzy string similarity (threshold=0.8)"""
    matched = set()
    for pred_entity in pred_set:
        for gt_entity in gt_set:
            if SequenceMatcher(None, pred_entity, gt_entity).ratio() > 0.8:
                matched.add(gt_entity)
                break
    return matched


# ============================================================================
# Result Printing and Formatting
# ============================================================================

def print_evaluation_results(results: Dict, method_name: str = "Method", use_logger: bool = False):
    """Print detailed evaluation results for a single method"""
    import logging
    logger = logging.getLogger(__name__)
    output = logger.info if use_logger else print
    
    output(f"\n{'='*80}")
    output(f"  {method_name} Evaluation Results")
    output(f"{'='*80}")
    
    # Exact match accuracy
    output(f"\nExact Match Accuracy: {results['accuracy']:.1%}")
    output(f"  Correct: {results['correct']}/{results['total']}")
    
    # Overall entity metrics
    overall = results['overall_entity_metrics']
    output(f"\nOverall Entity-Level Metrics:")
    output(f"  Precision: {overall['precision']:.1%}")
    output(f"  Recall:    {overall['recall']:.1%}")
    output(f"  F1-Score:  {overall['f1']:.1%}")
    output(f"  TP/FP/FN: {overall['tp']}/{overall['fp']}/{overall['fn']}")
    
    # Per-type metrics
    output(f"\nPer-Entity-Type Metrics:")
    output(f"{'Type':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP/FP/FN':<12}")
    output("-" * 80)
    
    per_type = results['per_entity_type']
    for entity_type, metrics in per_type.items():
        print(
            f"{entity_type:<15} "
            f"{metrics['precision']:.1%}     "
            f"{metrics['recall']:.1%}     "
            f"{metrics['f1']:.1%}     "
            f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']}"
        )


def print_comparison_table(results: Dict, sort_by: str = 'f1'):
    """Print comparison table sorted by specified metric"""
    # Sort methods
    if sort_by == 'f1':
        sorted_methods = sorted(
            results.items(),
            key=lambda x: x[1]['overall_entity_metrics']['f1'],
            reverse=True
        )
    elif sort_by == 'accuracy':
        sorted_methods = sorted(
            results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
    else:
        sorted_methods = list(results.items())

    print("\n" + "="*100)
    print(f"  Method Comparison (sorted by {sort_by.upper()})")
    print("="*100)

    # Overall metrics
    print(f"\n{'Method':<20} {'Exact Match':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 100)
    for method, result in sorted_methods:
        overall = result.get('overall_entity_metrics', {})
        print(
            f"{method:<20} "
            f"{result.get('percentage', 'N/A'):<15} "
            f"{overall.get('precision_pct', 'N/A'):<12} "
            f"{overall.get('recall_pct', 'N/A'):<12} "
            f"{overall.get('f1_pct', 'N/A'):<12}"
        )

    # Per-type F1 comparison
    print("\n" + "="*100)
    print("  Per-Type F1-Score Comparison")
    print("="*100)
    print(f"\n{'Method':<20} {'Person':<12} {'Organizations':<15} {'Address':<12}")
    print("-" * 100)

    entity_types = ['person', 'organizations', 'address']
    for method, result in sorted_methods:
        per_type = result.get('per_entity_type', {})
        print(
            f"{method:<20} "
            f"{per_type.get('person', {}).get('f1_pct', 'N/A'):<12} "
            f"{per_type.get('organizations', {}).get('f1_pct', 'N/A'):<15} "
            f"{per_type.get('address', {}).get('f1_pct', 'N/A'):<12}"
        )

    print("\n" + "="*100)


# ============================================================================
# JSON Response Parsing
# ============================================================================

def parse_ner_response(response_text: str) -> Dict[str, List[str]]:
    """Parse JSON response from LLM with robust markdown handling
    
    Handles:
    - Markdown code blocks (```json, ```)
    - Extra text before/after JSON
    - Key variations (organizations/organization, address/addresses)
    """
    default_result = {
        "person": [],
        "organizations": [],
        "address": []
    }

    # Remove markdown code blocks
    cleaned_text = response_text.strip()
    
    if cleaned_text.startswith('```'):
        first_newline = cleaned_text.find('\n')
        if first_newline != -1:
            cleaned_text = cleaned_text[first_newline + 1:]
    
    if '```' in cleaned_text:
        last_backticks = cleaned_text.rfind('```')
        if last_backticks != -1:
            cleaned_text = cleaned_text[:last_backticks]
    
    cleaned_text = cleaned_text.strip()

    # Extract JSON using regex
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text, re.DOTALL)

    if not json_match:
        return default_result

    try:
        parsed = json.loads(json_match.group())
        
        # Handle key variations
        result = {
            "person": parsed.get("person", []),
            "organizations": parsed.get("organizations", parsed.get("organization", [])),
            "address": parsed.get("address", parsed.get("addresses", []))
        }

        # Ensure all values are lists
        for key in result:
            if not isinstance(result[key], list):
                result[key] = [str(result[key])] if result[key] else []

        return result

    except json.JSONDecodeError:
        return default_result


def save_json_with_numpy_conversion(data: Any, filepath: str):
    """Save data to JSON, converting numpy types to Python types"""
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    converted_data = convert_numpy(data)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    return filepath


# ============================================================================
# Comprehensive Metrics (Recommended for production use)
# ============================================================================

def compute_entity_metrics_with_fuzzy(
    predictions: List[Dict],
    ground_truths: List[Dict],
    entity_type: str,
    use_fuzzy_match: bool = True,
    similarity_threshold: float = 0.8
) -> Dict:
    """Compute precision/recall/F1 for specific entity type with optional fuzzy matching"""
    true_positives = false_positives = false_negatives = correct_samples = 0
    total_samples = len(ground_truths)

    for pred, gt in zip(predictions, ground_truths):
        pred_entities = set(pred.get(entity_type, []))
        gt_entities = set(gt.get(entity_type, []))

        if use_fuzzy_match:
            matched = _fuzzy_match_entities_with_threshold(pred_entities, gt_entities, similarity_threshold)
            tp = len(matched)
            fn = len(gt_entities) - tp
            fp = len(pred_entities) - tp

            if tp == len(gt_entities) and len(pred_entities) == len(gt_entities):
                correct_samples += 1
        else:
            pred_entities_norm = set(_normalize_entity(e) for e in pred_entities)
            gt_entities_norm = set(_normalize_entity(e) for e in gt_entities)

            tp = len(pred_entities_norm & gt_entities_norm)
            fp = len(pred_entities_norm - gt_entities_norm)
            fn = len(gt_entities_norm - pred_entities_norm)

            if pred_entities_norm == gt_entities_norm:
                correct_samples += 1

        true_positives += tp
        false_positives += fp
        false_negatives += fn

    precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
    recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    accuracy = (correct_samples / total_samples) if total_samples > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'correct_samples': correct_samples,
        'total_samples': total_samples,
        'support': true_positives + false_negatives
    }


def _normalize_entity(entity: str) -> str:
    """Normalize entity string (lowercase, stripped)"""
    return entity.strip().lower()


def _fuzzy_match_entities_with_threshold(
    pred_set: set,
    gt_set: set,
    similarity_threshold: float = 0.8
) -> set:
    """Match entities using configurable similarity threshold"""
    matched = set()
    for pred_entity in pred_set:
        for gt_entity in gt_set:
            if SequenceMatcher(None, pred_entity, gt_entity).ratio() > similarity_threshold:
                matched.add(gt_entity)
                break
    return matched


def compute_comprehensive_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    entity_types: List[str] = None,
    similarity_threshold: float = 0.8
) -> Dict:
    """Calculate comprehensive accuracy and metrics with fuzzy matching
    
    Returns:
    - Overall exact match accuracy
    - Per-entity-type metrics (precision, recall, F1, accuracy)
    - Overall entity-level metrics (micro-averaged)
    
    All metrics returned as percentages (0-100).
    """
    if entity_types is None:
        entity_types = ['person', 'organizations', 'address']
    
    correct = 0
    total = len(ground_truth)

    # Per-type counters
    per_type_metrics = {
        entity_type: {'tp': 0, 'fp': 0, 'fn': 0, 'correct_samples': 0}
        for entity_type in entity_types
    }

    # Overall counters
    overall_tp = overall_fp = overall_fn = 0

    # Process each prediction
    for pred, gt in zip(predictions, ground_truth):
        # Check exact match with fuzzy matching
        exact_match_flag = True
        for key in entity_types:
            pred_entities = set(pred.get(key, []))
            gt_entities = set(gt.get(key, []))
            
            matched = _fuzzy_match_entities_with_threshold(pred_entities, gt_entities, similarity_threshold)
            if len(matched) != len(gt_entities):
                exact_match_flag = False
                break
        
        if exact_match_flag:
            correct += 1

        # Per-type metrics
        for entity_type in entity_types:
            pred_entities = set(pred.get(entity_type, []))
            gt_entities = set(gt.get(entity_type, []))

            matched = _fuzzy_match_entities_with_threshold(pred_entities, gt_entities, similarity_threshold)

            if len(matched) == len(gt_entities) and len(pred_entities) == len(gt_entities):
                per_type_metrics[entity_type]['correct_samples'] += 1

            tp = len(matched)
            fn = len(gt_entities) - tp
            fp = len(pred_entities) - tp

            per_type_metrics[entity_type]['tp'] += tp
            per_type_metrics[entity_type]['fp'] += fp
            per_type_metrics[entity_type]['fn'] += fn

            overall_tp += tp
            overall_fp += fp
            overall_fn += fn

    # Calculate overall accuracy (as percentage)
    accuracy = (correct / total) * 100 if total > 0 else 0

    # Calculate per-type metrics (as percentages)
    per_type_results = {}
    for entity_type in entity_types:
        metrics = per_type_metrics[entity_type]
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']

        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        accuracy_type = (metrics['correct_samples'] / total) * 100 if total > 0 else 0

        per_type_results[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy_type,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'correct_samples': metrics['correct_samples'],
            'total_samples': total,
            'precision_pct': f"{precision:.1f}%",
            'recall_pct': f"{recall:.1f}%",
            'f1_pct': f"{f1:.1f}%",
            'accuracy_pct': f"{accuracy_type:.1f}%"
        }

    # Calculate overall metrics (as percentages)
    overall_precision = (overall_tp / (overall_tp + overall_fp)) * 100 if (overall_tp + overall_fp) > 0 else 0
    overall_recall = (overall_tp / (overall_tp + overall_fn)) * 100 if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'percentage': f"{accuracy:.1f}%",
        'per_entity_type': per_type_results,
        'overall_entity_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'tp': overall_tp,
            'fp': overall_fp,
            'fn': overall_fn,
            'precision_pct': f"{overall_precision:.1f}%",
            'recall_pct': f"{overall_recall:.1f}%",
            'f1_pct': f"{overall_f1:.1f}%"
        }
    }


def print_comprehensive_comparison(
    results: Dict,
    entity_types: List[str] = None
):
    """Print professional comparison table with comprehensive metrics"""
    if entity_types is None:
        entity_types = ['person', 'organizations', 'address']
    
    print("\n" + "="*100)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*100)

    # Overall exact match accuracy
    print("\nOVERALL EXACT MATCH ACCURACY:")
    print("-"*100)
    print(f"{'Method':<25} {'Accuracy':<15} {'Correct/Total':<20}")
    print("-"*100)
    for method, result in results.items():
        accuracy_data = result.get('comprehensive_metrics', result)
        if accuracy_data:
            correct = accuracy_data.get('correct', 'N/A')
            total = accuracy_data.get('total', 'N/A')
            percentage = accuracy_data.get('percentage', 'N/A')
            print(f"{method:<25} {percentage:<15} {correct}/{total:<20}")

    # Overall entity metrics
    print("\nOVERALL ENTITY-LEVEL METRICS:")
    print("-"*100)
    print(f"{'Method':<25} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-"*100)
    for method, result in results.items():
        overall = result.get('comprehensive_metrics', result).get('overall_entity_metrics', {})
        if overall:
            print(f"{method:<25} {overall.get('precision_pct', 'N/A'):<15} "
                  f"{overall.get('recall_pct', 'N/A'):<15} "
                  f"{overall.get('f1_pct', 'N/A'):<15}")

    # Per-type metrics
    print("\nPER-ENTITY-TYPE METRICS:")
    print("-"*100)
    for entity_type in entity_types:
        print(f"\n  {entity_type.upper()}:")
        print(f"  {'Method':<25} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Accuracy':<15}")
        print("  " + "-"*98)
        for method, result in results.items():
            per_type = result.get('comprehensive_metrics', result).get('per_entity_type', {}).get(entity_type, {})
            if per_type:
                print(f"  {method:<25} {per_type.get('precision_pct', 'N/A'):<15} "
                      f"{per_type.get('recall_pct', 'N/A'):<15} "
                      f"{per_type.get('f1_pct', 'N/A'):<15} "
                      f"{per_type.get('accuracy_pct', 'N/A'):<15}")

    print("\n" + "="*100)
