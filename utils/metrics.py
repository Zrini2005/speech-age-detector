"""Evaluation metrics for age estimation with child/adult differentiation."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, classification_report
from scipy.stats import pearsonr
from typing import List, Tuple, Dict, Optional
import pandas as pd


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics for age estimation.
    
    Args:
        y_true: Ground truth ages
        y_pred: Predicted ages
        
    Returns:
        Dictionary with MAE, RMSE, and Pearson correlation
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Pearson correlation
    if len(y_true) > 1:
        corr, p_value = pearsonr(y_true, y_pred)
    else:
        corr, p_value = 0.0, 1.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'pearson_r': corr,
        'pearson_p': p_value
    }


def is_child(age: float, threshold: int = 18) -> bool:
    """Determine if age is child (under threshold)."""
    return age < threshold


def compute_child_adult_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    child_threshold: int = 18
) -> Dict[str, float]:
    """
    Compute metrics for child vs adult binary classification.
    
    This is the KEY metric for differentiating children from adults.
    
    Args:
        y_true: Ground truth ages
        y_pred: Predicted ages
        child_threshold: Age below which is considered child (default: 18)
        
    Returns:
        Dictionary with child/adult classification metrics
    """
    # Convert to binary labels (True = child, False = adult)
    true_child = np.array([is_child(age, child_threshold) for age in y_true])
    pred_child = np.array([is_child(age, child_threshold) for age in y_pred])
    
    # Calculate metrics
    tp = np.sum(true_child & pred_child)  # True positives (correctly identified children)
    tn = np.sum(~true_child & ~pred_child)  # True negatives (correctly identified adults)
    fp = np.sum(~true_child & pred_child)  # False positives (adults misclassified as children)
    fn = np.sum(true_child & ~pred_child)  # False negatives (children misclassified as adults)
    
    total = len(y_true)
    
    # Calculate metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    
    # Child-specific metrics
    child_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    child_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    child_f1 = 2 * child_precision * child_recall / (child_precision + child_recall) \
               if (child_precision + child_recall) > 0 else 0
    
    # Adult-specific metrics  
    adult_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    adult_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    adult_f1 = 2 * adult_precision * adult_recall / (adult_precision + adult_recall) \
               if (adult_precision + adult_recall) > 0 else 0
    
    return {
        'child_adult_accuracy': accuracy,
        'child_precision': child_precision,
        'child_recall': child_recall,
        'child_f1': child_f1,
        'adult_precision': adult_precision,
        'adult_recall': adult_recall,
        'adult_f1': adult_f1,
        'true_children': int(np.sum(true_child)),
        'pred_children': int(np.sum(pred_child)),
        'true_adults': int(np.sum(~true_child)),
        'pred_adults': int(np.sum(~pred_child)),
        'confusion_child_adult': {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }
    }


def age_to_group(age: float) -> str:
    """
    Convert age to age group.
    
    Args:
        age: Age in years
        
    Returns:
        Age group label
    """
    if age < 13:
        return 'child'
    elif age < 20:
        return 'teen'
    elif age < 40:
        return 'young_adult'
    elif age < 60:
        return 'middle_aged'
    else:
        return 'senior'


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, any]:
    """
    Compute classification metrics for age groups.
    
    Args:
        y_true: Ground truth ages
        y_pred: Predicted ages
        
    Returns:
        Dictionary with accuracy, confusion matrix, etc.
    """
    # Convert to age groups
    true_groups = [age_to_group(age) for age in y_true]
    pred_groups = [age_to_group(age) for age in y_pred]
    
    # Compute accuracy
    accuracy = np.mean([t == p for t, p in zip(true_groups, pred_groups)])
    
    # Confusion matrix
    labels = ['child', 'teen', 'young_adult', 'middle_aged', 'senior']
    cm = confusion_matrix(true_groups, pred_groups, labels=labels)
    
    # Per-class metrics
    class_metrics = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'labels': labels,
        'class_metrics': class_metrics
    }


def format_metrics_table(metrics: Dict[str, Dict]) -> pd.DataFrame:
    """
    Format metrics into a pandas DataFrame for display.
    
    Args:
        metrics: Dictionary mapping model names to their metrics
        
    Returns:
        Formatted DataFrame
    """
    rows = []
    for model_name, model_metrics in metrics.items():
        row = {
            'Model': model_name,
            'MAE': f"{model_metrics.get('mae', 0):.2f}",
            'RMSE': f"{model_metrics.get('rmse', 0):.2f}",
            'Pearson r': f"{model_metrics.get('pearson_r', 0):.3f}",
            'Accuracy': f"{model_metrics.get('accuracy', 0):.2%}",
            'Avg Time (s)': f"{model_metrics.get('avg_time', 0):.3f}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df
