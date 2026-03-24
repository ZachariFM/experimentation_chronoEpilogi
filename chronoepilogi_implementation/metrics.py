
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np


def get_forecasting_metrics():

    metric_dict = {"mfe":lambda y_true, y_pred: np.mean(y_true-y_pred),
               "mae":mean_absolute_error,
               "rmse":lambda x,y:np.sqrt(mean_squared_error(x,y)),
               "r2": r2_score,
               "mape":lambda y_true, y_pred: 100 * mean_absolute_percentage_error(y_true, y_pred),
               "smape": lambda y_true, y_pred: 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))),
               "mase": lambda y_true, y_pred, in_sample: mean_absolute_error(y_true, y_pred)/ mean_absolute_error(in_sample[1:], in_sample[:-1])}
    return metric_dict



def compute_eq_tp(fs_set, causal_ground_truth):
    return sum([any([x in c for x in fs_set]) for c in causal_ground_truth])
def compute_eq_fn(fs_set, causal_ground_truth):
    return sum([all([x not in c for x in fs_set]) for c in causal_ground_truth])
def compute_eq_fp(fs_set, causal_ground_truth):
    return len(fs_set) - compute_eq_tp(fs_set, causal_ground_truth)


def evaluate_selection_metrics(fs_set, equivalence_classes=None, causal_ground_truth=None):
    metrics = dict()
    # size
    metrics["fs_size"] = len(fs_set)
    if causal_ground_truth is not None:
        # eq-precision, eq-recall, eq-f1
        tp = compute_eq_tp(fs_set, causal_ground_truth)
        fn = compute_eq_fn(fs_set, causal_ground_truth)
        fp = compute_eq_fp(fs_set, causal_ground_truth)
        eq_precision = tp / (tp + fp)
        eq_recall = tp / (tp + fn)
        eq_f1 = 2* tp / (2*tp + fn + fp)
        metrics["eq-tp"] = tp
        metrics["eq-fp"] = fp
        metrics["eq-fn"] = fn
        metrics["eq-precision"] = eq_precision
        metrics["eq-recall"] = eq_recall
        metrics["eq-f1"] = eq_f1

    if equivalence_classes is not None:
        # number of selected  TS overall
        total_size = sum([len(c) for c in equivalence_classes])
        metrics['total size'] = total_size
        
        if causal_ground_truth is not None:
            # precision, recall, f1
            detected = set([x for c in equivalence_classes for x in c])
            true = set([x for c in causal_ground_truth for x in c])
            tp = len(detected.intersection(true))
            fn = len(true.difference(detected))
            fp = len(detected) - tp
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2*tp / (2*tp + fp + fn)
            metrics["tp"] = tp
            metrics["fp"] = fp
            metrics["fn"] = fn
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1

    return metrics