import numpy as np


def predictions_to_labels(outputs):
    outputs = np.asarray(outputs, dtype=float)
    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)
    if outputs.shape[1] == 1:
        return (outputs[:, 0] >= 0.5).astype(int)
    return np.argmax(outputs, axis=1)


def accuracy(true_labels, predicted_labels):
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    if true_labels.size == 0:
        return 0.0
    return float(np.mean(true_labels == predicted_labels))


def top_k_accuracy(scores, true_labels, k=2):
    scores = np.asarray(scores, dtype=float)
    true_labels = np.asarray(true_labels, dtype=int)
    k = min(int(k), scores.shape[1])
    top = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean(np.any(top == true_labels[:, None], axis=1)))


def confusion_matrix(true_labels, predicted_labels, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (np.asarray(true_labels, dtype=int), np.asarray(predicted_labels, dtype=int)), 1)
    return matrix


def per_class_scores(matrix):
    true_positive = np.diag(matrix).astype(float)
    predicted_total = matrix.sum(axis=0).astype(float)
    actual_total = matrix.sum(axis=1).astype(float)
    precision = np.divide(true_positive, predicted_total, out=np.zeros_like(true_positive), where=predicted_total > 0)
    recall = np.divide(true_positive, actual_total, out=np.zeros_like(true_positive), where=actual_total > 0)
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return precision, recall, f1


def classification_metrics(true_labels, predicted_labels, num_classes, scores=None):
    matrix = confusion_matrix(true_labels, predicted_labels, num_classes)
    precision, recall, f1 = per_class_scores(matrix)
    result = {
        "accuracy": accuracy(true_labels, predicted_labels),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
    }
    if scores is not None and num_classes > 2:
        result["top2_accuracy"] = top_k_accuracy(scores, true_labels, k=2)
    return result


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(len(y_pred), -1)
    errors = y_pred - y_true
    mse = float(np.mean(errors ** 2))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(errors))),
        "r2": float(r2),
    }
