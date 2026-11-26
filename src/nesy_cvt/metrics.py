from __future__ import annotations

from collections.abc import Sequence


def _as_rows(logits: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = [list(row) for row in logits]
    if not rows:
        raise ValueError("logits must contain at least one row")
    width = len(rows[0])
    if width == 0:
        raise ValueError("logit rows must not be empty")
    if any(len(row) != width for row in rows):
        raise ValueError("all logit rows must have the same length")
    return rows


def accuracy_at_k(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    rows = _as_rows(logits)
    if len(rows) != len(labels):
        raise ValueError("logits and labels must have the same length")

    results: dict[str, float] = {}
    for k in topk:
        if k <= 0:
            raise ValueError("top-k values must be positive")
        hits = 0
        for row, label in zip(rows, labels):
            ranked = sorted(range(len(row)), key=row.__getitem__, reverse=True)
            if int(label) in ranked[:k]:
                hits += 1
        results[f"top{k}"] = hits / len(rows)
    return results


def macro_f1(predictions: Sequence[int], labels: Sequence[int], num_classes: int) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    scores: list[float] = []
    for class_id in range(num_classes):
        tp = sum(1 for pred, label in zip(predictions, labels) if pred == class_id and label == class_id)
        fp = sum(1 for pred, label in zip(predictions, labels) if pred == class_id and label != class_id)
        fn = sum(1 for pred, label in zip(predictions, labels) if pred != class_id and label == class_id)
        if tp == 0 and fp == 0 and fn == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / num_classes


def macro_precision_recall(
    predictions: Sequence[int], labels: Sequence[int], num_classes: int
) -> dict[str, float]:
    """Return unweighted one-vs-rest precision and recall across all classes."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not predictions:
        raise ValueError("predictions and labels must not be empty")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    precisions: list[float] = []
    recalls: list[float] = []
    for class_id in range(num_classes):
        true_positive = sum(
            pred == class_id and label == class_id
            for pred, label in zip(predictions, labels)
        )
        predicted_positive = sum(pred == class_id for pred in predictions)
        actual_positive = sum(label == class_id for label in labels)
        precisions.append(true_positive / predicted_positive if predicted_positive else 0.0)
        recalls.append(true_positive / actual_positive if actual_positive else 0.0)
    return {
        "precision_macro": sum(precisions) / num_classes,
        "recall_macro": sum(recalls) / num_classes,
    }


def _binary_roc_auc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    """Compute ROC AUC from average ranks, assigning average ranks to ties."""
    positives = sum(positive)
    negatives = len(positive) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both positive and negative examples")

    order = sorted(range(len(scores)), key=scores.__getitem__)
    positive_rank_sum = 0.0
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(positive[order[index]] for index in range(cursor, end))
        cursor = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def multiclass_ovr_macro_auc(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int], num_classes: int
) -> float:
    """Compute unweighted macro one-vs-rest ROC AUC from class probabilities."""
    rows = _as_rows(probabilities)
    if len(rows) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if len(rows[0]) != num_classes:
        raise ValueError("probability width must equal num_classes")

    per_class_auc = []
    for class_id in range(num_classes):
        class_scores = [row[class_id] for row in rows]
        class_labels = [int(label) == class_id for label in labels]
        per_class_auc.append(_binary_roc_auc(class_scores, class_labels))
    return sum(per_class_auc) / num_classes


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    num_bins: int = 15,
) -> float:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct flags must have the same length")
    if not confidences:
        raise ValueError("confidences must not be empty")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    total = len(confidences)
    ece = 0.0
    for index in range(num_bins):
        lower = index / num_bins
        upper = (index + 1) / num_bins
        in_bin = [
            i
            for i, confidence in enumerate(confidences)
            if lower < confidence <= upper or (index == 0 and confidence == 0.0)
        ]
        if not in_bin:
            continue
        accuracy = sum(1 for i in in_bin if correct[i]) / len(in_bin)
        avg_confidence = sum(confidences[i] for i in in_bin) / len(in_bin)
        ece += (len(in_bin) / total) * abs(accuracy - avg_confidence)
    return ece
