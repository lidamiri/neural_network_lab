import numpy as np
import pytest

from nnlab.metrics import (
    accuracy,
    classification_metrics,
    confusion_matrix,
    per_class_scores,
    predictions_to_labels,
    regression_metrics,
    top_k_accuracy,
)
from nnlab.reporting import benchmark_table, classification_report, regression_report


def test_predictions_to_labels_for_binary_and_multiclass():
    assert predictions_to_labels(np.array([[0.2], [0.9]])).tolist() == [0, 1]
    assert predictions_to_labels(np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])).tolist() == [1, 0]


def test_accuracy_handles_empty_input():
    assert accuracy([], []) == 0.0
    assert accuracy([1, 0, 1], [1, 1, 1]) == pytest.approx(2 / 3)


def test_confusion_matrix_counts_pairs():
    matrix = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2], 3)
    expected = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(matrix, expected)


def test_per_class_scores_are_zero_for_absent_classes():
    matrix = np.array([[2, 0, 0], [0, 0, 0], [0, 0, 1]])
    precision, recall, f1 = per_class_scores(matrix)
    assert precision[1] == 0.0
    assert recall[1] == 0.0
    assert f1[1] == 0.0
    assert f1[0] == pytest.approx(1.0)


def test_classification_metrics_summary_fields():
    result = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], 2)
    assert result["accuracy"] == pytest.approx(0.75)
    assert len(result["confusion_matrix"]) == 2
    assert "macro_f1" in result


def test_top_k_accuracy_counts_membership_in_top_scores():
    scores = np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]])
    assert top_k_accuracy(scores, [0, 2], k=2) == pytest.approx(0.5)


def test_regression_metrics_perfect_and_constant_cases():
    values = np.array([1.0, 2.0, 3.0])
    perfect = regression_metrics(values, values)
    assert perfect["mse"] == pytest.approx(0.0)
    assert perfect["r2"] == pytest.approx(1.0)
    constant = regression_metrics(values, np.full(3, 2.0))
    assert constant["mae"] == pytest.approx(2.0 / 3.0)


def test_reports_render_expected_lines():
    metrics = classification_metrics([0, 1], [0, 1], 2)
    text = classification_report(metrics, ["zero", "one"])
    assert "accuracy" in text and "confusion matrix" in text and "zero" in text
    assert "rmse" in regression_report(regression_metrics([1.0, 2.0], [1.0, 2.5]))
    table = benchmark_table([{"name": "a", "value": 0.5}, {"name": "bb", "value": 0.25}])
    assert "0.2500" in table
    assert benchmark_table([]) == "no runs were executed"
