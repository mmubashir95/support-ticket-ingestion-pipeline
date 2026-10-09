import numpy as np
import pytest

from ticket_classification.calibration import (
    SigmoidScoreCalibrator,
    TemperatureScaler,
    build_reliability_bins,
    evaluate_calibration,
    expected_calibration_error,
    fit_sigmoid_score_calibration,
    fit_temperature_scaling,
    multiclass_brier_score,
    negative_log_likelihood,
    negative_log_likelihood_from_logits,
    softmax_logits,
    validate_probabilities,
)


def test_perfect_predictions_have_zero_calibration_losses():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0)
    assert multiclass_brier_score(probabilities, labels) == pytest.approx(0.0)
    assert negative_log_likelihood(probabilities, labels) == pytest.approx(0.0)


def test_known_multiclass_brier_and_nll_conventions():
    probabilities = np.array([[0.8, 0.2], [0.25, 0.75]])
    labels = np.array([0, 1])
    assert multiclass_brier_score(probabilities, labels) == pytest.approx(
        ((0.2**2 + 0.2**2) + (0.25**2 + 0.25**2)) / 2
    )
    assert negative_log_likelihood(probabilities, labels) == pytest.approx(
        -(np.log(0.8) + np.log(0.75)) / 2
    )


def test_overconfident_and_underconfident_examples_have_known_ece():
    overconfident_wrong = np.array([[0.9, 0.1], [0.1, 0.9]])
    underconfident_correct = np.array([[0.6, 0.4], [0.4, 0.6]])
    assert expected_calibration_error(
        overconfident_wrong, np.array([1, 0])
    ) == pytest.approx(0.9)
    assert expected_calibration_error(
        underconfident_correct, np.array([0, 1])
    ) == pytest.approx(0.4)


def test_reliability_bins_retain_empty_bins_and_include_one_in_last_bin():
    rows = build_reliability_bins(
        np.array([[1.0, 0.0], [0.4, 0.6]]), np.array([0, 1]), bins=10
    )
    assert len(rows) == 10
    assert rows[-1]["count"] == 1
    assert rows[-1]["upper_inclusive"] is True
    assert rows[0]["mean_confidence"] is None
    assert sum(row["count"] for row in rows) == 2


@pytest.mark.parametrize(
    "probabilities",
    [
        [[0.5, 0.4]],
        [[-0.1, 1.1]],
        [[np.nan, np.nan]],
        [[1.0]],
        [0.5, 0.5],
    ],
)
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError):
        validate_probabilities(probabilities)


def test_incorrect_label_dimensions_are_rejected():
    with pytest.raises(ValueError):
        evaluate_calibration([[0.5, 0.5]], [2])
    with pytest.raises(ValueError):
        validate_probabilities([[0.5, 0.5]], expected_classes=4)


def test_extreme_probabilities_have_finite_nll():
    value = negative_log_likelihood([[0.0, 1.0]], [0])
    assert np.isfinite(value)
    assert value > 100


def test_logit_nll_matches_softmax_probability_nll():
    logits = np.array([[1000.0, 999.0], [-1000.0, -999.0]])
    labels = np.array([0, 1])
    probabilities = softmax_logits(logits)
    assert negative_log_likelihood_from_logits(logits, labels) == pytest.approx(
        negative_log_likelihood(probabilities, labels)
    )


def test_temperature_fitting_is_positive_deterministic_and_non_mutating():
    logits = np.array(
        [[4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 3.0], [3.0, 0.0]]
    )
    labels = np.array([0, 0, 1, 1, 1])
    original = logits.copy()
    first = fit_temperature_scaling(logits, labels)
    second = fit_temperature_scaling(logits, labels)
    assert first.temperature > 0
    assert first.temperature == pytest.approx(second.temperature)
    np.testing.assert_array_equal(logits, original)
    np.testing.assert_array_equal(
        logits.argmax(axis=1), first.transform(logits).argmax(axis=1)
    )
    np.testing.assert_allclose(first.transform(logits).sum(axis=1), 1.0)
    assert TemperatureScaler.from_dict(first.to_dict()) == first


def test_sigmoid_score_calibration_produces_reloadable_probabilities():
    scores = np.array(
        [
            [3.0, -1.0, -2.0],
            [2.0, -0.5, -1.0],
            [-1.0, 3.0, -2.0],
            [-0.5, 2.0, -1.0],
            [-1.0, -2.0, 3.0],
            [-0.5, -1.0, 2.0],
        ]
    )
    labels = np.array([0, 0, 1, 1, 2, 2])
    calibrator = fit_sigmoid_score_calibration(
        scores, labels, ("a", "b", "c")
    )
    reloaded = SigmoidScoreCalibrator.from_dict(calibrator.to_dict())
    probabilities = reloaded.transform(scores)
    assert probabilities.shape == scores.shape
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
