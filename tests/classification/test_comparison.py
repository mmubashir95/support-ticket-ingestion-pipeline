"""Focused frozen-baseline comparison and error-analysis tests."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scipy.sparse import csr_matrix

from ticket_classification.comparison import (
    MODELS, class_errors, compare_per_class, confusion_pairs, extract_disagreements,
    extract_errors, feature_overlap, index_predictions, length_analysis, minority_analysis,
    near_duplicate_analysis, nearest_train_matches, nearest_train_rows,
    render_comparison_report, review_candidates, run_classical_model_comparison,
    top_features, validate_compatibility,
)
from ticket_classification.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[2] / 'artifacts/classification/run'


def row(record_id, actual, predicted):
    return {'record_id': record_id, 'actual_label': actual, 'predicted_label': predicted}


def records():
    return [SimpleNamespace(record_id=str(i), label=label, text=text,
                            source_fields=['subject', 'message'])
            for i, (label, text) in enumerate([('A', 'tiny'), ('B', 'a medium ticket'),
                                               ('C', 'a much longer ticket with detailed context'), ('A', 'same')])]


def predictions():
    lr = [row('0', 'A', 'A'), row('1', 'B', 'A'), row('2', 'C', 'A'), row('3', 'A', 'B')]
    svm = [row('3', 'A', 'B'), row('2', 'C', 'B'), row('1', 'B', 'B'), row('0', 'A', 'B')]
    return lr, svm


@pytest.mark.parametrize('key', ['dataset_version', 'split_version', 'feature_version',
                                 'feature_configuration', 'class_order', 'label_mapping', 'target', 'task'])
def test_compatibility_rejects_mismatches(key):
    manifests = {name: json.loads((ROOT / name / 'experiment_manifest.json').read_text()) for name in MODELS}
    metadata = json.loads((ROOT / 'tfidf/feature_metadata.json').read_text())
    configuration = json.loads((ROOT / 'tfidf/feature_config.json').read_text())
    classes = manifests[MODELS[0]]['class_order']
    validate_compatibility(manifests, metadata, configuration, classes)
    manifests[MODELS[1]][key] = 'incompatible'
    with pytest.raises(ValueError, match=key):
        validate_compatibility(manifests, metadata, configuration, classes)


def test_metric_alignment_and_known_deltas():
    classes = ['A', 'B', 'C']
    lr, svm = predictions()
    metrics = {name: evaluate_predictions([r['actual_label'] for r in rows],
                                         [r['predicted_label'] for r in rows], classes)[0]
               for name, rows in zip(MODELS, (lr, svm))}
    result = compare_per_class(metrics, classes)
    assert result[0]['class'] == 'A'
    assert result[0]['svm_minus_lr']['recall'] == -.5
    assert result[1]['svm_minus_lr']['recall'] == 1
    assert result[0][MODELS[0]]['support'] == 2
    metrics[MODELS[1]]['class_order'] = list(reversed(classes))
    with pytest.raises(ValueError, match='ordering'):
        compare_per_class(metrics, classes)


def test_confusion_pairs_rank_and_exclude_diagonal():
    matrix = {'row_labels': ['A', 'B', 'C'], 'column_labels': ['A', 'B', 'C'],
              'values': [[10, 2, 2], [3, 7, 0], [0, 1, 9]]}
    pairs = confusion_pairs(matrix)
    assert [(r['actual_class'], r['predicted_class'], r['error_count']) for r in pairs] == [
        ('B', 'A', 3), ('A', 'B', 2), ('A', 'C', 2), ('C', 'B', 1)]
    assert pairs[0]['error_rate'] == .3


def test_fp_fn_and_context():
    lr, _ = predictions()
    lr[1]['probabilities'] = [.8, .1, .1]
    errors = extract_errors(lr, records(), MODELS[0])
    assert [r['record_id'] for r in class_errors(errors, 'A', 'false_positive')] == ['1', '2']
    assert [r['record_id'] for r in class_errors(errors, 'A', 'false_negative')] == ['3']
    assert errors[0]['text'] == 'a medium ticket'
    assert errors[0]['probabilities'] == [.8, .1, .1]
    assert errors[0]['model_name'] == MODELS[0]


def test_disagreement_join_categories_and_exclusion():
    lr, svm = predictions()
    differing = extract_disagreements(lr, svm)
    assert [(r['record_id'], r['category']) for r in differing] == [
        ('0', 'lr_correct_svm_wrong'), ('1', 'svm_correct_lr_wrong'), ('2', 'both_wrong_different')]
    assert extract_disagreements(lr, list(reversed(lr))) == []
    with pytest.raises(ValueError, match='identities'):
        extract_disagreements(lr, svm[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        index_predictions(lr + lr[:1])
    svm[0]['actual_label'] = 'C'
    with pytest.raises(ValueError, match='actual labels'):
        extract_disagreements(lr, svm)


def test_feature_mapping_signed_binary_and_overlap():
    model = SimpleNamespace(n_features_in_=3, classes_=np.array(['A', 'B']), coef_=np.array([[2., -3., 1.]]))
    vectorizer = SimpleNamespace(get_feature_names_out=lambda: np.array(['alpha', 'beta', 'gamma']))
    features = top_features(model, vectorizer, 2)
    assert features['B']['positive'] == [{'feature': 'alpha', 'weight': 2.}, {'feature': 'gamma', 'weight': 1.}]
    assert features['A']['positive'] == [{'feature': 'beta', 'weight': 3.}]
    assert features['A']['negative'][0] == {'feature': 'alpha', 'weight': -2.}
    assert feature_overlap(dict.fromkeys(MODELS, features), ['A', 'B'])['B']['positive']['shared'] == ['alpha', 'gamma']
    with pytest.raises(ValueError, match='top_k'):
        top_features(model, vectorizer, 0)


def test_minority_all_smallest_ties():
    lr, svm = predictions()
    classes = ['A', 'B', 'C']
    metrics = {name: evaluate_predictions([r['actual_label'] for r in rows], [r['predicted_label'] for r in rows], classes)[0]
               for name, rows in zip(MODELS, (lr, svm))}
    errors = {name: extract_errors(rows, records(), name) for name, rows in zip(MODELS, (lr, svm))}
    result = minority_analysis({'A': 10, 'B': 2, 'C': 2}, compare_per_class(metrics, classes), errors)
    assert [r['class'] for r in result] == ['B', 'C']
    assert result[0][MODELS[0]]['false_negative_count'] == 1
    assert result[0][MODELS[1]]['false_positive_count'] == 3


def test_review_flags_are_deterministic_heuristics():
    lr, svm = predictions()
    texts = {r.record_id: r.text for r in records()}
    ambiguous, labels = review_candidates(lr, svm, texts, 3)
    assert (ambiguous, labels) == review_candidates(list(reversed(lr)), list(reversed(svm)), texts, 3)
    assert [r['record_id'] for r in labels] == ['3']
    assert labels[0]['review_reason'] == 'possible_label_issue'
    assert all(r['heuristic_only'] and r['requires_manual_review'] for r in ambiguous + labels)
    assert lr[3]['actual_label'] == 'A'
    assert ambiguous[0]['review_reason'] == 'possible_ambiguity'


def test_length_groups_and_error_rates():
    lr, svm = predictions()
    result = length_analysis(records(), dict(zip(MODELS, (lr, svm))), 4, 30)
    assert result['short']['record_count'] == 2
    assert result['long']['record_count'] == 1
    assert result['short'][MODELS[0]]['error_rate'] == .5
    assert result['long'][MODELS[0]]['elevated_vs_all'] is True


def test_nearest_train_matches_buckets_and_boundaries():
    train = csr_matrix([[1., 0., 0.], [0., 1., 0.]])
    heldout = csr_matrix([[2., 0., 0.],    # exact copy of train row 0 (unnormalised)
                          [.6, .8, 0.],    # cosine 0.8 to train row 1: boundary is near_copy
                          [.6, 0., .8],    # cosine 0.6 to train row 0: boundary is intermediate
                          [0., 0., 1.],    # orthogonal to train: novel, no neighbour
                          [0., 0., 0.]])   # empty row: novel, no neighbour
    best, index = nearest_train_matches(train, heldout, batch_size=2)
    assert best.tolist() == [1.0, .8, .6, 0.0, 0.0]
    assert index[:3].tolist() == [0, 1, 0]
    train_records = [SimpleNamespace(record_id='t0', label='A'), SimpleNamespace(record_id='t1', label='B')]
    held = [SimpleNamespace(record_id=str(i), label=label) for i, label in enumerate('ABACA')]
    rows = nearest_train_rows(held, train_records, best, index)
    assert [r['bucket'] for r in rows] == ['near_copy', 'near_copy', 'intermediate', 'novel', 'novel']
    assert [r['label_matches_nearest_train'] for r in rows] == [True, True, True, False, False]
    assert rows[4]['nearest_train_record_id'] is None and 'text' not in rows[0]
    with pytest.raises(ValueError, match='feature space'):
        nearest_train_matches(train, csr_matrix([[1., 0.]]))


def test_near_duplicate_analysis_scores_each_bucket_with_shared_evaluator():
    lr, svm = predictions()
    neighbours = [{'record_id': rid, 'bucket': bucket, 'label_matches_nearest_train': match}
                  for rid, bucket, match in [('0', 'near_copy', True), ('1', 'near_copy', True),
                                             ('2', 'novel', False), ('3', 'intermediate', True)]]
    result = near_duplicate_analysis(neighbours, dict(zip(MODELS, (lr, svm))), ['A', 'B', 'C'])
    assert [result[b]['record_count'] for b in ('all', 'near_copy', 'intermediate', 'novel')] == [4, 2, 1, 1]
    assert result['near_copy']['nearest_train_label_agreement'] == 1.0
    assert result['near_copy'][MODELS[0]]['accuracy'] == .5
    assert result['novel'][MODELS[1]]['accuracy'] == 0.0
    full = evaluate_predictions([r['actual_label'] for r in lr], [r['predicted_label'] for r in lr], ['A', 'B', 'C'])[0]
    assert result['all'][MODELS[0]]['macro_f1'] == full['macro_f1']
    with pytest.raises(ValueError, match='identities'):
        near_duplicate_analysis(neighbours[:3], dict(zip(MODELS, (lr, svm))), ['A', 'B', 'C'])


def test_frozen_workflow_report_deterministic_no_training(tmp_path, monkeypatch):
    import ticket_classification.classical as classical
    def forbidden(*args, **kwargs):
        raise AssertionError('Phase 2.6 must not retrain')
    monkeypatch.setattr(classical, 'train_logistic_regression', forbidden)
    monkeypatch.setattr(classical, 'train_linear_svm', forbidden)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    monkeypatch.setattr(LogisticRegression, 'fit', forbidden)
    monkeypatch.setattr(LinearSVC, 'fit', forbidden)
    monkeypatch.setattr(TfidfVectorizer, 'fit', forbidden)
    monkeypatch.setattr(TfidfVectorizer, 'fit_transform', forbidden)
    arguments = [ROOT / 'splits', ROOT / 'tfidf', ROOT / MODELS[0], ROOT / MODELS[1], tmp_path / 'evaluation']
    a = run_classical_model_comparison(*arguments, audit_path=ROOT / 'classification_dataset_audit.json')
    report = render_comparison_report(a)
    assert report == render_comparison_report(copy.deepcopy(a))
    assert report == (arguments[-1] / 'classical_model_comparison.md').read_text()
    for section in ['Experiment Context', 'Overall Metrics', 'Near-Duplicate Sensitivity',
                    'Per-Class Performance', 'Confusion Matrix Analysis',
                    'Most Confused Class Pairs', 'False Positive Analysis', 'False Negative Analysis',
                    'Minority-Class Analysis', 'Logistic Regression vs SVM Disagreements',
                    'Short / Long Ticket Analysis', 'Possible Ambiguous Tickets', 'Possible Label Issues',
                    'Feature Explainability', 'Training Time', 'CPU Inference Latency', 'Model Size',
                    'Model Issues vs Data-Quality Issues', 'Class Weighting Assessment', 'Classical Baseline Conclusion']:
        assert f'## {section}' in report
    assert a['overall']['test'][MODELS[1]]['macro_f1'] > .8
    assert a['conclusion']['selection_split'] == 'validation'
    assert len(list(arguments[-1].glob('*.jsonl'))) == 12
    for split in ('validation', 'test'):
        buckets = a['near_duplicates'][split]
        assert sum(buckets[b]['record_count'] for b in ('near_copy', 'intermediate', 'novel')) == buckets['all']['record_count']
        assert buckets['all'][MODELS[1]]['macro_f1'] == a['overall'][split][MODELS[1]]['macro_f1']
    assert a['conclusion']['ranking_holds_on_novel_validation'] is True
    assert a['context']['class_counts_full']['Change'] == 2922
    before = {path.name: path.read_bytes() for path in arguments[-1].iterdir()}
    repeated = run_classical_model_comparison(*arguments, audit_path=ROOT / 'classification_dataset_audit.json')
    assert repeated == a
    assert before == {path.name: path.read_bytes() for path in arguments[-1].iterdir()}


@pytest.mark.parametrize('mutation', ['ids', 'scores', 'metrics', 'confusion'])
def test_workflow_rejects_corrupt_outputs(tmp_path, mutation):
    import shutil
    directory = tmp_path / 'lr'
    shutil.copytree(ROOT / MODELS[0], directory)
    if mutation in ('ids', 'scores'):
        path = directory / 'validation_predictions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == 'ids':
            rows[0]['record_id'] = 'unknown'
        else:
            rows[0]['probabilities'][0] += .1
        path.write_text('\n'.join(json.dumps(r) for r in rows))
    else:
        path = directory / ('metrics.json' if mutation == 'metrics' else 'confusion_matrix.json')
        value = json.loads(path.read_text())
        if mutation == 'metrics':
            value['validation']['zero_division'] = 1
        else:
            value['validation']['values'][0][0] += 1
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        run_classical_model_comparison(ROOT / 'splits', ROOT / 'tfidf', directory,
                                       ROOT / MODELS[1], tmp_path / 'evaluation')


def test_output_cannot_overwrite_source_artifacts():
    with pytest.raises(ValueError, match='separate'):
        run_classical_model_comparison(ROOT / 'splits', ROOT / 'tfidf', ROOT / MODELS[0],
                                       ROOT / MODELS[1], ROOT / 'tfidf')
