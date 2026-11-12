"""Маленькие проверяемые примеры для одного эксперимента K=10."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_split_construction_preserves_source_rows_and_duplicates():
    from src import run_k10 as experiment

    data = {
        "train": [["повтор", "known"], ["повтор", "known"], ["near train", "near"]],
        "val": [["validation only", "known"], ["near val", "near"]],
        "test": [["test only", "known"], ["повтор", "near"], ["unused", "other"]],
        "oos_test": [["original", "oos"]],
    }
    groups = experiment.prepare_subsets(data, ["known"], ["near"])
    assert groups == {
        "train": data["train"][:2], "val": data["val"][:1],
        "known_test": data["test"][:1], "near_oos_test": data["test"][1:2],
        "original_oos_test": data["oos_test"],
    }
    with pytest.raises(ValueError, match="пересекаются"):
        experiment.prepare_subsets(data, ["known"], ["known"])


def test_threshold_is_largest_observed_score_accepting_at_least_target():
    from src import run_k10 as experiment

    scores = [i / 20 for i in range(20)]
    threshold = experiment.select_threshold(scores, 0.95)
    assert threshold == pytest.approx(0.05)
    assert experiment.accept_scores([0.049, 0.05, 0.9], threshold).tolist() == [False, True, True]
    tied = [0.1, 0.1] + [0.8] * 18
    assert experiment.select_threshold(tied, 0.95) == 0.1
    assert experiment.accept_scores(tied, 0.1).mean() == 1.0
    assert experiment.select_threshold([0.4], 0.95) == 0.4
    assert experiment.select_threshold([0.1, 0.9], 1.0) == 0.1
    for invalid in ([], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            experiment.select_threshold(invalid, 0.95)
    for invalid_target in (0, 1.01):
        with pytest.raises(ValueError):
            experiment.select_threshold([0.5], invalid_target)


def test_metric_denominators_include_rejected_and_misclassified_known():
    from src import run_k10 as experiment

    result = experiment.evaluate_decisions(
        ["a", "a", "b", "b"], ["a", "b", "b", "a"],
        [True, True, False, False], [True, False, False], [True, True], ["a", "b"],
    )
    assert result["known_n"] == 4
    assert result["known_rejected"] == 2
    assert result["known_correct_accepted"] == 1
    assert result["frr"] == 0.5
    assert result["correct_routing_rate"] == 0.25
    assert result["accuracy_no_reject"] == 0.5
    assert result["macro_f1_no_reject"] == 0.5
    assert result["original_oos_n"] == 3
    assert result["original_oos_accepted"] == 1
    assert result["far_original_oos"] == pytest.approx(1 / 3)
    assert result["near_oos_n"] == result["near_oos_accepted"] == 2
    assert result["far_near_oos"] == 1.0
    for known_accept, original_accept, near_accept in (([], [True], [True]), ([True], [], [True]), ([True], [True], [])):
        with pytest.raises(ValueError):
            experiment.evaluate_decisions(["a"], ["a"], known_accept, original_accept, near_accept, ["a"])


def test_training_vocabulary_and_cosine_use_only_individual_train_rows():
    import numpy as np
    from src import run_k10 as experiment

    config = {
        "vectorizer": {"type": "TfidfVectorizer", "ngram_range": [1, 1], "norm": "l2"},
        "classifier": {"type": "LogisticRegression", "C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    }
    train = [["apple", "a"], ["pear", "a"], ["train", "b"], ["bus", "b"]]
    vectorizer, classifier, train_matrix = experiment.fit_model(train, config, seed=17)
    vocabulary = dict(vectorizer.vocabulary_)
    idf = vectorizer.idf_.copy()
    predicted, confidence, cosine = experiment.score_queries(
        vectorizer, classifier, train_matrix, ["apple", "validationonly", "apple pear"],
    )
    assert vectorizer.vocabulary_ == vocabulary
    np.testing.assert_array_equal(vectorizer.idf_, idf)
    assert "validationonly" not in vocabulary
    assert train_matrix.shape[0] == len(train)
    assert cosine.tolist() == pytest.approx([1.0, 0.0, 1 / np.sqrt(2)])
    assert predicted[0] == "a"
    assert set(predicted) <= {"a", "b"}
    assert np.all((confidence >= 0.5) & (confidence <= 1))


def test_cli_end_to_end_metrics_match_saved_decisions(tmp_path):
    import csv
    import json
    import os
    import subprocess
    from sklearn.metrics import f1_score

    data_dir = Path(os.environ.get("CLINC150_DATA_DIR", ROOT / "data"))
    output = tmp_path / "result"
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(ROOT / "src/run_k10.py"),
         "--data-dir", str(data_dir), "--config", str(ROOT / "configs/experiment.json"),
         "--output-dir", str(output)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not result.stderr
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["k"] == 10 and metrics["seed"] == 17
    assert metrics["split_sizes"] == {"train": 1000, "val": 200, "known_test": 300, "near_oos_test": 300, "original_oos_test": 1000}
    assert len(metrics["methods"]) == 3
    with (output / "predictions.csv").open(encoding="utf-8-sig", newline="") as file:
        predictions = list(csv.DictReader(file))
    assert len(predictions) == 1800
    assert "текст" not in predictions[0]
    known = [row for row in predictions if row["группа"] == "known_test"]
    assert metrics["methods"][0]["macro_f1_no_reject"] == pytest.approx(f1_score(
        [row["истинная_метка"] for row in known], [row["прогноз"] for row in known],
        labels=metrics["known_intents"], average="macro",
    ))
    for method in metrics["methods"]:
        column = {"no_reject": "принят_без_отказа", "confidence": "принят_confidence", "cosine": "принят_cosine"}[method["method"]]
        for row in predictions:
            if method["method"] == "no_reject":
                expected_acceptance = True
            else:
                score_column = "максимальная_вероятность" if method["method"] == "confidence" else "максимальное_cosine"
                expected_acceptance = float(row[score_column]) >= method["threshold"]
            assert (row[column] == "True") == expected_acceptance
        assert method["known_n"] == 300
        assert method["known_rejected"] == sum(row[column] == "False" for row in known)
        assert method["known_correct_accepted"] == sum(row[column] == "True" and row["истинная_метка"] == row["прогноз"] for row in known)
        assert method["frr"] == method["known_rejected"] / 300
        assert method["correct_routing_rate"] == method["known_correct_accepted"] / 300
        for group, prefix, n in (("original_oos_test", "original_oos", 1000), ("near_oos_test", "near_oos", 300)):
            accepted = sum(row[column] == "True" for row in predictions if row["группа"] == group)
            assert method[prefix + "_n"] == n
            assert method[prefix + "_accepted"] == accepted
            assert method["far_" + prefix] == accepted / n
        val = [row for row in predictions if row["группа"] == "val"]
        assert method["validation_accepted"] == sum(row[column] == "True" for row in val)
        assert method["validation_acceptance"] == method["validation_accepted"] / 200
        if method["method"] != "no_reject":
            score_column = "максимальная_вероятность" if method["method"] == "confidence" else "максимальное_cosine"
            scores = sorted(float(row[score_column]) for row in val)
            assert method["threshold"] == scores[10]
            assert method["validation_acceptance"] >= 0.95
    assert (output / "metrics.csv").is_file()
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["configuration"]["experiments"][0]["seed"] == 17
    assert provenance["training"]["train_matrix_shape"][0] == 1000
