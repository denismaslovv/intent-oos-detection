"""Один эксперимент CLINC150: K=10, seed=17, два правила отказа."""

import argparse
import csv
import json
import math
import platform
import sys
import warnings
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_json  # noqa: E402
from src.generate_class_sets import generate_config, sha256  # noqa: E402


def prepare_subsets(data, known_intents, near_intents):
    """Фильтрация исходныъ разбиений без перемешивания и удаления повторов."""
    known, near = set(known_intents), set(near_intents)
    if known & near:
        raise ValueError("Known и Held-out пересекаются")
    return {
        "train": [row for row in data["train"] if row[1] in known],
        "val": [row for row in data["val"] if row[1] in known],
        "known_test": [row for row in data["test"] if row[1] in known],
        "near_oos_test": [row for row in data["test"] if row[1] in near],
        "original_oos_test": list(data["oos_test"]),
    }


def select_threshold(validation_scores, target_acceptance):
    """Максимальный наблюдаемый порог с принятием не ниже цели."""
    scores = np.asarray(validation_scores, dtype=float)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Нужны непустые конечные validation scores")
    if not 0 < target_acceptance <= 1:
        raise ValueError("Целевая доля должна лежать в (0, 1]")
    required = math.ceil(target_acceptance * len(scores))
    return float(np.sort(scores)[len(scores) - required])


def accept_scores(scores, threshold):
    return np.asarray(scores) >= threshold


def evaluate_decisions(y_true, y_pred, known_accept, original_accept, near_accept, labels):
    """Считаем доли по всем запросам соответствующей группы, не только принятым."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    known_accept, original_accept, near_accept = [
        np.asarray(values, dtype=bool) for values in (known_accept, original_accept, near_accept)
    ]
    if not (len(y_true) == len(y_pred) == len(known_accept)):
        raise ValueError("Длины known-массивов различаются")
    if not all(len(values) for values in (y_true, original_accept, near_accept)):
        raise ValueError("Знаменатель метрики не может быть нулевым")
    rejected = int((~known_accept).sum())
    correct_accepted = int(((y_true == y_pred) & known_accept).sum())
    return {
        "accuracy_no_reject": float(accuracy_score(y_true, y_pred)),
        "macro_f1_no_reject": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "known_n": len(y_true),
        "known_rejected": rejected,
        "known_correct_accepted": correct_accepted,
        "frr": rejected / len(y_true),
        "correct_routing_rate": correct_accepted / len(y_true),
        "original_oos_n": len(original_accept),
        "original_oos_accepted": int(original_accept.sum()),
        "far_original_oos": float(original_accept.mean()),
        "near_oos_n": len(near_accept),
        "near_oos_accepted": int(near_accept.sum()),
        "far_near_oos": float(near_accept.mean()),
    }


def fit_model(train_rows, config, seed):
    """Обучить один TF-IDF и один классификатор исключительно на known train."""
    vectorizer_params = {
        key: value for key, value in config["vectorizer"].items() if key != "type"
    }
    vectorizer_params["ngram_range"] = tuple(vectorizer_params.get("ngram_range", (1, 1)))
    classifier_params = {
        key: value for key, value in config["classifier"].items() if key != "type"
    }
    vectorizer = TfidfVectorizer(**vectorizer_params)
    train_matrix = vectorizer.fit_transform([row[0] for row in train_rows])
    classifier = LogisticRegression(**classifier_params, random_state=seed)
    # Несходимость — ошибка запуска, а не основание подстраивать модель по test.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(train_matrix, [row[1] for row in train_rows])
    return vectorizer, classifier, train_matrix


def score_queries(vectorizer, classifier, train_matrix, texts):
    """Без дообучения получить метку и оба score; сравнение с отдельными train-текстами."""
    matrix = vectorizer.transform(texts)
    probabilities = classifier.predict_proba(matrix)
    predicted = classifier.classes_[probabilities.argmax(axis=1)]
    similarities = cosine_similarity(matrix, train_matrix, dense_output=False)
    max_cosine = similarities.max(axis=1).toarray().ravel()
    return predicted, probabilities.max(axis=1), max_cosine


def run_experiment(config_path, data_dir, output_dir):
    """Проверить протокол, обучить модель и сохранить метрики с решениями без сырых текстов."""
    config_path, data_dir, output_dir = map(Path, (config_path, data_dir, output_dir))
    data_dir = data_dir.resolve()
    config = load_json(config_path.resolve())
    if generate_config(data_dir / "data_full.json", data_dir / "domains.json", config_path) != config:
        raise ValueError("Конфигурация отличается от воспроизводимо сформированной")
    if config["threshold_fit_split"] != "val" or config["dataset"]["known_splits"] != {
        "train": "train", "validation": "val", "test": "test",
    }:
        raise ValueError("Нужны исходные train/val/test и настройка порога по val")
    if config["dataset"]["original_oos_test"] != "oos_test" or config["dataset"]["near_oos_test"] != "test":
        raise ValueError("Изменены источники OOS test")
    run = next(run for run in config["experiments"] if run["seed"] == 17)
    known, near = run["known_intents"]["10"], config["near_oos_intents"]
    data = load_json(data_dir / "data_full.json")
    groups = prepare_subsets(data, known, near)
    expected_groups = (
        ("train", known, 100), ("val", known, 20),
        ("known_test", known, 30), ("near_oos_test", near, 30),
    )
    for name, intents, per_class in expected_groups:
        if Counter(row[1] for row in groups[name]) != Counter({label: per_class for label in intents}):
            raise ValueError(f"Неверный состав или размер группы {name}")
    if len(groups["original_oos_test"]) != 1000 or any(row[1] != "oos" for row in groups["original_oos_test"]):
        raise ValueError("Нужны неизменные 1000 Original OOS-test")
    if any(not isinstance(row[0], str) or not row[0].strip() for rows in groups.values() for row in rows):
        raise ValueError("В выбранных данных есть пустой или нестроковый текст")

    vectorizer, classifier, train_matrix = fit_model(groups["train"], config, seed=17)
    scored = {
        "val": score_queries(
            vectorizer, classifier, train_matrix, [row[0] for row in groups["val"]],
        )
    }
    target = config["target_known_validation_acceptance"]
    thresholds = {
        "no_reject": None,
        "confidence": select_threshold(scored["val"][1], target),
        "cosine": select_threshold(scored["val"][2], target),
    }
    # Только после фиксации порогов вычисляем test scores; никакой перенастройки ниже нет.
    for group in ("known_test", "original_oos_test", "near_oos_test"):
        scored[group] = score_queries(
            vectorizer, classifier, train_matrix, [row[0] for row in groups[group]],
        )
    decisions = {}
    methods = []
    for method, threshold in thresholds.items():
        score_index = 1 if method == "confidence" else 2
        decisions[method] = {
            group: (
                np.ones(len(groups[group]), dtype=bool) if threshold is None
                else accept_scores(values[score_index], threshold)
            )
            for group, values in scored.items()
        }
        accepted = decisions[method]
        metrics = evaluate_decisions(
            [row[1] for row in groups["known_test"]], scored["known_test"][0],
            accepted["known_test"], accepted["original_oos_test"], accepted["near_oos_test"], known,
        )
        methods.append({
            "method": method, "threshold": threshold,
            "validation_n": len(groups["val"]), "validation_accepted": int(accepted["val"].sum()),
            "validation_acceptance": float(accepted["val"].mean()),
            **metrics, "macro_f1_core_no_reject": metrics["macro_f1_no_reject"],
        })

    source_indices = {
        "val": [i for i, row in enumerate(data["val"]) if row[1] in known],
        "known_test": [i for i, row in enumerate(data["test"]) if row[1] in known],
        "near_oos_test": [i for i, row in enumerate(data["test"]) if row[1] in near],
        "original_oos_test": list(range(len(data["oos_test"]))),
    }
    predictions = []
    for group, (predicted, confidence, cosine) in scored.items():
        source_split = "val" if group == "val" else "oos_test" if group == "original_oos_test" else "test"
        for i, row in enumerate(groups[group]):
            predictions.append({
                "группа": group, "исходное_разбиение": source_split,
                "индекс_в_разбиении": source_indices[group][i], "истинная_метка": row[1], "прогноз": predicted[i],
                "максимальная_вероятность": float(confidence[i]), "максимальное_cosine": float(cosine[i]),
                "принят_без_отказа": bool(decisions["no_reject"][group][i]),
                "принят_confidence": bool(decisions["confidence"][group][i]),
                "принят_cosine": bool(decisions["cosine"][group][i]),
            })
    collisions = [
        {"known_train_subset_index": i, "near_test_source_index": source_indices["near_oos_test"][j],
         "known_label": train_row[1], "heldout_label": test_row[1]}
        for i, train_row in enumerate(groups["train"])
        for j, test_row in enumerate(groups["near_oos_test"]) if train_row[0] == test_row[0]
    ]
    result = {
        "k": 10, "seed": 17, "known_intents": known, "near_oos_intents": near,
        "target_validation_acceptance": target, "decision_rule": "score >= threshold",
        "split_sizes": {name: len(rows) for name, rows in groups.items()},
        "methods": methods, "known_train_near_test_exact_collisions": collisions,
    }
    provenance = {
        "configuration": config, "config_sha256": sha256(config_path),
        "data_sha256": {name: sha256(data_dir / name) for name in ("data_full.json", "domains.json")},
        "source_sha256": {
            path.name: sha256(path) for path in (
                Path(__file__), ROOT / "src/data_loader.py", ROOT / "src/generate_class_sets.py",
            )
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{package: version(package) for package in ("numpy", "scipy", "scikit-learn", "pandas", "pytest")}},
        "training": {"train_matrix_shape": list(train_matrix.shape), "classifier_iterations": classifier.n_iter_.tolist(),
                     "vectorizer_params": vectorizer.get_params(), "classifier_params": classifier.get_params()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("metrics.json", result), ("provenance.json", provenance)):
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8",
        )
    with (output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    names = {
        "method": "Метод", "threshold": "Порог", "validation_acceptance": "Принятие known val (доля)",
        "accuracy_no_reject": "Accuracy без отказа (доля)", "macro_f1_no_reject": "Macro-F1 без отказа",
        "frr": "FRR known test (доля)", "correct_routing_rate": "Правильная маршрутизация known test (доля)",
        "far_original_oos": "FAR Original OOS (доля)", "far_near_oos": "FAR near OOS (доля)",
    }
    method_names = {"no_reject": "Без отказа", "confidence": "Порог вероятности", "cosine": "Порог косинусного сходства"}
    with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(names.values()))
        writer.writeheader()
        writer.writerows(
            {label: method_names[row[key]] if key == "method" else row[key] for key, label in names.items()}
            for row in methods
        )
    return result


def main():
    """Запустить только оговорённую конфигурацию K=10, seed=17."""
    parser = argparse.ArgumentParser(description="CLINC150: K=10, seed=17, baseline и два метода отказа")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/k10_seed17")
    args = parser.parse_args()
    result = run_experiment(args.config, args.data_dir, args.output_dir)
    print(f"Готово: K={result['k']}, seed={result['seed']}. Метрики и решения: {args.output_dir}")


if __name__ == "__main__":
    main()
