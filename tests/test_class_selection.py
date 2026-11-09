"""Проверки зафиксированного near-OOS протокола без CSV кандидатов."""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.generate_class_sets import generate_config  # noqa: E402


def input_path(filename: str) -> Path:
    for path in (ROOT / "data" / filename, ROOT / filename):
        if path.is_file():
            return path
    pytest.fail(f"Отсутствует {filename}: положите его в data/ или корень проекта")


@pytest.fixture(scope="module")
def paths():
    return input_path("data_full.json"), input_path("domains.json"), ROOT / "configs/experiment.json"


@pytest.fixture(scope="module")
def config(paths):
    return generate_config(*paths)


@pytest.fixture(scope="module")
def dataset(paths):
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_checksums(config, paths):
    for path in paths[:2]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config["dataset"]["checksums_sha256"][path.name]


def test_frozen_config_and_determinism(config, paths):
    assert config == json.loads(paths[2].read_text(encoding="utf-8"))
    assert generate_config(*paths) == config


def test_distinct_sets_and_anchor_rotation(config):
    pools = config["eligible_known_by_domain"]
    k10 = [set(run["known_intents"]["10"]) for run in config["experiments"]]
    assert len({frozenset(s) for s in k10}) == 3
    assert max(len(a & b) for i, a in enumerate(k10) for b in k10[i + 1:]) <= 5
    for domain, pool in pools.items():
        picked = {run["anchors_by_domain"][domain] for run in config["experiments"]}
        assert picked <= set(pool)
        assert len(picked) == min(3, len(pool))


def test_sizes_nesting_and_no_overlap(config, paths):
    domains = json.loads(paths[1].read_text(encoding="utf-8"))
    near = set(config["near_oos_intents"])
    assert len(near) == len(domains) == 10
    assert len(config["experiments"]) == 3
    assert config["seeds"] == [17, 42, 123]
    for run in config["experiments"]:
        sets = {k: set(run["known_intents"][str(k)]) for k in (10, 20, 30)}
        assert sets[10] < sets[20] < sets[30]
        assert sets[10] == set(run["anchors_by_domain"].values())
        for k, known in sets.items():
            assert len(known) == len(run["known_intents"][str(k)]) == k
            assert known.isdisjoint(near)
            assert all(len(set(members) & known) == k // 10 for members in domains.values())


def test_each_heldout_has_eligible_close_known(config, paths):
    domains = json.loads(paths[1].read_text(encoding="utf-8"))
    for domain, heldout in config["near_oos_by_domain"].items():
        assert heldout in domains[domain]
        pool = config["eligible_known_by_domain"][domain]
        assert pool
        assert set(pool) <= set(domains[domain]) - {heldout}
        for run in config["experiments"]:
            anchor = run["anchors_by_domain"][domain]
            assert anchor in pool
            assert anchor in run["known_intents"]["10"]


def test_original_splits_and_oos_unchanged(config, dataset):
    assert {s: len(dataset[s]) for s in ("train", "val", "test", "oos_test")} == {
        "train": 15000, "val": 3000, "test": 4500, "oos_test": 1000,
    }
    near = set(config["near_oos_intents"])
    near_test = [tuple(row) for row in dataset["test"] if row[1] in near]
    original_oos = [tuple(row) for row in dataset["oos_test"]]
    assert len(near_test) == config["expected_test_sizes"]["near_oos"] == 300
    assert len(original_oos) == config["expected_test_sizes"]["original_oos"] == 1000
    for run in config["experiments"]:
        for k in (10, 20, 30):
            known = set(run["known_intents"][str(k)])
            assert not known & near
            assert [tuple(row) for row in dataset["test"] if row[1] in near] == near_test
            assert [tuple(row) for row in dataset["oos_test"]] == original_oos
            for split, count in (("train", 100), ("val", 20), ("test", 30)):
                selected = Counter(label for _, label in dataset[split] if label in known)
                assert len(selected) == k
                assert set(selected.values()) == {count}


def test_rejects_changed_dataset(paths, tmp_path):
    changed = tmp_path / "data_full.json"
    changed.write_bytes(paths[0].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Контрольная сумма"):
        generate_config(changed, paths[1], paths[2])


def test_rejects_invalid_heldout(paths, tmp_path):
    config = json.loads(paths[2].read_text(encoding="utf-8"))
    config["near_oos_by_domain"]["utility"] = "invalid_label"
    config["near_oos_intents"][8] = "invalid_label"
    bad = tmp_path / "invalid.json"
    bad.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Некорректный Held-out"):
        generate_config(paths[0], paths[1], bad)


def test_rejects_heldout_in_eligible_pool(paths, tmp_path):
    config = json.loads(paths[2].read_text(encoding="utf-8"))
    config["eligible_known_by_domain"]["utility"].append(config["near_oos_by_domain"]["utility"])
    bad = tmp_path / "invalid.json"
    bad.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Недопустимый Known"):
        generate_config(paths[0], paths[1], bad)


def test_cli_check_without_csv(paths):
    original = paths[2].read_bytes()
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/generate_class_sets.py"),
         "--data", str(paths[0]), "--domains", str(paths[1]),
         "--protocol", str(paths[2]), "--output", str(paths[2]), "--check"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert paths[2].read_bytes() == original


def test_cli_detects_tampered_experiments(paths, tmp_path):
    config = json.loads(paths[2].read_text(encoding="utf-8"))
    config["experiments"][0]["known_intents"]["10"].pop()
    altered = tmp_path / "experiment.json"
    altered.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/generate_class_sets.py"),
         "--data", str(paths[0]), "--domains", str(paths[1]),
         "--protocol", str(altered), "--output", str(altered), "--check"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "Конфигурация отличается" in result.stderr
