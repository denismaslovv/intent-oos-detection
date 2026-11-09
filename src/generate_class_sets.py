"""Детерминированный выбор K=10/20/30 намерений CLINC150 по train-аудиту.

В выборе классов не участвуют validation, test, предсказания и метрики модели.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SEEDS = (17, 42, 123)
K_VALUES = (10, 20, 30)
EXPECTED_SHA256 = {
    "data_full.json": "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    "domains.json": "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_config(data_path: Path, domains_path: Path, protocol_path: Path) -> dict:
    """Воссоздать K-наборы из зафиксированных допустимых пар, без CSV."""
    data_path, domains_path, protocol_path = map(Path, (data_path, domains_path, protocol_path))
    for source in (data_path, domains_path):
        if sha256(source) != EXPECTED_SHA256[source.name]:
            raise ValueError(f"Контрольная сумма {source.name} отличается от ожидаемой CLINC150")

    domains = json.loads(domains_path.read_text(encoding="utf-8"))
    data = json.loads(data_path.read_text(encoding="utf-8"))
    config = json.loads(protocol_path.read_text(encoding="utf-8"))
    domain_names = sorted(domains)
    if len(domain_names) != 10 or any(len(intents) != 15 for intents in domains.values()):
        raise ValueError("Ожидалось 10 доменов по 15 намерений")
    all_intents = {intent for members in domains.values() for intent in members}
    if len(all_intents) != 150:
        raise ValueError("В domains.json дублируются или отсутствуют намерения")
    counts = Counter(intent for _, intent in data["train"])
    if set(counts) != all_intents or set(counts.values()) != {100}:
        raise ValueError("Ожидались 150 классов по 100 train-примеров")

    heldout = config["near_oos_by_domain"]
    pools = config["eligible_known_by_domain"]
    if set(heldout) != set(domain_names) or set(pools) != set(domain_names):
        raise ValueError("Не хватает доменов для Held-out или близких Known")
    if len(set(heldout.values())) != 10:
        raise ValueError("Требуется 10 разных Held-out намерений")
    if config["near_oos_intents"] != [heldout[d] for d in domain_names]:
        raise ValueError("near_oos_intents не согласован с near_oos_by_domain")
    for domain in domain_names:
        candidates = pools[domain]
        if heldout[domain] not in domains[domain]:
            raise ValueError(f"Некорректный Held-out: {domain}")
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError(f"Пустой пул или повторные кандидаты: {domain}")
        if any(c not in domains[domain] or c == heldout[domain] for c in candidates):
            raise ValueError(f"Недопустимый Known в пуле {domain}")

    anchors_by_domain = {}
    for domain in domain_names:
        options = sorted(pools[domain])
        random.Random(f"near-oos-v1:{domain}:{','.join(map(str, SEEDS))}").shuffle(options)
        anchors_by_domain[domain] = options

    experiments = []
    near = set(heldout.values())
    for repeat_index, seed in enumerate(SEEDS):
        rng = random.Random(seed)
        domain_orders = {}
        anchors = {}
        for domain in domain_names:
            anchor = anchors_by_domain[domain][repeat_index % len(anchors_by_domain[domain])]
            anchors[domain] = anchor
            others = sorted(set(domains[domain]) - near - {anchor})
            rng.shuffle(others)
            domain_orders[domain] = [anchor, *others]
        selected = {
            str(k): [domain_orders[domain][i] for domain in domain_names for i in range(k // 10)]
            for k in K_VALUES
        }
        experiments.append({"seed": seed, "anchors_by_domain": anchors, "known_intents": selected})
    config["experiments"] = experiments
    return config


def find_data_file(name: str) -> Path:
    for path in (Path("data") / name, Path(name)):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Нет {name}; положите его в data/ или укажите --data/--domains")


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация вложенных near-OOS K-наборов")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--domains", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--check", action="store_true", help="Сравнить с конфигурацией без записи")
    args = parser.parse_args()
    config = generate_config(
        args.data or find_data_file("data_full.json"),
        args.domains or find_data_file("domains.json"),
        args.protocol,
    )
    if args.check:
        if json.loads(args.output.read_text(encoding="utf-8")) != config:
            parser.error("Конфигурация отличается от воспроизводимо сформированной")
        print(f"OK: {args.output} полностью соответствует исходным данным, утверждённым парам и seed")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Создан {args.output}: {len(config['experiments'])} повторения, K=10/20/30")
        for run in config["experiments"]:
            print(f"  seed {run['seed']}: " + ", ".join(f"K={k}: {len(run['known_intents'][str(k)])}" for k in K_VALUES))


if __name__ == "__main__":
    main()
