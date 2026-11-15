"""Prepare merged, shuffled, and stratified train/val JSONL datasets."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from validate_dataset import validate_record

DATA_BATCHES_DIR = Path("data_batches")
MERGED_OUT = Path("dataset_exam_generator.jsonl")
TRAIN_OUT = Path("dataset_train.jsonl")
VAL_OUT = Path("dataset_val.jsonl")
SEED = 42


def _extract_meta(user_prompt: str) -> dict[str, Any]:
    level_match = re.search(
        r"(?:Education Level|Jenjang Pendidikan):\s*(.+)", user_prompt
    )
    lang_match = re.search(r"(?:Language|Bahasa):\s*(.+)", user_prompt)
    level = level_match.group(1).strip() if level_match else "unknown"
    lang_raw = lang_match.group(1).strip().lower() if lang_match else "unknown"
    lang = "en" if "english" in lang_raw else ("id" if "indonesi" in lang_raw else lang_raw)
    has_ref = (
        "Reference Material:" in user_prompt or "Materi Referensi:" in user_prompt
    )
    return {"level": level, "lang": lang, "has_ref": has_ref}


def _load_and_validate_shards() -> tuple[list[dict[str, Any]], dict[str, int]]:
    shard_paths = sorted(DATA_BATCHES_DIR.glob("*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No .jsonl shards found in {DATA_BATCHES_DIR}")

    records: list[dict[str, Any]] = []
    shard_counts: dict[str, int] = {}
    seen_prompts: set[str] = set()
    all_errors: list[str] = []
    all_warnings: list[str] = []

    for shard_path in shard_paths:
        count = 0
        with shard_path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                errors, warnings, _ = validate_record(record, idx=line_no)
                if errors:
                    all_errors.extend(f"{shard_path.name}:{line_no}: {e}" for e in errors)
                if warnings:
                    all_warnings.extend(f"{shard_path.name}:{line_no}: {w}" for w in warnings)

                user_prompt = record["messages"][1]["content"]
                if user_prompt in seen_prompts:
                    all_errors.append(
                        f"{shard_path.name}:{line_no}: Duplicate user prompt across shards"
                    )
                seen_prompts.add(user_prompt)

                records.append({"shard": shard_path.name, "record": record})
                count += 1
        shard_counts[shard_path.name] = count

    if all_errors:
        raise ValueError("Validation errors found:\n" + "\n".join(all_errors))
    if all_warnings:
        print("Validation warnings:\n" + "\n".join(all_warnings))

    return records, shard_counts


def _stratified_split(
    items: list[dict[str, Any]], seed: int = SEED
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_shard: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_shard.setdefault(item["shard"], []).append(item)

    # Pick 10 validation records (~10% of 96): 1 per shard (8) + 1 extra from 2 shards.
    shards_sorted = sorted(by_shard.keys())
    extra_shards = set(rng.sample(shards_sorted, k=min(2, len(shards_sorted))))

    val_items: list[dict[str, Any]] = []
    train_items: list[dict[str, Any]] = []

    for shard in shards_sorted:
        group = list(by_shard[shard])
        rng.shuffle(group)
        k = 2 if shard in extra_shards else 1
        # Keep pure shuffled order per shard so val represents the corpus distribution
        # (~2-3 with_ref, ~2-3 EN, all shards/levels covered).
        val_items.extend(group[:k])
        train_items.extend(group[k:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _print_stats(name: str, items: list[dict[str, Any]]) -> None:
    levels = Counter()
    langs = Counter()
    refs = 0
    chars: list[int] = []

    for item in items:
        msgs = item["record"]["messages"]
        meta = _extract_meta(msgs[1]["content"])
        levels[meta["level"]] += 1
        langs[meta["lang"]] += 1
        if meta["has_ref"]:
            refs += 1
        total_chars = sum(len(m["content"]) for m in msgs)
        chars.append(total_chars)

    avg_chars = sum(chars) / len(chars) if chars else 0.0
    avg_tokens = avg_chars / 4.0
    print(
        f"[{name}] count={len(items)} | levels={dict(levels)} | "
        f"langs={dict(langs)} | with_ref={refs} | "
        f"chars(min/avg/max)={min(chars)}/{avg_chars:.1f}/{max(chars)} | "
        f"approx_tokens_avg={avg_tokens:.1f}"
    )


def main() -> int:
    items, shard_counts = _load_and_validate_shards()
    print("Loaded shards:")
    for shard, cnt in shard_counts.items():
        print(f"  - {shard}: {cnt}")

    rng = random.Random(SEED)
    merged_items = list(items)
    rng.shuffle(merged_items)

    train_items, val_items = _stratified_split(items, seed=SEED)

    _write_jsonl(MERGED_OUT, [x["record"] for x in merged_items])
    _write_jsonl(TRAIN_OUT, [x["record"] for x in train_items])
    _write_jsonl(VAL_OUT, [x["record"] for x in val_items])

    print("\nDataset summary:")
    _print_stats(MERGED_OUT.name, merged_items)
    _print_stats(TRAIN_OUT.name, train_items)
    _print_stats(VAL_OUT.name, val_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
