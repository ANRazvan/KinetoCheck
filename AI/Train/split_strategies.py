from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


@dataclass(frozen=True)
class SplitPlan:
    split_mode: str
    fold_index: int
    fold_name: str
    held_out_subject: int | None
    train_records: list[dict]
    val_records: list[dict]
    test_records: list[dict]


def _shuffle_subjects(subject_ids: Iterable[int], seed: int) -> list[int]:
    subjects = sorted({int(subject_id) for subject_id in subject_ids})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    return subjects


def _split_subject_ids(subject_ids: list[int], ratios: tuple[float, ...], seed: int) -> list[list[int]]:
    if not subject_ids:
        return [[] for _ in ratios]

    total = len(subject_ids)
    shuffled = subject_ids[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    positive_ratios = [max(0.0, float(ratio)) for ratio in ratios]
    ratio_sum = sum(positive_ratios)
    if ratio_sum <= 0.0:
        raise ValueError("At least one split ratio must be positive.")

    active_indices = [index for index, ratio in enumerate(positive_ratios) if ratio > 0.0]
    normalized = [ratio / ratio_sum for ratio in positive_ratios]
    ideals = [normalized[index] * total for index in range(len(ratios))]

    counts = [int(math.floor(value)) if positive_ratios[index] > 0.0 else 0 for index, value in enumerate(ideals)]
    min_counts = [1 if positive_ratios[index] > 0.0 and total >= len(active_indices) else 0 for index in range(len(ratios))]

    for index in range(len(counts)):
        if counts[index] < min_counts[index]:
            counts[index] = min_counts[index]

    def current_total() -> int:
        return sum(counts)

    def fractional_part(index: int) -> float:
        return ideals[index] - math.floor(ideals[index])

    while current_total() > total:
        removable = [index for index in range(len(counts)) if counts[index] > min_counts[index]]
        if not removable:
            break
        removable.sort(key=lambda index: (fractional_part(index), counts[index]))
        counts[removable[0]] -= 1

    while current_total() < total:
        candidates = active_indices if active_indices else list(range(len(counts)))
        candidates.sort(key=lambda index: (fractional_part(index), positive_ratios[index]), reverse=True)
        counts[candidates[0]] += 1

    partitions: list[list[int]] = []
    offset = 0
    for count in counts:
        partitions.append(shuffled[offset : offset + count])
        offset += count

    return partitions


def _records_for_subjects(records: list[dict], subject_ids: Iterable[int]) -> list[dict]:
    subject_set = {int(subject_id) for subject_id in subject_ids}
    return [record for record in records if int(record["subject_id"]) in subject_set]


def build_subject_wise_split(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitPlan:
    subjects = _shuffle_subjects((record["subject_id"] for record in records), seed)
    train_ids, val_ids, test_ids = _split_subject_ids(subjects, (train_ratio, val_ratio, test_ratio), seed)
    return SplitPlan(
        split_mode="subject",
        fold_index=0,
        fold_name="subject_split",
        held_out_subject=None,
        train_records=_records_for_subjects(records, train_ids),
        val_records=_records_for_subjects(records, val_ids),
        test_records=_records_for_subjects(records, test_ids),
    )


def build_loso_splits(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> list[SplitPlan]:
    subjects = _shuffle_subjects((record["subject_id"] for record in records), seed)
    plans: list[SplitPlan] = []

    for fold_index, held_out_subject in enumerate(subjects):
        remaining_records = [record for record in records if int(record["subject_id"]) != held_out_subject]
        remaining_subjects = _shuffle_subjects((record["subject_id"] for record in remaining_records), seed + held_out_subject)
        train_ids, val_ids = _split_subject_ids(remaining_subjects, (train_ratio, val_ratio), seed + held_out_subject)
        plans.append(
            SplitPlan(
                split_mode="loso",
                fold_index=fold_index,
                fold_name=f"loso_subject_{held_out_subject:02d}",
                held_out_subject=held_out_subject,
                train_records=_records_for_subjects(remaining_records, train_ids),
                val_records=_records_for_subjects(remaining_records, val_ids),
                test_records=[record for record in records if int(record["subject_id"]) == held_out_subject],
            )
        )

    return plans


def build_split_plans(
    records: list[dict],
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[SplitPlan]:
    mode = split_mode.strip().lower()
    if mode in {"subject", "subject-wise", "subjectwise"}:
        return [build_subject_wise_split(records, train_ratio, val_ratio, test_ratio, seed)]
    if mode == "loso":
        return build_loso_splits(records, train_ratio, val_ratio, seed)
    raise ValueError(f"Unsupported split mode: {split_mode!r}")
