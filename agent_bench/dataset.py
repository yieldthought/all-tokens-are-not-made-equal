from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from datasets import load_dataset


@dataclass(frozen=True)
class Problem:
    index: int
    question_id: str
    problem: str
    answer: int


def load_aime24() -> List[Problem]:
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    problems: List[Problem] = []
    for i, row in enumerate(dataset):
        question_id = str(row.get("ID", str(i + 1)))
        answer = int(row["Answer"])
        problems.append(
            Problem(
                index=i + 1,
                question_id=question_id,
                problem=row["Problem"],
                answer=answer,
            )
        )
    return problems


def load_dataset_by_name(name: str) -> Iterable[Problem]:
    if name.lower() in {"aime24", "aime_2024", "aime-2024"}:
        return load_aime24()
    raise ValueError(f"Unsupported dataset: {name}")
