from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PLUS_MINUS = "\u00b1"
CHECK = "\u2705"
KEYCAP_SUFFIX = "\uFE0F\u20E3"


@dataclass
class QuestionStats:
    question_id: str
    mean_tokens: float
    std_tokens: float
    correct_count: int
    runs: int


@dataclass
class ReportTable:
    headers: List[str]
    rows: List[List[str]]

    def to_markdown(self) -> str:
        widths = [len(h) for h in self.headers]
        for row in self.rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def fmt_row(values: List[str]) -> str:
            padded = [values[i].ljust(widths[i]) for i in range(len(values))]
            return "| " + " | ".join(padded) + " |"

        header = fmt_row(self.headers)
        sep = "| " + " | ".join("-" * w for w in widths) + " |"
        body = "\n".join(fmt_row(row) for row in self.rows)
        return "\n".join([header, sep, body])


def load_stats(csv_path: Path) -> Tuple[str, Dict[str, QuestionStats]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    model_name = rows[0].get("model", csv_path.stem)
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        qid = row["question_id"]
        grouped.setdefault(qid, []).append(row)

    stats: Dict[str, QuestionStats] = {}
    for qid, items in grouped.items():
        tokens = [int(item["output_tokens"]) for item in items]
        correct = sum(1 for item in items if item.get("correct") == "1")
        mean_tokens = statistics.mean(tokens)
        std_tokens = statistics.pstdev(tokens) if len(tokens) > 1 else 0.0
        stats[qid] = QuestionStats(
            question_id=qid,
            mean_tokens=mean_tokens,
            std_tokens=std_tokens,
            correct_count=correct,
            runs=len(tokens),
        )
    return model_name, stats


def build_table(stats_by_model: List[Tuple[str, Dict[str, QuestionStats]]]) -> ReportTable:
    if not stats_by_model:
        raise ValueError("No CSV data provided")

    all_ids = sorted({qid for _, stats in stats_by_model for qid in stats.keys()})

    # Sort by mean tokens of the first model.
    primary_stats = stats_by_model[0][1]
    all_ids.sort(key=lambda qid: primary_stats.get(qid, QuestionStats(qid, 1e12, 0, 0, 0)).mean_tokens)

    headers = ["ID"] + [name for name, _ in stats_by_model]
    rows: List[List[str]] = []

    for qid in all_ids:
        row = [qid]
        for _, stats in stats_by_model:
            cell = format_cell(stats.get(qid))
            row.append(cell)
        rows.append(row)

    return ReportTable(headers=headers, rows=rows)


def format_cell(stats: QuestionStats | None) -> str:
    if stats is None:
        return "-"
    mean = int(round(stats.mean_tokens, 0))
    std = int(round(stats.std_tokens, 0))
    suffix = correct_suffix(stats.correct_count, stats.runs)
    return f"{mean} {PLUS_MINUS} {std} {suffix}"


def correct_suffix(correct: int, runs: int) -> str:
    if runs == 5 and correct == 5:
        return CHECK
    if 0 <= correct <= 9:
        return f"{correct}{KEYCAP_SUFFIX}"
    return str(correct)


def render_report(csv_paths: Iterable[Path]) -> str:
    stats_by_model: List[Tuple[str, Dict[str, QuestionStats]]] = []
    for path in csv_paths:
        stats_by_model.append(load_stats(path))
    table = build_table(stats_by_model)
    return table.to_markdown()
