from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from wcwidth import wcswidth

PLUS_MINUS = "\u00b1"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESET = "\x1b[0m"


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
    align: List[str]

    def to_markdown(self) -> str:
        widths = [display_width(h) for h in self.headers]
        for row in self.rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], display_width(cell))

        def fmt_row(values: List[str]) -> str:
            padded = []
            for i, cell in enumerate(values):
                padded.append(pad_display(cell, widths[i], self.align[i]))
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


def build_table(
    stats_by_model: List[Tuple[str, Dict[str, QuestionStats]]],
    *,
    use_color: bool,
    apm_by_model: List[Tuple[float, int, int]],
    tea_by_model: List[float],
) -> ReportTable:
    if not stats_by_model:
        raise ValueError("No CSV data provided")

    all_ids = sorted({qid for _, stats in stats_by_model for qid in stats.keys()})

    # Sort by mean tokens of the first model.
    primary_stats = stats_by_model[0][1]
    all_ids.sort(key=lambda qid: primary_stats.get(qid, QuestionStats(qid, 1e12, 0, 0, 0)).mean_tokens)

    headers = ["ID"] + [name for name, _ in stats_by_model]
    align = ["left"] + ["right"] * len(stats_by_model)
    rows: List[List[str]] = []

    mean_widths: List[int] = []
    std_widths: List[int] = []
    for _, stats in stats_by_model:
        if stats:
            mean_widths.append(
                max(len(str(int(round(item.mean_tokens, 0)))) for item in stats.values())
            )
            std_widths.append(
                max(len(str(int(round(item.std_tokens, 0)))) for item in stats.values())
            )
        else:
            mean_widths.append(1)
            std_widths.append(1)

    for qid in all_ids:
        row = [_format_aime_id(qid)]
        for idx, (_, stats) in enumerate(stats_by_model):
            cell = format_cell(stats.get(qid), mean_widths[idx], std_widths[idx], use_color=use_color)
            row.append(cell)
        rows.append(row)

    # Append summary row
    summary = ["APM"]
    for apm, _, tokens in apm_by_model:
        if tokens == 0:
            summary.append("0.0")
        else:
            summary.append(f"{apm:.1f}")
    rows.append(summary)

    tea_row = ["TEA"]
    for tea in tea_by_model:
        tea_row.append(f"{tea:.3f}")
    rows.append(tea_row)

    return ReportTable(headers=headers, rows=rows, align=align)


def format_cell(
    stats: QuestionStats | None,
    mean_width: int,
    std_width: int,
    *,
    use_color: bool,
) -> str:
    if stats is None:
        return "-"
    mean = int(round(stats.mean_tokens, 0))
    std = int(round(stats.std_tokens, 0))
    suffix = correct_suffix(stats.correct_count, stats.runs, use_color=use_color)
    return f"{str(mean).rjust(mean_width)} {PLUS_MINUS} {str(std).rjust(std_width)} {suffix}"


def correct_suffix(correct: int, runs: int, *, use_color: bool) -> str:
    label = f"@ {correct}/{runs}"
    if not use_color or runs <= 0:
        return label
    color = _color_for_ratio(correct, runs)
    return f"@ \x1b[38;5;{color}m{correct}/{runs}{RESET}"


def render_report(csv_paths: Iterable[Path], *, use_color: bool = True) -> str:
    stats_by_model: List[Tuple[str, Dict[str, QuestionStats]]] = []
    apm_by_model: List[Tuple[float, int, int]] = []
    tea_by_model: List[float] = []
    for path in csv_paths:
        stats_by_model.append(load_stats(path))
        apm_by_model.append(_compute_apm(path))
        tea_by_model.append(_compute_tea(path))
    table = build_table(
        stats_by_model,
        use_color=use_color,
        apm_by_model=apm_by_model,
        tea_by_model=tea_by_model,
    )
    return table.to_markdown()


def display_width(text: str) -> int:
    stripped = ANSI_RE.sub("", text)
    width = wcswidth(stripped)
    return width if width >= 0 else len(stripped)


def pad_display(text: str, width: int, align: str) -> str:
    current = display_width(text)
    if current >= width:
        return text
    padding = " " * (width - current)
    if align == "right":
        return padding + text
    return text + padding


def _compute_apm(csv_path: Path) -> Tuple[float, int, int]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        total_tokens = 0
        total_correct = 0
        for row in reader:
            try:
                total_tokens += int(row.get("output_tokens") or 0)
            except ValueError:
                pass
            if row.get("correct") == "1":
                total_correct += 1
    if total_tokens == 0:
        return 0.0, total_correct, total_tokens
    apm = total_correct / (total_tokens / 1_000_000.0)
    return apm, total_correct, total_tokens


def _compute_tea(csv_path: Path, *, budget: int = 32768) -> float:
    # Token-Efficiency AUC: 1 - mean(min(correct_tokens, B))/B over questions.
    per_q: Dict[str, List[Tuple[int, int]]] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qid = row.get("question_id")
            if not qid:
                continue
            try:
                tokens = int(row.get("output_tokens") or 0)
            except ValueError:
                tokens = 0
            correct = 1 if row.get("correct") == "1" else 0
            per_q.setdefault(qid, []).append((tokens, correct))

    if not per_q:
        return 0.0

    capped = []
    for runs in per_q.values():
        correct_tokens = [t for t, ok in runs if ok == 1]
        if correct_tokens:
            best = min(correct_tokens)
            capped.append(min(best, budget))
        else:
            capped.append(budget)

    mean_tokens = sum(capped) / len(capped)
    return max(0.0, min(1.0, 1.0 - (mean_tokens / budget)))


def _format_aime_id(qid: str) -> str:
    parts = qid.split("-")
    if len(parts) >= 3 and parts[0].isdigit():
        year = parts[0]
        part = parts[1]
        num = parts[2]
        if len(part) <= 2 and num.isdigit() and len(num) <= 2:
            return f"{year}-{part.rjust(2)}-{num.rjust(2)}"
    return qid


HSV_S = 0.45
HSV_V = 1.0


def _color_for_ratio(correct: int, runs: int) -> int:
    # HSV with fixed S/V and H varying from red (0) through yellow (60) to green (120).
    if runs <= 0:
        return 15
    ratio = max(0.0, min(1.0, correct / runs))
    hue = 120.0 * ratio  # 0=red, 60=yellow, 120=green
    r, g, b = _hsv_to_rgb(hue, HSV_S, HSV_V)
    return _rgb_to_ansi256(r, g, b)


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    h = h % 360.0
    c = v * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = v - c
    if 0 <= h < 60:
        rp, gp, bp = c, x, 0
    elif 60 <= h < 120:
        rp, gp, bp = x, c, 0
    elif 120 <= h < 180:
        rp, gp, bp = 0, c, x
    elif 180 <= h < 240:
        rp, gp, bp = 0, x, c
    elif 240 <= h < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x
    return rp + m, gp + m, bp + m


def _rgb_to_ansi256(r: float, g: float, b: float) -> int:
    # Map RGB [0,1] to ANSI 256-color cube (16-231).
    def to_6(v: float) -> int:
        return int(round(v * 5))

    r6, g6, b6 = to_6(r), to_6(g), to_6(b)
    r6, g6, b6 = max(0, min(5, r6)), max(0, min(5, g6)), max(0, min(5, b6))
    return 16 + 36 * r6 + 6 * g6 + b6
