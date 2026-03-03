from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import List

from agent_bench.dataset import load_dataset_by_name
from agent_bench.report import render_report
from agent_bench.runner import build_prompt, run_agent


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-bench")
    parser.add_argument("--model", action="append", help="Cursor model name (repeatable)")
    parser.add_argument("--models", help="Comma-separated list of models")
    parser.add_argument("--dataset", default="aime24", help="Dataset name (default: aime24)")
    parser.add_argument("--runs", type=int, default=5, help="Runs per question (default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N questions")
    parser.add_argument("--quiet", action="store_true", help="Suppress markdown output")
    parser.add_argument("--results-dir", default="results", help="Directory for CSV outputs")
    parser.add_argument("--list-models", action="store_true", help="List available models via agent")
    parser.add_argument("--report", nargs="+", help="Generate markdown report from CSV file(s)")

    args = parser.parse_args()

    if args.list_models:
        raise SystemExit(_list_models())

    if args.report:
        report_paths = [Path(p) for p in args.report]
        markdown = render_report(report_paths)
        print(markdown)
        return

    models: List[str] = []
    if args.model:
        models.extend(args.model)
    if args.models:
        models.extend([m.strip() for m in args.models.split(",") if m.strip()])

    if not models:
        raise SystemExit("--model is required unless --report or --list-models is used")

    dataset = list(load_dataset_by_name(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_paths: List[Path] = []

    total_runs = len(models) * len(dataset) * args.runs
    progress = tqdm(
        total=total_runs,
        desc=models[0],
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    for model in models:
        progress.set_description(model)
        model_runs = 0
        model_correct = 0
        csv_path = results_dir / f"{args.dataset}-{model}-{timestamp}.csv"
        csv_paths.append(csv_path)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestamp",
                    "model",
                    "dataset",
                    "question_index",
                    "question_id",
                    "run_index",
                    "output_tokens",
                    "response",
                    "parsed_answer",
                    "correct",
                    "answer",
                    "session_id",
                    "request_id",
                    "duration_ms",
                    "problem",
                    "prompt",
                ]
            )

            for problem in dataset:
                prompt = build_prompt(problem.problem, problem.question_id)
                for run_idx in range(1, args.runs + 1):
                    result = run_agent(model, prompt)
                    correct = int(result.parsed_answer == problem.answer)
                    model_runs += 1
                    model_correct += correct
                    acc = int(round((model_correct / model_runs) * 100, 0))
                    writer.writerow(
                        [
                            dt.datetime.now().isoformat(),
                            model,
                            args.dataset,
                            problem.index,
                            problem.question_id,
                            run_idx,
                            result.output_tokens,
                            result.response_text,
                            result.parsed_answer if result.parsed_answer is not None else "",
                            correct,
                            problem.answer,
                            result.session_id,
                            result.request_id,
                            result.duration_ms if result.duration_ms is not None else "",
                            problem.problem,
                            prompt,
                        ]
                    )
                    progress.update(1)
                    progress.set_postfix_str(f"acc {acc}%")

    progress.close()

    markdown = render_report(csv_paths)
    if not args.quiet:
        print(markdown)
    for path in csv_paths:
        print(f"Saved results to {path}")


def _list_models() -> int:
    import subprocess

    proc = subprocess.run(["agent", "--list-models"], text=True)
    return proc.returncode


if __name__ == "__main__":
    main()
