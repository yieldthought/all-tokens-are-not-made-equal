from __future__ import annotations

import argparse
import csv
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

from agent_bench.dataset import Problem, load_dataset_by_name
from agent_bench.report import render_report
from agent_bench.runner import build_prompt, run_agent, set_start_stagger

DEFAULT_HEADER = [
    "timestamp",
    "model",
    "dataset",
    "runs",
    "concurrency",
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


class _Postfix:
    def __init__(self) -> None:
        self.value = ""

    def __str__(self) -> str:
        return self.value


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-bench")
    parser.add_argument("--model", action="append", help="Cursor model name (repeatable)")
    parser.add_argument("--models", help="Comma-separated list of models")
    parser.add_argument("--dataset", default="aime24", help="Dataset name (default: aime24)")
    parser.add_argument("--runs", type=int, default=5, help="Runs per question (default: 5)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Parallel runs per question (default: --runs)",
    )
    parser.add_argument(
        "--resume",
        nargs="+",
        help="Resume a run from one or more CSV files",
    )
    parser.add_argument(
        "--agent-start-stagger",
        type=float,
        default=5.0,
        help="Minimum seconds between starting agent processes (default: 5)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N questions")
    parser.add_argument("--quiet", action="store_true", help="Suppress markdown output")
    parser.add_argument("--results-dir", default="reports", help="Directory for CSV outputs")
    parser.add_argument("--list-models", action="store_true", help="List available models via agent")
    parser.add_argument("--report", nargs="+", help="Generate markdown report from CSV file(s)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored correctness markers")

    args = parser.parse_args()

    if args.list_models:
        raise SystemExit(_list_models())

    if args.report:
        report_paths = [Path(p) for p in args.report]
        markdown = render_report(report_paths, use_color=not args.no_color)
        print(markdown)
        return

    set_start_stagger(args.agent_start_stagger)

    if args.resume:
        _resume_runs(args)
        return

    models: List[str] = []
    if args.model:
        models.extend(args.model)
    if args.models:
        models.extend([m.strip() for m in args.models.split(",") if m.strip()])

    if not models:
        raise SystemExit("--model is required unless --report, --resume, or --list-models is used")

    dataset = list(load_dataset_by_name(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    concurrency = args.concurrency or args.runs
    if concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_paths: List[Path] = []

    total_runs = len(models) * len(dataset) * args.runs
    progress, postfix = _build_progress(total_runs, models[0] if models else "agent-bench")

    try:
        for model in models:
            progress.set_description(model)
            model_runs = 0
            model_correct = 0
            csv_path = results_dir / f"{args.dataset}-{model}-{timestamp}.csv"
            csv_paths.append(csv_path)
            header = DEFAULT_HEADER
            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)

                for problem in dataset:
                    prompt = build_prompt(problem.problem, problem.question_id)
                    missing_runs = list(range(1, args.runs + 1))
                    model_runs, model_correct = _run_parallel(
                        writer=writer,
                        header=header,
                        model=model,
                        dataset_name=args.dataset,
                        problem=problem,
                        prompt=prompt,
                        missing_runs=missing_runs,
                        runs=args.runs,
                        concurrency=concurrency,
                        model_runs=model_runs,
                        model_correct=model_correct,
                        progress=progress,
                        postfix=postfix,
                    )
    except Exception:
        progress.close()
        _print_resume_hint(csv_paths)
        raise

    progress.close()

    markdown = render_report(csv_paths, use_color=not args.no_color)
    if not args.quiet:
        print(markdown)
    for path in csv_paths:
        print(f"Saved results to {path}")


def _build_progress(total_runs: int, desc: str):
    from tqdm import tqdm

    postfix = _Postfix()
    progress = tqdm(
        total=total_runs,
        desc=desc,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    progress.postfix = postfix
    return progress, postfix


def _list_models() -> int:
    import subprocess

    proc = subprocess.run(["agent", "--list-models"], text=True)
    return proc.returncode


def _run_parallel(
    *,
    writer: csv.writer,
    header: List[str],
    model: str,
    dataset_name: str,
    problem: Problem,
    prompt: str,
    missing_runs: List[int],
    runs: int,
    concurrency: int,
    model_runs: int,
    model_correct: int,
    progress,
    postfix: _Postfix,
) -> Tuple[int, int]:
    if not missing_runs:
        return model_runs, model_correct

    max_workers = min(concurrency, len(missing_runs))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_agent, model, prompt): run_idx for run_idx in missing_runs
        }
        for future in as_completed(futures):
            run_idx = futures[future]
            try:
                result = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
            correct = int(result.parsed_answer == problem.answer)
            model_runs += 1
            model_correct += correct
            acc = int(round((model_correct / model_runs) * 100, 0))
            progress.update(1)
            postfix.value = f"acc {acc}%"

            row = {
                "timestamp": dt.datetime.now().isoformat(),
                "model": model,
                "dataset": dataset_name,
                "runs": runs,
                "concurrency": concurrency,
                "question_index": problem.index,
                "question_id": problem.question_id,
                "run_index": run_idx,
                "output_tokens": result.output_tokens,
                "response": result.response_text,
                "parsed_answer": result.parsed_answer if result.parsed_answer is not None else "",
                "correct": correct,
                "answer": problem.answer,
                "session_id": result.session_id,
                "request_id": result.request_id,
                "duration_ms": result.duration_ms if result.duration_ms is not None else "",
                "problem": problem.problem,
                "prompt": prompt,
            }
            _write_row(writer, header, row)

    return model_runs, model_correct


def _write_row(writer: csv.writer, header: List[str], row: Dict[str, object]) -> None:
    writer.writerow([row.get(field, "") for field in header])


def _print_resume_hint(csv_paths: List[Path]) -> None:
    if not csv_paths:
        return
    paths = " ".join(str(path) for path in csv_paths)
    print(f"\nRun interrupted. Resume with:\n  agent-bench --resume {paths}\n")


def _resume_runs(args: argparse.Namespace) -> None:
    resume_paths = [Path(p) for p in args.resume]
    jobs = []
    total_missing = 0

    for path in resume_paths:
        if not path.exists():
            raise SystemExit(f"Resume file not found: {path}")
        (
            header,
            model,
            dataset_name,
            runs,
            concurrency,
            completed,
            model_runs,
            model_correct,
        ) = _load_state(path, args)
        dataset = list(load_dataset_by_name(dataset_name))
        if args.limit:
            dataset = dataset[: args.limit]
        missing_map: Dict[str, List[int]] = {}
        for problem in dataset:
            missing = [
                run_idx
                for run_idx in range(1, runs + 1)
                if (problem.question_id, run_idx) not in completed
            ]
            if missing:
                missing_map[problem.question_id] = missing
                total_missing += len(missing)
        jobs.append(
            (path, header, model, dataset_name, runs, concurrency, dataset, missing_map, model_runs, model_correct)
        )

    if total_missing == 0:
        print("No missing runs to resume.")
        return

    progress, postfix = _build_progress(total_missing, jobs[0][2] if jobs else "agent-bench")

    try:
        for path, header, model, dataset_name, runs, concurrency, dataset, missing_map, model_runs, model_correct in jobs:
            progress.set_description(model)
            with path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                for problem in dataset:
                    missing_runs = missing_map.get(problem.question_id, [])
                    if not missing_runs:
                        continue
                    prompt = build_prompt(problem.problem, problem.question_id)
                    model_runs, model_correct = _run_parallel(
                        writer=writer,
                        header=header,
                        model=model,
                        dataset_name=dataset_name,
                        problem=problem,
                        prompt=prompt,
                        missing_runs=missing_runs,
                        runs=runs,
                        concurrency=concurrency,
                        model_runs=model_runs,
                        model_correct=model_correct,
                        progress=progress,
                        postfix=postfix,
                    )
    except Exception:
        progress.close()
        _print_resume_hint(resume_paths)
        raise

    progress.close()
    markdown = render_report(resume_paths, use_color=not args.no_color)
    if not args.quiet:
        print(markdown)
    for path in resume_paths:
        print(f"Saved results to {path}")


def _load_state(
    path: Path, args: argparse.Namespace
) -> Tuple[
    List[str],
    str,
    str,
    int,
    int,
    Set[Tuple[str, int]],
    int,
    int,
]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames or DEFAULT_HEADER

    if not rows:
        raise SystemExit(f"Resume file is empty: {path}")

    first = rows[0]
    model = first.get("model") or path.stem
    dataset_name = first.get("dataset") or "aime24"
    runs = int(first.get("runs") or args.runs or 0)
    if runs == 0:
        runs = max(int(row.get("run_index") or 0) for row in rows)
        if runs == 0:
            raise SystemExit(f"Could not infer runs from {path}; pass --runs")
    concurrency = int(first.get("concurrency") or args.concurrency or runs)
    completed: Set[Tuple[str, int]] = set()
    model_runs = 0
    model_correct = 0
    for row in rows:
        qid = row.get("question_id")
        run_idx = row.get("run_index")
        if qid and run_idx:
            completed.add((qid, int(run_idx)))
        model_runs += 1
        if row.get("correct") == "1":
            model_correct += 1

    return header, model, dataset_name, runs, concurrency, completed, model_runs, model_correct


if __name__ == "__main__":
    main()
