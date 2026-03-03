# agent-bench

Benchmark Cursor `agent` models on AIME24 problems using lm-eval’s dataset source. Each question is run in a fresh `agent` invocation, and we record `usage.outputTokens` as the canonical **“tokens used including thinking.”** Input/cache tokens are ignored.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requirements:
- Cursor `agent` CLI installed and authenticated (`agent whoami` should work).
- `CURSOR_API_KEY` or logged-in Cursor session.

## Usage

List models:

```bash
agent-bench --list-models
```

Run AIME24 with 5 runs per question (default):

```bash
agent-bench --model gpt-5.3-codex
```

Smoke test (first 2 questions, 1 run each):

```bash
agent-bench --model gpt-5.3-codex --runs 1 --limit 2
```

Generate a report from a CSV:

```bash
agent-bench --report reports/aime24-gpt-5.3-codex-20260303-120000.csv
```

Compare multiple models:

```bash
agent-bench --report reports/aime24-modelA-*.csv reports/aime24-modelB-*.csv

Run multiple models in one command (single progress bar, one CSV per model):

```bash
agent-bench --models gpt-5.3-codex,sonnet-4-thinking --runs 5
```

Parallel runs per question (defaults to `--runs`, so all runs for a question execute concurrently):

```bash
agent-bench --model gpt-5.3-codex --runs 5 --concurrency 5
```

Resume a partial run:

```bash
agent-bench --resume reports/aime24-gpt-5.3-codex-20260303-120000.csv
```
```

## Output

Each run is recorded to a CSV in `reports/` with:
- per-run output tokens (`outputTokens`)
- raw model response
- parsed answer
- correctness vs. AIME24 answer
- full prompt and problem text

The report prints a Markdown table sorted by **ascending mean tokens** for the first CSV. Each cell is formatted:

```
<mean> +/- <std> <correctness>
```

Where correctness is a checkmark for 5/5 or a keycap digit (e.g., 4) otherwise.

Notes:
- Standard deviation uses population stdev (`pstdev`).
- AIME24 dataset is loaded from `Maxwell-Jia/AIME_2024`.

## Prompt policy

Each question is sent with a strict prompt:
- no tools / no code
- do math by hand
- output only the final integer answer (0-999)

Parsing uses the **last integer in [0, 999]** if extra text appears.
