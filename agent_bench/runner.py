from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentResult:
    output_tokens: int
    response_text: str
    parsed_answer: Optional[int]
    session_id: str
    request_id: str
    duration_ms: Optional[int]


_START_LOCK = threading.Lock()
_LAST_START = 0.0
_MIN_START_GAP = 5.0


def set_start_stagger(seconds: float) -> None:
    global _MIN_START_GAP
    _MIN_START_GAP = max(0.0, seconds)


def _throttle_start() -> None:
    global _LAST_START
    if _MIN_START_GAP <= 0:
        return
    with _START_LOCK:
        now = time.monotonic()
        wait = _MIN_START_GAP - (now - _LAST_START)
        if wait > 0:
            time.sleep(wait)
        _LAST_START = time.monotonic()


def build_prompt(problem_text: str, question_id: str) -> str:
    return (
        "You are taking AIME 2024. Solve the following problem.\n"
        "Rules (must follow):\n"
        "- Do NOT use any tools, external resources, or code.\n"
        "- Do all math by hand.\n"
        "\n"
        f"Problem ({question_id}):\n"
        f"{problem_text}\n"
        "Please reason step by step, and put your final answer within\\boxed\n"
    )


def run_agent(model: str, prompt: str) -> AgentResult:
    _throttle_start()
    cmd = [
        "agent",
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
        "--mode",
        "ask",
        "--sandbox",
        "enabled",
        prompt,
    ]
    env = os.environ.copy()
    # Keep locale stable to avoid shell warnings in some environments.
    env.setdefault("LC_ALL", "en_US.UTF-8")
    env.setdefault("LANG", "en_US.UTF-8")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    stdout = proc.stdout.strip()
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(f"agent failed (code {proc.returncode}): {stderr}\nstdout: {stdout}")

    json_line = _extract_json_line(stdout)
    payload = json.loads(json_line)
    usage = payload.get("usage") or {}
    output_tokens = int(usage.get("outputTokens", 0))
    response_text = payload.get("result", "")
    parsed_answer = parse_answer(response_text)
    return AgentResult(
        output_tokens=output_tokens,
        response_text=response_text,
        parsed_answer=parsed_answer,
        session_id=str(payload.get("session_id", "")),
        request_id=str(payload.get("request_id", "")),
        duration_ms=payload.get("duration_ms"),
    )


def parse_answer(text: str) -> Optional[int]:
    candidates = []
    for match in re.finditer(r"\d+", text):
        value = int(match.group(0))
        if 0 <= value <= 999:
            candidates.append(value)
    if not candidates:
        return None
    return candidates[-1]


def _extract_json_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            return line
    raise ValueError("Could not find JSON output from agent")
