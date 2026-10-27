"""Evaluation harness: runs agents over fixed seed sets and aggregates metrics.

Fairness model
--------------
Every agent plays the same seeds (``seed_base, seed_base+1, ...``), so all
agents face identical piece sequences. Environments are fully deterministic,
scores are directly comparable.

Invalid-move policy
-------------------
An illegal move or an unparseable LLM reply gets fed back to the agent; it
may retry up to ``invalid_action_retries`` times per move. If it still fails,
the episode ends (reason ``invalid_action``) and keeps its partial score —
making format-following ability visible in ``invalid_per_move`` and
``valid_completion_rate`` instead of silently discarding data.
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agents.base import Agent, AgentError, AgentParseError
from .env import BlockBlastEnv, InvalidActionError
from .render import board_rows_as_strings
from .scoring import ScoringConfig

REASON_BOARD_FULL = "board_full"
REASON_MOVE_CAP = "move_cap"
REASON_INVALID = "invalid_action"
REASON_AGENT_ERROR = "agent_error"


@dataclass
class EvalConfig:
    episodes: int = 10
    seed_base: int = 2024
    max_moves: int = 3000
    invalid_action_retries: int = 2
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    results_dir: Path = Path("results")
    save_transcripts: bool = True
    save_dialog: bool = False
    concurrency: int = 1
    tag: str = ""

    def seeds(self) -> List[int]:
        return [self.seed_base + i for i in range(self.episodes)]


@dataclass
class EpisodeResult:
    seed: int
    agent: str
    score: int
    moves: int
    lines_cleared: int
    max_combo: int
    perfect_clears: int
    invalid_actions: int
    reason: str
    duration_s: float
    agent_stats: Dict[str, Any] = field(default_factory=dict)
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    dialog: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""


@dataclass
class RunResult:
    run_id: str
    agent: str
    model: Optional[str]
    episodes: int
    seed_base: int
    seeds: List[int]
    scoring: Dict[str, Any]
    started_at: str
    results: List[EpisodeResult] = field(default_factory=list)
    resumed_from: Optional[str] = None
    """Run id this run continued from, when produced by ``resume_eval``.

    Makes a resume chain traceable: reports can collapse a chain to its
    final state instead of showing every intermediate snapshot as a
    separate (and misleadingly comparable) leaderboard row.
    """


def replay_transcript(env: BlockBlastEnv, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-apply recorded moves onto a fresh env (deterministic restore).

    Stops at the first terminal-error record; returns the clean move records
    that were replayed. The env ends in exactly the recorded state because
    piece draws depend only on the seed and the action sequence.
    """
    records: List[Dict[str, Any]] = []
    for rec in transcript:
        if rec.get("terminal_error"):
            break
        env.step(rec["piece"], rec["row"], rec["col"])
        records.append(dict(rec))
    return records


def run_episode(
    agent: Agent,
    seed: int,
    cfg: EvalConfig,
    on_move: Optional[Callable[[Dict[str, Any]], None]] = None,
    resume_from: Optional[List[Dict[str, Any]]] = None,
) -> EpisodeResult:
    """Play one full episode with retry-on-invalid semantics.

    ``resume_from`` continues an interrupted episode: its move records are
    replayed into the fresh env before the agent takes over. The restored
    env state is exact; an agent with internal tie-break randomness may
    still diverge from the hypothetical uninterrupted game (a stateless
    LLM agent at temperature 0 does not).
    """
    t0 = time.monotonic()
    env = BlockBlastEnv(seed=seed, scoring=cfg.scoring)
    agent.reset(seed)
    transcript: List[Dict[str, Any]] = []
    invalid_total = 0
    if resume_from:
        transcript = replay_transcript(env, resume_from)
        invalid_total = sum(len(r.get("invalid_attempts", [])) for r in transcript)
    obs = env.observation()
    reason = REASON_BOARD_FULL
    error = ""

    while True:
        if obs.is_game_over():
            reason = REASON_BOARD_FULL
            break
        if env.moves >= cfg.max_moves:
            reason = REASON_MOVE_CAP
            break

        attempts: List[str] = []
        step_res = None
        aborted = False
        for _attempt in range(cfg.invalid_action_retries + 1):
            feedback = attempts[-1] if attempts else None
            try:
                action = agent.act(obs, feedback=feedback)
                step_res = env.step(action.piece, action.row, action.col)
                break
            except (InvalidActionError, AgentParseError) as exc:
                attempts.append(str(exc))
                invalid_total += 1
            except AgentError as exc:
                reason, error, aborted = REASON_AGENT_ERROR, str(exc), True
                break
            except Exception as exc:  # defensive: a bug must not kill the whole run
                reason, error, aborted = (
                    REASON_AGENT_ERROR,
                    f"unexpected {type(exc).__name__}: {exc}",
                    True,
                )
                break
        if aborted or step_res is None:
            if not aborted:
                reason = REASON_INVALID
                error = attempts[-1] if attempts else "no attempt"
            transcript.append({"move": env.moves + 1, "terminal_error": error, "reason": reason})
            break

        record = {
            "move": step_res.observation.moves,
            "piece": step_res.action[0],
            "piece_name": step_res.piece_name,
            "row": step_res.action[1],
            "col": step_res.action[2],
            "cells": step_res.cells,
            "lines": step_res.lines,
            "cleared_rows": list(step_res.cleared_rows),
            "cleared_cols": list(step_res.cleared_cols),
            "combo": step_res.combo,
            "perfect_clear": step_res.perfect_clear,
            "points": step_res.points,
            "score": step_res.observation.score,
            "board_after": list(board_rows_as_strings(step_res.observation.board)),
        }
        if attempts:
            record["invalid_attempts"] = attempts
        transcript.append(record)
        if on_move is not None:
            on_move(record)
        obs = step_res.observation

    return EpisodeResult(
        seed=seed,
        agent=agent.name,
        score=env.score,
        moves=env.moves,
        lines_cleared=env.lines_cleared,
        max_combo=env.max_combo,
        perfect_clears=env.perfect_clears,
        invalid_actions=invalid_total,
        reason=reason,
        duration_s=round(time.monotonic() - t0, 3),
        agent_stats=agent.stats,
        transcript=transcript,
        dialog=list(getattr(agent, "dialog", []) or []),
        error=error,
    )


def run_eval(
    agent_factory: Callable[[], Agent],
    cfg: EvalConfig,
    agent_name: str,
    model: Optional[str] = None,
) -> RunResult:
    """Run ``cfg.episodes`` episodes (one fresh agent instance each)."""
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_id = _make_run_id(agent_name, model, cfg.tag)
    seeds = cfg.seeds()

    def _one(seed: int) -> EpisodeResult:
        return run_episode(agent_factory(), seed, cfg)

    if cfg.concurrency > 1:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            results = list(pool.map(_one, seeds))
    else:
        results = [_one(seed) for seed in seeds]

    return RunResult(
        run_id=run_id,
        agent=agent_name,
        model=model,
        episodes=len(results),
        seed_base=cfg.seed_base,
        seeds=seeds,
        scoring=cfg.scoring.to_dict(),
        started_at=started_at,
        results=results,
    )


def _make_run_id(agent_name: str, model: Optional[str], tag: str) -> str:
    import re
    import uuid

    parts = [time.strftime("%Y%m%d-%H%M%S"), agent_name]
    if model:
        parts.append(model)
    if tag:
        parts.append(tag)
    slug = "-".join(re.sub(r"[^A-Za-z0-9._-]+", "_", p) for p in parts)
    return f"{slug}-{uuid.uuid4().hex[:4]}"


#: episode end reasons that leave the game genuinely finished
FINISHED_REASONS = (REASON_BOARD_FULL, REASON_MOVE_CAP)


def _episode_from_payload(payload: Dict[str, Any], agent_stats: Dict[str, Any]) -> EpisodeResult:
    return EpisodeResult(
        seed=payload["seed"],
        agent=payload.get("agent", "?"),
        score=payload["score"],
        moves=payload["moves"],
        lines_cleared=payload.get("lines_cleared", 0),
        max_combo=payload.get("max_combo", 0),
        perfect_clears=payload.get("perfect_clears", 0),
        invalid_actions=payload.get("invalid_actions", 0),
        reason=payload.get("reason", ""),
        duration_s=payload.get("duration_s", 0.0),
        agent_stats=agent_stats,
        transcript=payload.get("transcript", []),
        dialog=payload.get("dialog", []),
        error=payload.get("error", ""),
    )


def _merge_stats(prior: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(prior)
    for k, v in new.items():
        if isinstance(v, (int, float)) and isinstance(merged.get(k), (int, float)):
            merged[k] = round(merged[k] + v, 3) if isinstance(v, float) else merged[k] + v
        else:
            merged[k] = v
    return merged


def resume_eval(
    agent_factory: Callable[[], Agent],
    prior_run_dir: Path,
    cfg: EvalConfig,
    agent_name: str,
    model: Optional[str] = None,
) -> RunResult:
    """Finish an interrupted run: keep finished episodes, resume the rest.

    Episodes that ended with ``agent_error`` (or other non-final reasons)
    are replayed from their saved transcript onto a fresh deterministic env
    and continued from the exact interruption point.
    """
    import json

    prior_run_dir = Path(prior_run_dir)
    summary = json.loads((prior_run_dir / "summary.json").read_text(encoding="utf-8"))
    prior_stats = {
        row["seed"]: row.get("agent_stats", {}) for row in summary.get("episodes_detail", [])
    }
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_id = _make_run_id(agent_name, model, cfg.tag or "resume")
    seeds = summary["seeds"]

    def _one(seed: int) -> EpisodeResult:
        ep_path = prior_run_dir / "episodes" / f"seed_{seed}.json"
        if not ep_path.exists():  # never started -> plain fresh episode
            return run_episode(agent_factory(), seed, cfg)
        payload = json.loads(ep_path.read_text(encoding="utf-8"))
        stats = prior_stats.get(seed, {})
        if payload.get("reason") in FINISHED_REASONS:
            return _episode_from_payload(payload, stats)
        prior_transcript = payload.get("transcript", [])
        result = run_episode(
            agent_factory(), seed, cfg, resume_from=prior_transcript
        )
        return EpisodeResult(
            seed=result.seed,
            agent=result.agent,
            score=result.score,
            moves=result.moves,
            lines_cleared=result.lines_cleared,
            max_combo=result.max_combo,
            perfect_clears=result.perfect_clears,
            invalid_actions=result.invalid_actions,
            reason=result.reason,
            duration_s=round(payload.get("duration_s", 0.0) + result.duration_s, 3),
            agent_stats=_merge_stats(stats, result.agent_stats),
            transcript=result.transcript,
            dialog=(payload.get("dialog", []) or []) + result.dialog,
            error=result.error,
        )

    if cfg.concurrency > 1:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            results = list(pool.map(_one, seeds))
    else:
        results = [_one(seed) for seed in seeds]

    return RunResult(
        run_id=run_id,
        agent=agent_name,
        model=model,
        episodes=len(results),
        seed_base=summary.get("seed_base", cfg.seed_base),
        seeds=seeds,
        scoring=cfg.scoring.to_dict(),
        started_at=started_at,
        results=results,
        resumed_from=summary.get("run_id") or prior_run_dir.name,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregates(run: RunResult) -> Dict[str, Any]:
    eps = run.results
    n = len(eps)
    scores = [e.score for e in eps]
    moves = [e.moves for e in eps]
    lines = [e.lines_cleared for e in eps]
    tokens = [int(e.agent_stats.get("total_tokens", 0) or 0) for e in eps]
    reasons = {}
    for e in eps:
        reasons[e.reason] = reasons.get(e.reason, 0) + 1
    total_moves = sum(moves) or 1
    return {
        "episodes": n,
        "mean_score": round(statistics.mean(scores), 1) if scores else 0.0,
        "median_score": round(statistics.median(scores), 1) if scores else 0,
        "max_score": max(scores, default=0),
        "min_score": min(scores, default=0),
        "stdev_score": round(statistics.pstdev(scores), 1) if n > 1 else 0.0,
        "mean_moves": round(statistics.mean(moves), 1) if moves else 0.0,
        "total_moves": sum(moves),
        "mean_lines": round(statistics.mean(lines), 1) if lines else 0.0,
        "total_lines": sum(lines),
        "best_combo": max((e.max_combo for e in eps), default=0),
        "perfect_clears": sum(e.perfect_clears for e in eps),
        "invalid_total": sum(e.invalid_actions for e in eps),
        "invalid_per_move": round(sum(e.invalid_actions for e in eps) / total_moves, 4),
        "valid_completion_rate": round(
            sum(1 for e in eps if e.reason not in (REASON_INVALID, REASON_AGENT_ERROR)) / n, 4
        ) if n else 0.0,
        "tokens_per_move": round(sum(tokens) / total_moves, 1),
        "reasons": reasons,
        "mean_duration_s": round(statistics.mean([e.duration_s for e in eps]), 2) if eps else 0.0,
    }
