"""Tests for the evaluation harness, persistence and leaderboard."""

import json

from blockblast.agents import GreedyAgent, HeuristicAgent, RandomAgent
from blockblast.harness import (
    REASON_BOARD_FULL,
    EvalConfig,
    aggregates,
    run_episode,
    run_eval,
)
from blockblast.report import leaderboard_md, load_runs, refresh_leaderboard, write_run


def _cfg(tmp_path, **kw):
    return EvalConfig(results_dir=tmp_path / "results", **kw)


def test_run_episode_transcript_and_stats():
    result = run_episode(RandomAgent(), seed=5, cfg=EvalConfig())
    assert result.agent == "random"
    assert result.moves == len(result.transcript)
    assert result.reason == REASON_BOARD_FULL
    for i, rec in enumerate(result.transcript, 1):
        assert rec["move"] == i
        assert len(rec["board_after"]) == 8
    last = result.transcript[-1]
    assert last["score"] == result.score


def test_run_eval_fixed_seeds_and_aggregates(tmp_path):
    cfg = _cfg(tmp_path, episodes=4, seed_base=777)
    run = run_eval(lambda: HeuristicAgent(), cfg, agent_name="heuristic")
    assert run.seeds == [777, 778, 779, 780]
    assert len(run.results) == 4
    agg = aggregates(run)
    assert agg["episodes"] == 4
    assert agg["mean_score"] > 0
    assert set(agg["reasons"]) <= {"board_full", "move_cap"}
    assert 0.0 <= agg["valid_completion_rate"] <= 1.0


def test_same_seeds_same_games_for_two_agents():
    """Both agents face identical piece sequences (same seeds => same draws)."""
    a = run_eval(lambda: RandomAgent(), EvalConfig(episodes=3, seed_base=42), "random")
    b = run_eval(lambda: RandomAgent(), EvalConfig(episodes=3, seed_base=42), "random")
    # same agent + same seeds => identical score progression
    assert [e.score for e in a.results] == [e.score for e in b.results]


def test_write_run_persists_everything(tmp_path):
    cfg = _cfg(tmp_path, episodes=2)
    run = run_eval(lambda: GreedyAgent(), cfg, agent_name="greedy", model=None)
    run_dir = write_run(run, cfg.results_dir)

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["run_id"] == run.run_id
    assert summary["agent"] == "greedy"
    assert len(summary["episodes_detail"]) == 2
    assert summary["aggregates"]["mean_score"] >= 0

    for ep in run.results:
        path = run_dir / "episodes" / f"seed_{ep.seed}.json"
        data = json.loads(path.read_text())
        assert data["score"] == ep.score
        assert len(data["transcript"]) == ep.moves

    index = load_runs(cfg.results_dir)
    assert len(index) == 1
    assert index[0]["run_id"] == run.run_id

    board_path = cfg.results_dir / "leaderboard.md"
    assert board_path.exists()
    text = board_path.read_text()
    assert "| 1 | greedy |" in text


def test_leaderboard_sorts_by_mean_score(tmp_path):
    cfg = _cfg(tmp_path)
    for agent in (RandomAgent, GreedyAgent, HeuristicAgent):
        run = run_eval(agent, cfg, agent_name=agent.name)
        write_run(run, cfg.results_dir)
    rows = load_runs(cfg.results_dir)
    assert len(rows) == 3
    expected_order = [
        r["agent"] for r in sorted(rows, key=lambda r: r["aggregates"]["mean_score"], reverse=True)
    ]
    md = leaderboard_md(rows)
    table_rows = [
        line for line in md.splitlines() if line.startswith("| ") and not line.startswith("| #")
    ]
    actual_order = [line.split("|")[2].strip() for line in table_rows]
    assert actual_order == expected_order
    # random must be the weakest baseline by a clear margin
    assert actual_order[-1] == "random"


def test_refresh_leaderboard_without_runs(tmp_path):
    assert refresh_leaderboard(tmp_path) is None


def test_move_cap_reason(tmp_path):
    cfg = _cfg(tmp_path, max_moves=2)
    result = run_episode(RandomAgent(), seed=9, cfg=EvalConfig(max_moves=2))
    assert result.reason == "move_cap"
    assert result.moves == 2


def test_concurrency_matches_sequential(tmp_path):
    """Concurrent episode execution must not change results."""
    seeds = EvalConfig(episodes=5, seed_base=31337)

    def factory():
        return HeuristicAgent()

    seq = run_eval(factory, seeds, "heuristic")
    par = run_eval(factory, EvalConfig(episodes=5, seed_base=31337, concurrency=5), "heuristic")
    assert [e.score for e in seq.results] == [e.score for e in par.results]
