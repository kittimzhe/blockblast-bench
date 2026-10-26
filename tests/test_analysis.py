"""Tests for the decision-quality analysis module."""

import json

from blockblast.agents import GreedyAgent, RandomAgent
from blockblast.analysis import analyze_episode, analyze_run, classify_invalid, render_analysis
from blockblast.harness import EvalConfig, run_episode, run_eval
from blockblast.report import write_run


def _payload(result):
    return {
        "seed": result.seed,
        "score": result.score,
        "moves": result.moves,
        "reason": result.reason,
        "transcript": result.transcript,
    }


def test_greedy_efficiency_is_one_for_greedy_agent():
    """The greedy agent IS the single-step optimum: efficiency must be 1.0."""
    result = run_episode(GreedyAgent(), seed=2024, cfg=EvalConfig())
    ana = analyze_episode(_payload(result))
    assert ana.greedy_efficiency == 1.0
    assert ana.missed_clears == 0


def test_random_agent_efficiency_below_one():
    result = run_episode(RandomAgent(), seed=2024, cfg=EvalConfig())
    ana = analyze_episode(_payload(result))
    assert 0.0 < ana.greedy_efficiency < 1.0
    assert ana.final_fill > 0.3  # died with a fairly full board
    assert ana.final_regions >= 1


def test_classify_invalid():
    assert classify_invalid("cannot place L5-B at (row=0, col=5): out of bounds or overlapping occupied cells") == "out_of_bounds_or_overlap"
    assert classify_invalid("tray slot 1 is already used") == "used_slot"
    assert classify_invalid("cannot parse JSON from reply: 'nope'") == "parse_error"
    assert classify_invalid("piece index 7 out of range 0..2") == "bad_piece_index"
    assert classify_invalid("???") == "other"


def test_analyze_run_and_render(tmp_path):
    cfg = EvalConfig(episodes=2, seed_base=2024, results_dir=tmp_path)
    run = run_eval(lambda: GreedyAgent(), cfg, agent_name="greedy")
    run_dir = write_run(run, cfg.results_dir)
    report = analyze_run(run_dir)
    assert report["agent"] == "greedy"
    assert len(report["episodes"]) == 2
    agg = report["aggregate"]
    assert agg["greedy_efficiency"] == 1.0
    assert agg["missed_clears"] == 0
    assert agg["clear_conversion"] in (None, 1.0)
    text = render_analysis(report)
    assert "greedy efficiency : 1.000" in text
    assert "seed" in text and "2024" in text
