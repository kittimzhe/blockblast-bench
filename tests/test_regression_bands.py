"""Statistical regression tests: baseline scores must stay in known bands.

These catch accidental changes to engine semantics or agent logic that the
exact golden tests can't see (e.g. a 'better' heuristic accidentally made
worse). Bands are wide enough to never flake — the env and agents are fully
deterministic for a fixed seed set, so any drift at all means code changed
behaviour.
"""

from __future__ import annotations

from blockblast.agents import GreedyAgent, HeuristicAgent, RandomAgent
from blockblast.harness import EvalConfig, run_eval

SEEDS = EvalConfig(episodes=5, seed_base=2024)  # 5 episodes keeps this fast


def _mean_score(factory) -> float:
    run = run_eval(factory, SEEDS, agent_name="x")
    return sum(e.score for e in run.results) / len(run.results)


def test_baseline_score_bands():
    random_mean = _mean_score(lambda: RandomAgent(seed=0))
    greedy_mean = _mean_score(lambda: GreedyAgent())
    heuristic_mean = _mean_score(lambda: HeuristicAgent())

    # deterministic given fixed seeds — bands have generous head-room anyway
    assert 20 <= random_mean <= 150, random_mean
    assert 80 <= greedy_mean <= 400, greedy_mean
    assert 250 <= heuristic_mean <= 1200, heuristic_mean

    # the capability ladder must hold: it is the benchmark's yardstick
    assert random_mean < greedy_mean < heuristic_mean


def test_all_baseline_episodes_finish_cleanly():
    for factory in (lambda: RandomAgent(seed=0), lambda: GreedyAgent(), lambda: HeuristicAgent()):
        run = run_eval(factory, SEEDS, agent_name="x")
        for ep in run.results:
            assert ep.reason == "board_full"
            assert ep.error == ""
            assert ep.invalid_actions == 0  # baselines only propose legal moves
