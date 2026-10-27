"""Tests for episode resume and strengthened network retry."""

import json
import urllib.error

import pytest

from blockblast.agents import GreedyAgent, HeuristicAgent
from blockblast.agents.base import AgentError
from blockblast.agents.llm_agent import LLMAgent, LLMConfig
from blockblast.env import BlockBlastEnv
from blockblast.harness import (
    EvalConfig,
    replay_transcript,
    resume_eval,
    run_episode,
    run_eval,
)
from blockblast.report import write_run


def test_replay_transcript_restores_exact_state():
    full = run_episode(HeuristicAgent(), seed=2024, cfg=EvalConfig())
    cut = len(full.transcript) // 2
    env = BlockBlastEnv(seed=2024)
    replay_transcript(env, full.transcript[:cut])
    obs = env.observation()
    assert obs.moves == cut
    assert obs.score == full.transcript[cut - 1]["score"]
    assert list(map(lambda r: "".join("X" if v else "." for v in r), obs.board)) == \
        full.transcript[cut - 1]["board_after"]


def test_resume_from_continues_exactly_at_cut():
    """Resumed episode keeps the prefix verbatim and finishes a legal game."""
    full = run_episode(HeuristicAgent(), seed=2025, cfg=EvalConfig())
    cut = len(full.transcript) // 3
    resumed = run_episode(
        HeuristicAgent(), seed=2025, cfg=EvalConfig(), resume_from=full.transcript[:cut]
    )
    # prefix preserved move-for-move
    assert [r["board_after"] for r in resumed.transcript[:cut]] == \
        [r["board_after"] for r in full.transcript[:cut]]
    assert resumed.transcript[cut - 1]["score"] == full.transcript[cut - 1]["score"]
    # and the game continued to a genuine end
    assert resumed.moves > cut
    assert resumed.reason == "board_full"
    assert resumed.score >= full.transcript[cut - 1]["score"]


class _DyingAgent(HeuristicAgent):
    """Heuristic agent that raises AgentError after N successful moves."""

    def __init__(self, die_after):
        super().__init__()
        self._left = die_after

    @property
    def name(self):
        return "heuristic"

    def act(self, obs, feedback=None):
        if self._left <= 0:
            raise AgentError("simulated network failure")
        self._left -= 1
        return super().act(obs, feedback)


def test_resume_eval_completes_interrupted_run(tmp_path):
    results_dir = tmp_path / "results"
    cfg = EvalConfig(episodes=2, seed_base=2024, results_dir=results_dir)
    # first run dies after 5 moves per episode
    broken = run_eval(lambda: _DyingAgent(5), cfg, agent_name="heuristic")
    assert all(e.reason == "agent_error" for e in broken.results)
    broken_dir = write_run(broken, results_dir)

    # resume with a healthy agent -> both episodes complete
    fixed = resume_eval(
        lambda: HeuristicAgent(), broken_dir, cfg, agent_name="heuristic"
    )
    assert [e.seed for e in fixed.results] == [2024, 2025]
    for e, b in zip(fixed.results, broken.results):
        assert e.reason == "board_full"
        assert e.moves > b.moves  # continued past the interruption
        assert e.score > b.score
        assert e.error == ""
        # the interrupted prefix is preserved verbatim
        prefix = [r for r in b.transcript if not r.get("terminal_error")]
        assert [r["board_after"] for r in e.transcript[: len(prefix)]] == \
            [r["board_after"] for r in prefix]


def test_resume_eval_keeps_finished_episodes(tmp_path):
    results_dir = tmp_path / "results"
    cfg = EvalConfig(episodes=2, seed_base=2024, results_dir=results_dir)
    clean = run_eval(lambda: HeuristicAgent(), cfg, agent_name="heuristic")
    clean_dir = write_run(clean, results_dir)

    calls = {"n": 0}

    def counting_factory():
        calls["n"] += 1
        return HeuristicAgent()

    again = resume_eval(counting_factory, clean_dir, cfg, agent_name="heuristic")
    assert calls["n"] == 0  # nothing to redo
    assert [e.score for e in again.results] == [e.score for e in clean.results]


def test_network_retry_backoff_and_recovery(monkeypatch):
    """Transient 503s and timeouts are retried with backoff; then success."""
    monkeypatch.setenv("FAKE_KEY", "k")
    sleeps = []
    monkeypatch.setattr("blockblast.agents.llm_agent.time.sleep", lambda s: sleeps.append(s))

    state = {"calls": 0}

    def flaky(cfg, api_key, messages, timeout=None):
        state["calls"] += 1
        if state["calls"] == 1:
            raise urllib.error.HTTPError("u", 503, "unavailable", {}, None)
        if state["calls"] == 2:
            raise TimeoutError("The read operation timed out")
        return json.dumps({"piece": 0, "row": 0, "col": 0}), {"total_tokens": 10}

    monkeypatch.setattr("blockblast.agents.llm_agent._post_chat", flaky)
    agent = LLMAgent(LLMConfig(api_key_env="FAKE_KEY", backoff_base=1.0))
    env = BlockBlastEnv(seed=1)
    action = agent.act(env.observation())
    assert action.as_tuple() == (0, 0, 0)
    assert state["calls"] == 3
    assert agent.stats["network_errors"] == 2
    assert sleeps == [1.0, 2.0]  # exponential backoff


def test_network_retry_timeout_grows(monkeypatch):
    """After a read timeout the next attempt gets a larger timeout budget."""
    monkeypatch.setenv("FAKE_KEY", "k")
    monkeypatch.setattr("blockblast.agents.llm_agent.time.sleep", lambda s: None)
    seen = []

    def flaky(cfg, api_key, messages, timeout=None):
        seen.append(timeout)
        if len(seen) < 3:
            raise TimeoutError("timed out")
        return json.dumps({"piece": 0, "row": 0, "col": 0}), {}

    monkeypatch.setattr("blockblast.agents.llm_agent._post_chat", flaky)
    agent = LLMAgent(LLMConfig(api_key_env="FAKE_KEY", timeout=30.0, timeout_cap=100.0))
    agent.act(BlockBlastEnv(seed=1).observation())
    assert seen == [30.0, 60.0, 100.0]  # doubled, then capped


def test_non_retryable_http_fails_fast(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "k")

    def unauthorized(cfg, api_key, messages, timeout=None):
        raise urllib.error.HTTPError("u", 401, "unauthorized", {}, None)

    monkeypatch.setattr("blockblast.agents.llm_agent._post_chat", unauthorized)
    agent = LLMAgent(LLMConfig(api_key_env="FAKE_KEY"))
    with pytest.raises(AgentError, match="HTTP 401"):
        agent.act(BlockBlastEnv(seed=1).observation())


def test_non_json_200_body_is_retried(monkeypatch):
    """A 200 response with a non-JSON/empty body is transient, not fatal."""
    monkeypatch.setenv("FAKE_KEY", "k")
    monkeypatch.setattr("blockblast.agents.llm_agent.time.sleep", lambda s: None)
    from blockblast.agents.llm_agent import TransientEndpointError

    state = {"calls": 0}

    def flaky(cfg, api_key, messages, timeout=None):
        state["calls"] += 1
        if state["calls"] == 1:
            raise TransientEndpointError("endpoint returned non-JSON body: ''")
        return json.dumps({"piece": 0, "row": 0, "col": 0}), {}

    monkeypatch.setattr("blockblast.agents.llm_agent._post_chat", flaky)
    agent = LLMAgent(LLMConfig(api_key_env="FAKE_KEY"))
    action = agent.act(BlockBlastEnv(seed=1).observation())
    assert action.as_tuple() == (0, 0, 0)
    assert agent.stats["network_errors"] == 1


def test_unexpected_agent_exception_does_not_kill_run():
    """A bug in an agent ends that episode as agent_error, not the whole run."""

    class Buggy(HeuristicAgent):
        @property
        def name(self):
            return "buggy"

        def act(self, obs, feedback=None):
            raise ValueError("boom")  # not an AgentError subclass

    result = run_episode(Buggy(), seed=1, cfg=EvalConfig())
    assert result.reason == "agent_error"
    assert "boom" in result.error
    assert result.moves == 0


def test_resume_records_lineage(tmp_path):
    """resume_eval must stamp resumed_from so chains stay traceable."""
    from blockblast.harness import EvalConfig, resume_eval, run_eval
    from blockblast.report import write_run

    cfg = EvalConfig(episodes=2, seed_base=2024, max_moves=6, results_dir=tmp_path)
    first = run_eval(lambda: GreedyAgent(), cfg, agent_name="greedy")
    write_run(first, tmp_path)
    assert first.resumed_from is None

    second = resume_eval(lambda: GreedyAgent(), tmp_path / first.run_id, cfg, "greedy")
    write_run(second, tmp_path)
    assert second.resumed_from == first.run_id

    summary = json.loads((tmp_path / second.run_id / "summary.json").read_text())
    assert summary["resumed_from"] == first.run_id


def test_collapse_resume_chains_keeps_only_final_link():
    from blockblast.report import collapse_resume_chains

    rows = [
        {"run_id": "a", "resumed_from": None},
        {"run_id": "b", "resumed_from": "a"},
        {"run_id": "c", "resumed_from": "b"},
        {"run_id": "standalone", "resumed_from": None},
        {"run_id": "legacy"},  # no lineage metadata at all
    ]
    kept = {r["run_id"] for r in collapse_resume_chains(rows)}
    assert kept == {"c", "standalone", "legacy"}
