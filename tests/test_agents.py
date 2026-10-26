"""Tests for agents: baselines + the LLM agent (with a fake endpoint)."""

import json
import re

import pytest

from blockblast.agents import GreedyAgent, HeuristicAgent, RandomAgent
from blockblast.agents.base import Action, AgentParseError
from blockblast.agents.llm_agent import LLMAgent, LLMConfig, parse_action
from blockblast.harness import EvalConfig, run_episode
from blockblast.pieces import SHAPE_BY_NAME


def _episode(agent, seed=11):
    cfg = EvalConfig(episodes=1, seed_base=seed)
    return run_episode(agent, seed, cfg)


def test_random_agent_plays_full_game():
    result = _episode(RandomAgent(), seed=11)
    assert result.moves > 0
    assert result.reason in ("board_full", "move_cap")
    assert result.invalid_actions == 0


def test_greedy_prefers_clear():
    from blockblast.env import BlockBlastEnv

    env = BlockBlastEnv(seed=0)
    # leave row 0 one cell short, give a 1x1 in the tray
    env._tray = [SHAPE_BY_NAME["1x1"], SHAPE_BY_NAME["1x1"], None]
    env._board = [[False] * 8 for _ in range(8)]
    for c in range(1, 8):
        env._board[0][c] = True
    obs = env.observation()
    agent = GreedyAgent()
    agent.reset(seed=1)
    assert agent.act(obs) == Action(0, 0, 0)


def test_greedy_beats_random_over_seed_set():
    scores = {}
    for name, agent in (("greedy", GreedyAgent()), ("random", RandomAgent())):
        total = 0
        for seed in range(100, 110):
            result = _episode(agent, seed=seed)
            total += result.score
        scores[name] = total / 10
    assert scores["greedy"] > scores["random"]


def test_heuristic_beats_random_over_seed_set():
    scores = {}
    for name, agent in (("heuristic", HeuristicAgent()), ("random", RandomAgent())):
        total = 0
        for seed in range(200, 210):
            result = _episode(agent, seed=seed)
            total += result.score
        scores[name] = total / 10
    assert scores["heuristic"] > scores["random"]


# -- LLM agent ---------------------------------------------------------------

def test_parse_action_clean_json():
    assert parse_action('{"piece": 1, "row": 2, "col": 3}') == Action(1, 2, 3)


def test_parse_action_fenced_json():
    text = 'Here is my move:\n```json\n{"piece": 0, "row": 7, "col": 1}\n```'
    assert parse_action(text) == Action(0, 7, 1)


def test_parse_action_with_prose():
    assert parse_action('I think {"piece": 2, "row": 0, "col": 4} is best.') == Action(2, 0, 4)


def test_parse_action_rejects_garbage():
    with pytest.raises(AgentParseError):
        parse_action("no json here")
    with pytest.raises(AgentParseError):
        parse_action('{"row": 1, "col": 1}')  # missing piece
    with pytest.raises(AgentParseError):
        parse_action('{"piece": "zero", "row": 1, "col": 1}')


class _FakeEndpoint:
    """Simulates an LLM that plays greedily by parsing the rendered board."""

    def __init__(self):
        self.calls = 0

    def __call__(self, cfg, api_key, messages, timeout=None):
        self.calls += 1
        user = messages[-1]["content"]
        action = self._first_legal(user)
        content = json.dumps(
            {"piece": action[0], "row": action[1], "col": action[2]}
        )
        usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        return content, usage

    @staticmethod
    def _first_legal(user: str):
        # parse the 8 board rows from the observation text
        board = []
        for m in re.finditer(r"^([0-7])\s+([.X ]+)$", user, re.M):
            board.append([ch == "X" for ch in m.group(2).split()])
        occupied = {
            (r, c)
            for r, row in enumerate(board)
            for c, v in enumerate(row)
            if v
        }
        # pick the first tray slot that is not used
        for idx in range(3):
            marker = f"[{idx}] "
            pos = user.find(marker)
            if pos == -1:
                continue
            first = user[pos + len(marker) :].split("\n")[0].split()[0]
            if first == "(used)":
                continue
            shape = SHAPE_BY_NAME[first]
            for r in range(9 - shape.rows):
                for c in range(9 - shape.cols):
                    if all(
                        (r + dr, c + dc) not in occupied for dr, dc in shape.cells
                    ):
                        return (idx, r, c)
        raise AssertionError("fake endpoint found no legal move")


def test_llm_agent_plays_with_fake_endpoint(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "test-key")
    monkeypatch.setattr("blockblast.agents.llm_agent._post_chat", _FakeEndpoint())
    cfg = LLMConfig(base_url="http://localhost:1/v1", model="fake-model", api_key_env="FAKE_KEY")
    agent = LLMAgent(cfg)
    result = _episode(agent, seed=11)
    assert result.moves > 0
    assert result.reason in ("board_full", "move_cap")
    assert result.invalid_actions == 0
    assert result.agent_stats["requests"] == result.moves
    assert result.agent_stats["total_tokens"] == 110 * result.moves


def test_llm_agent_missing_key_fails_fast(monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY_VAR", raising=False)
    with pytest.raises(RuntimeError, match="missing API key"):
        LLMAgent(LLMConfig(api_key_env="NO_SUCH_KEY_VAR"))


def test_llm_agent_invalid_reply_counts_and_retries(monkeypatch):
    """An out-of-range reply is rejected by the env, fed back, then recovered."""
    monkeypatch.setenv("FAKE_KEY", "k")
    endpoint = _FakeEndpoint()
    state = {"bad_left": 1}

    def flaky(cfg, api_key, messages, timeout=None):
        if state["bad_left"] > 0:
            state["bad_left"] -= 1
            return json.dumps({"piece": 0, "row": 8, "col": 8}), {}  # out of range
        return endpoint(cfg, api_key, messages)

    monkeypatch.setattr("blockblast.agents.llm_agent._post_chat", flaky)
    cfg = LLMConfig(base_url="http://localhost:1/v1", model="fake-model", api_key_env="FAKE_KEY")
    agent = LLMAgent(cfg)
    result = _episode(agent, seed=11)
    assert result.moves > 0
    assert result.invalid_actions == 1
    assert result.reason in ("board_full", "move_cap")
    with_invalid = [r for r in result.transcript if r.get("invalid_attempts")]
    assert with_invalid and len(with_invalid[0]["invalid_attempts"]) == 1
