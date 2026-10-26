"""inspect_ai Task adapter for blockblast-bench.

Each Sample is one episode (one seed). A custom solver drives the
deterministic BlockBlastEnv game loop, asking the active inspect model
for a move each turn; a custom scorer reports the final game score.

Run (any provider inspect supports):

    inspect eval blockblast/inspect_task.py --model openai/gpt-4o-mini
    inspect eval blockblast/inspect_task.py -T episodes=10 -T seed_base=2024

For an OpenAI-compatible endpoint such as the mass platform:

    export MASS_API_KEY=...
    inspect eval blockblast/inspect_task.py \
        --model openai-api/mass/qwen3.7-plus \
        -M base_url=https://maas.appia.vip/api/open-apis/v1

Requires Python 3.10+ with inspect-ai installed (the core benchmark
itself stays 3.9-compatible and dependency-free).
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, accuracy, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from blockblast.agents.base import AgentParseError
from blockblast.agents.llm_agent import DEFAULT_SYSTEM_PROMPT, RETRY_SUFFIX, parse_action
from blockblast.env import BlockBlastEnv, InvalidActionError
from blockblast.harness import (
    REASON_BOARD_FULL,
    REASON_INVALID,
    REASON_MOVE_CAP,
)
from blockblast.render import render_observation


@solver
def play_blockblast(max_moves: int = 3000, invalid_action_retries: int = 2):
    """Game-loop solver: one full episode against the active model."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        seed = int(state.metadata["seed"])
        model = get_model()
        env = BlockBlastEnv(seed=seed)
        obs = env.observation()
        invalid_total = 0
        moves = 0
        reason = REASON_BOARD_FULL

        while True:
            if obs.is_game_over():
                reason = REASON_BOARD_FULL
                break
            if moves >= max_moves:
                reason = REASON_MOVE_CAP
                break

            feedback = None
            step_res = None
            for _attempt in range(invalid_action_retries + 1):
                user = render_observation(obs)
                if feedback:
                    user += RETRY_SUFFIX.format(feedback=feedback)
                output = await model.generate(
                    [
                        ChatMessageSystem(content=DEFAULT_SYSTEM_PROMPT),
                        ChatMessageUser(content=user),
                    ]
                )
                try:
                    action = parse_action(output.completion)
                    step_res = env.step(action.piece, action.row, action.col)
                    break
                except (AgentParseError, InvalidActionError) as exc:
                    invalid_total += 1
                    feedback = str(exc)
            if step_res is None:
                reason = REASON_INVALID
                break
            moves += 1
            obs = step_res.observation

        state.metadata["score"] = env.score
        state.metadata["moves"] = env.moves
        state.metadata["lines_cleared"] = env.lines_cleared
        state.metadata["max_combo"] = env.max_combo
        state.metadata["perfect_clears"] = env.perfect_clears
        state.metadata["invalid_actions"] = invalid_total
        state.metadata["reason"] = reason
        state.output.completion = (
            f"score={env.score} moves={env.moves} lines={env.lines_cleared} "
            f"max_combo={env.max_combo} invalid={invalid_total} reason={reason}"
        )
        return state

    return solve


@scorer(metrics=[mean(), stderr()])
def game_score():
    """Final game score as the Score value, rich details in metadata."""

    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        return Score(
            value=float(md.get("score", 0)),
            answer=state.output.completion,
            metadata={
                k: md.get(k)
                for k in (
                    "seed",
                    "moves",
                    "lines_cleared",
                    "max_combo",
                    "perfect_clears",
                    "invalid_actions",
                    "reason",
                )
            },
        )

    return score


@scorer(metrics=[accuracy()])
def finished_cleanly():
    """1.0 when the episode ended by the rules (board full / move cap)."""

    async def score(state: TaskState, target: Target) -> Score:
        ok = state.metadata.get("reason") in (REASON_BOARD_FULL, REASON_MOVE_CAP)
        return Score(value=1.0 if ok else 0.0)

    return score


@task
def blockblast(
    episodes: int = 10,
    seed_base: int = 2024,
    max_moves: int = 3000,
    invalid_action_retries: int = 2,
) -> Task:
    """Block Blast spatial-planning benchmark (one sample per seed)."""
    dataset = [
        Sample(
            id=f"seed_{seed_base + i}",
            input=f"Play Block Blast episode with seed {seed_base + i}.",
            metadata={"seed": seed_base + i},
        )
        for i in range(episodes)
    ]
    return Task(
        dataset=dataset,
        solver=play_blockblast(
            max_moves=max_moves, invalid_action_retries=invalid_action_retries
        ),
        scorer=[game_score(), finished_cleanly()],
    )
