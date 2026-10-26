"""Property-based invariant tests and fuzz tests for the game engine.

No third-party property-testing library (the project is stdlib-only):
randomised play is driven by seeded ``random.Random`` instances, so every
failure is reproducible from the printed seed.
"""

from __future__ import annotations

import random

import pytest

from blockblast.env import (
    BOARD_SIZE,
    TRAY_SIZE,
    BlockBlastEnv,
    InvalidActionError,
)
from blockblast.scoring import compute_move_points

N_GAMES = 60  # random games per invariant suite (each ~10-30 moves)


def _filled(board) -> int:
    return sum(1 for row in board for v in row if v)


def _snapshot(env: BlockBlastEnv):
    """Full observable state, for 'exception must not mutate' checks."""
    obs = env.observation()
    return (
        obs.board,
        tuple(s.name if s else None for s in obs.tray),
        obs.score,
        obs.combo,
        obs.moves,
        obs.lines_cleared,
        obs.max_combo,
        obs.perfect_clears,
    )


def _play_random_game(seed: int, check) -> None:
    """Play one full random game, calling ``check(env, before, action, res)``
    after every successful step."""
    env = BlockBlastEnv(seed=seed)
    rng = random.Random(seed ^ 0xA5A5)
    for _ in range(500):  # hard cap; games end well before this
        legal = env.legal_actions()
        if not legal:
            break
        action = rng.choice(legal)
        before = env.observation()
        res = env.step(*action)
        check(env, before, action, res)


# ---------------------------------------------------------------------------
# invariants over random play
# ---------------------------------------------------------------------------

def test_invariant_cell_conservation():
    """filled_after == filled_before + placed - cleared-union size."""

    def check(env, before, action, res):
        rows, cols = res.cleared_rows, res.cleared_cols
        cleared_union = (
            len(rows) * BOARD_SIZE + len(cols) * BOARD_SIZE - len(rows) * len(cols)
        )
        got = _filled(res.observation.board)
        want = _filled(before.board) + res.cells - cleared_union
        assert got == want, f"seed game: cells leaked ({got} != {want})"

    for seed in range(N_GAMES):
        _play_random_game(seed, check)


def test_invariant_cleared_lines_are_empty():
    def check(env, before, action, res):
        b = res.observation.board
        for r in res.cleared_rows:
            assert not any(b[r]), f"cleared row {r} not empty"
        for c in res.cleared_cols:
            assert not any(b[r][c] for r in range(BOARD_SIZE)), f"cleared col {c} not empty"

    for seed in range(N_GAMES):
        _play_random_game(seed, check)


def test_invariant_combo_and_score_accounting():
    """combo increments exactly on clears; score/moves/lines accumulate; points re-derivable."""

    def check(env, before, action, res):
        if res.lines > 0:
            assert res.combo == before.combo + 1
        else:
            assert res.combo == 0
        obs = res.observation
        assert obs.moves == before.moves + 1
        assert obs.lines_cleared == before.lines_cleared + res.lines
        assert obs.score == before.score + res.points
        assert res.points >= res.cells >= 1  # score strictly increases
        assert obs.max_combo >= obs.combo
        # the reported points match the documented scoring formula
        assert res.points == compute_move_points(
            cells=res.cells,
            lines=res.lines,
            combo=res.combo,
            perfect_clear=res.perfect_clear,
            cfg=env.scoring,
        )

    for seed in range(N_GAMES):
        _play_random_game(seed, check)


def test_invariant_tray_lifecycle():
    """Used slot becomes None; tray refills to 3 exactly when emptied."""

    def check(env, before, action, res):
        piece, _, _ = action
        obs = res.observation
        used_before = before.pieces_in_tray
        if used_before == 1:  # last piece -> batch refill
            assert obs.pieces_in_tray == TRAY_SIZE
            assert all(s is not None for s in obs.tray)
        else:
            assert obs.pieces_in_tray == used_before - 1
            assert obs.tray[piece] is None

    for seed in range(N_GAMES):
        _play_random_game(seed, check)


def test_invariant_game_over_iff_no_legal_actions():
    for seed in range(N_GAMES):
        env = BlockBlastEnv(seed=seed)
        rng = random.Random(seed ^ 0x5A5A)
        while True:
            legal = env.legal_actions()
            assert env.is_game_over() == (len(legal) == 0)
            if not legal:
                # every remaining tray piece really fits nowhere
                obs = env.observation()
                assert obs.pieces_in_tray >= 1
                break
            env.step(*rng.choice(legal))


def test_invariant_perfect_clear_board_empty():
    """perfect_clear flag is set iff the post-step board is empty."""
    seen_perfect = 0

    def check(env, before, action, res):
        nonlocal seen_perfect
        empty = _filled(res.observation.board) == 0
        assert res.perfect_clear == empty
        if res.perfect_clear:
            seen_perfect += 1
            assert res.lines > 0  # a perfect clear requires a clear

    for seed in range(300):  # perfect clears are rare: scan more games
        _play_random_game(seed, check)


def test_invariant_replay_determinism():
    """Same seed + same action sequence => identical final state."""
    for seed in (7, 77, 777):
        env1 = BlockBlastEnv(seed=seed)
        rng = random.Random(seed)
        actions = []
        while True:
            legal = env1.legal_actions()
            if not legal:
                break
            a = rng.choice(legal)
            actions.append(a)
            env1.step(*a)
        env2 = BlockBlastEnv(seed=seed)
        for a in actions:
            env2.step(*a)
        assert env2.observation() == env1.observation()
        assert env2.score == env1.score


# ---------------------------------------------------------------------------
# fuzz: hostile inputs must raise InvalidActionError and mutate nothing
# ---------------------------------------------------------------------------

HOSTILE_ACTIONS = [
    (-1, 0, 0), (3, 0, 0), (99, 0, 0),           # piece index out of range
    (0, -1, 0), (0, 0, -1), (0, 8, 0), (0, 0, 8),  # off-board
    (0, 7, 7),                                    # likely out of bounds for multi-cell pieces
    ("0", 0, 0), (0, "1", 0), (0, 0, None),       # wrong types
    (0, 0.5, 0), (True, 0, 0),                    # near-integers (bool is int: allowed types-wise)
]


def test_fuzz_hostile_actions_raise_and_do_not_mutate():
    rng = random.Random(1234)
    for seed in range(30):
        env = BlockBlastEnv(seed=seed)
        # advance to a random mid-game state
        for _ in range(rng.randrange(0, 8)):
            legal = env.legal_actions()
            if not legal:
                break
            env.step(*rng.choice(legal))
        legal = set(env.legal_actions())
        before = _snapshot(env)
        for action in HOSTILE_ACTIONS:
            if action in legal:  # (True,0,0) == (1,0,0) may genuinely be legal
                continue
            with pytest.raises(InvalidActionError):
                env.step(*action)
            assert _snapshot(env) == before, f"{action} mutated env state"


def test_fuzz_random_triples_never_crash_engine():
    """Any integer triple either steps legally or raises InvalidActionError."""
    rng = random.Random(99)
    env = BlockBlastEnv(seed=0)
    successes = 0
    for i in range(3000):
        if i % 100 == 0:  # keep boards sparse so legal hits stay reachable
            env.reset(seed=rng.randrange(10_000))
        action = (rng.randrange(-2, 5), rng.randrange(-3, 11), rng.randrange(-3, 11))
        legal = set(env.legal_actions())
        if not legal:
            env.reset(seed=rng.randrange(10_000))
            continue
        before = _snapshot(env)
        try:
            env.step(*action)
            successes += 1
            assert action in legal, f"{action} accepted but was not legal"
        except InvalidActionError:
            assert action not in legal, f"{action} rejected but was legal"
            assert _snapshot(env) == before
    assert successes > 50  # the fuzz actually exercised the success path


def test_fuzz_used_slot_rejected():
    rng = random.Random(5)
    for seed in range(20):
        env = BlockBlastEnv(seed=seed)
        legal = env.legal_actions()
        if not legal:
            continue
        piece, row, col = rng.choice(legal)
        env.step(piece, row, col)
        if env.observation().tray[piece] is None:  # not refilled
            before = _snapshot(env)
            with pytest.raises(InvalidActionError, match="already used"):
                env.step(piece, 0, 0)
            assert _snapshot(env) == before


def test_fuzz_malformed_action_shapes():
    env = BlockBlastEnv(seed=0)
    before = _snapshot(env)
    for bad in [(), (1,), (1, 2), (1, 2, 3, 4)]:
        with pytest.raises(InvalidActionError):
            env.step(*bad)
        assert _snapshot(env) == before
    # tuple-style call is allowed; a malformed tuple is not
    with pytest.raises((InvalidActionError, ValueError)):
        env.step((1, 2))


# ---------------------------------------------------------------------------
# golden determinism: behaviour changes must be deliberate
# ---------------------------------------------------------------------------

def _golden_play(seed: int):
    """Deterministic policy: always the first legal action."""
    env = BlockBlastEnv(seed=seed)
    while True:
        legal = env.legal_actions()
        if not legal:
            return env.score, env.moves, env.lines_cleared
        env.step(*legal[0])


def test_golden_fixed_seed_games():
    """Hash-pinned outcomes for fixed seeds and a fixed policy.

    A failure here means engine semantics changed: piece catalog order,
    RNG draws, clearing rules or scoring. If the change is intentional,
    update the golden values and invalidate existing leaderboards.
    """
    golden = {
        0: (114, 17, 4),
        1: (55, 11, 1),
        2: (425, 23, 5),
        42: (120, 20, 5),
        2024: (104, 16, 3),
    }
    actual = {seed: _golden_play(seed) for seed in golden}
    assert actual == golden
