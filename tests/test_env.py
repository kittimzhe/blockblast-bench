"""Tests for the game environment mechanics."""

import pytest

from blockblast.env import (
    BOARD_SIZE,
    TRAY_SIZE,
    BlockBlastEnv,
    InvalidActionError,
    can_place,
    clear_lines,
    full_lines,
    legal_actions_for,
    place_cells,
)
from blockblast.pieces import SHAPE_BY_NAME


def _empty_board():
    return [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]


# -- pure helpers -----------------------------------------------------------

def test_can_place_bounds_and_overlap():
    board = _empty_board()
    line5 = SHAPE_BY_NAME["5-line-h"]
    assert can_place(board, line5, 0, 0)
    assert can_place(board, line5, 0, 3)
    assert not can_place(board, line5, 0, 4)  # would exceed col 7
    assert not can_place(board, line5, -1, 0)
    board[0][0] = True
    assert not can_place(board, line5, 0, 0)  # overlap
    assert can_place(board, line5, 1, 0)


def test_place_and_full_lines():
    board = _empty_board()
    line5 = SHAPE_BY_NAME["5-line-h"]
    line3 = SHAPE_BY_NAME["3-line-h"]
    board = place_cells(board, line5, 0, 0)
    board = place_cells(board, line3, 0, 5)
    rows, cols = full_lines(board)
    assert rows == [0] and cols == []
    cleared = clear_lines(board, rows, cols)
    assert all(not v for v in cleared[0])


def test_simultaneous_row_and_col_clear():
    board = _empty_board()
    # fill row 3 except (3,0); fill col 0 except (3,0); place 1x1 at (3,0) -> both clear
    for c in range(1, 8):
        board[3][c] = True
    for r in range(8):
        if r != 3:
            board[r][0] = True
    rows, cols = full_lines(board)
    assert rows == [] and cols == []
    board = place_cells(board, SHAPE_BY_NAME["1x1"], 3, 0)
    rows, cols = full_lines(board)
    assert rows == [3] and cols == [0]


def test_legal_actions_count_on_empty_board():
    board = _empty_board()
    tray = tuple(SHAPE_BY_NAME[n] for n in ["1x1", "1x1", "1x1"])
    assert len(legal_actions_for(board, tray)) == 3 * 64
    tray = (SHAPE_BY_NAME["1x1"], None, SHAPE_BY_NAME["1x1"])
    assert len(legal_actions_for(board, tray)) == 2 * 64


# -- environment ------------------------------------------------------------

def test_reset_state():
    env = BlockBlastEnv(seed=1)
    obs = env.observation()
    assert obs.moves == 0 and obs.score == 0 and obs.combo == 0
    assert all(not v for row in obs.board for v in row)
    assert obs.pieces_in_tray == TRAY_SIZE


def test_step_places_and_scores_cells():
    env = BlockBlastEnv(seed=1)
    obs = env.observation()
    shape = obs.tray[0]
    res = env.step(0, 0, 0)
    assert res.cells == shape.size
    assert res.points == shape.size  # no clear
    assert res.observation.moves == 1
    assert res.observation.pieces_in_tray == TRAY_SIZE - 1


def test_invalid_actions_raise():
    env = BlockBlastEnv(seed=1)
    with pytest.raises(InvalidActionError):
        env.step(0, -1, 0)
    with pytest.raises(InvalidActionError):
        env.step(0, 0, 99)
    with pytest.raises(InvalidActionError):
        env.step(5, 0, 0)
    with pytest.raises(InvalidActionError):
        env.step(0, 0, 0)
        env.step(0, 0, 0)  # slot already used


def test_tray_refills_after_all_used():
    env = BlockBlastEnv(seed=3)
    obs = env.observation()
    for i in range(TRAY_SIZE):
        # first legal placement of tray slot i (board stays nearly empty)
        action = next(a for a in sorted(obs.legal_actions()) if a[0] == i)
        obs = env.step(*action).observation
    assert obs.pieces_in_tray == TRAY_SIZE


def test_row_clear_and_combo_logic():
    env = BlockBlastEnv(seed=0)
    # build a full row deterministically via direct state manipulation
    env._tray = [SHAPE_BY_NAME["1x1"]] * TRAY_SIZE
    env._board = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for c in range(1, 8):
        env._board[0][c] = True  # row 0 misses only (0,0)
    for c in range(7):
        env._board[5][c] = True  # filler so the board is not empty after the clear
    obs = env.observation()
    assert (0, 0, 0) in obs.legal_actions()
    res = env.step(0, 0, 0)
    assert res.lines == 1 and res.cleared_rows == (0,)
    assert res.combo == 1
    assert not res.perfect_clear
    assert res.points == 1 + 10  # 1 cell + 1 line x1.0
    assert all(not v for v in res.observation.board[0])
    # a non-clearing move resets the streak
    res2 = env.step(1, 7, 7)
    assert res2.combo == 0


def test_consecutive_clears_build_combo():
    env = BlockBlastEnv(seed=0)
    env._tray = [SHAPE_BY_NAME["1x1"], SHAPE_BY_NAME["2-line-h"], None]
    # row 0 misses only (0,0); row 1 misses (1,0) and (1,1)
    env._board = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for c in range(1, 8):
        env._board[0][c] = True
    for c in range(2, 8):
        env._board[1][c] = True
    for c in range(7):
        env._board[5][c] = True  # filler so no perfect clear interferes
    res1 = env.step(0, 0, 0)  # clears row 0 -> combo 1, 1 cell + 10*1.0
    assert res1.combo == 1 and res1.points == 11
    res2 = env.step(1, 1, 0)  # clears row 1 -> combo 2, 2 cells + 10*1.5
    assert res2.combo == 2 and res2.points == 17
    assert res2.observation.max_combo == 2


def test_perfect_clear_awards_bonus():
    env = BlockBlastEnv(seed=0)
    env._tray = [SHAPE_BY_NAME["1x1"]] * TRAY_SIZE
    env._board = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for c in range(7):
        env._board[4][c] = True
    res = env.step(0, 4, 7)
    assert res.perfect_clear
    assert res.points == 1 + 10 + 300
    assert res.observation.perfect_clears == 1


def test_determinism_same_seed_same_game():
    def play(seed):
        env = BlockBlastEnv(seed=seed)
        obs = env.observation()
        trays, actions_used = [], []
        scores = []
        for _ in range(30):
            legal = obs.legal_actions()
            if not legal:
                break  # "first legal action" is a weak policy; it may die early
            action = sorted(legal)[0]
            res = env.step(*action)
            trays.append([s.name if s else None for s in res.observation.tray])
            actions_used.append(res.action)
            scores.append(env.score)
            obs = res.observation
        return trays, actions_used, scores, env.score

    t1, a1, r1, s1 = play(123)
    t2, a2, r2, s2 = play(123)
    assert len(a1) > 0
    assert (t1, a1, r1, s1) == (t2, a2, r2, s2)
    t3, a3, r3, s3 = play(124)
    assert s1 != s3 or t1 != t3  # different seed -> different game (piece draws)


def test_game_over_detection():
    env = BlockBlastEnv(seed=0)
    # board nearly full: only cell (0,0) free, tray holds a 2x2 -> no fit
    env._board = [[True] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    env._board[0][0] = False
    env._tray = [SHAPE_BY_NAME["2x2"], None, None]
    obs = env.observation()
    assert obs.is_game_over()
    # but a 1x1 still fits
    env._tray[0] = SHAPE_BY_NAME["1x1"]
    assert not env.observation().is_game_over()


def test_step_accepts_action_tuple():
    env = BlockBlastEnv(seed=1)
    res = env.step((0, 0, 0))
    assert res.action == (0, 0, 0)
