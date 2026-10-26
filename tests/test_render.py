"""Tests for text rendering (board / tray / observation / step)."""

from blockblast.env import BlockBlastEnv
from blockblast.render import (
    board_rows_as_strings,
    render_board,
    render_observation,
    render_shape,
    render_step,
    render_tray,
)
from blockblast.pieces import SHAPE_BY_NAME


def test_render_board_has_indices_and_8_rows():
    env = BlockBlastEnv(seed=1)
    env.step(0, 0, 0)
    text = render_board(env.observation().board)
    lines = text.splitlines()
    assert lines[0].strip() == "0 1 2 3 4 5 6 7"
    assert len(lines) == 9
    assert lines[1].startswith("0   ")
    assert "X" in lines[1]


def test_render_shape_dimensions():
    text = render_shape(SHAPE_BY_NAME["2x2"])
    assert text.splitlines() == ["    X X", "    X X"]
    text = render_shape(SHAPE_BY_NAME["3-line-v"])
    assert text.splitlines() == ["    X", "    X", "    X"]


def test_render_tray_marks_used_slots():
    text = render_tray((SHAPE_BY_NAME["1x1"], None, None))
    assert "[0] 1x1" in text
    assert "[1] (used)" in text


def test_render_observation_is_parseable():
    env = BlockBlastEnv(seed=3)
    obs = env.observation()
    text = render_observation(obs)
    assert text.count("\n") >= 10
    assert "BOARD" in text and "TRAY" in text
    assert f"score={obs.score}" in text
    for idx in range(3):
        assert f"[{idx}]" in text


def test_render_step_mentions_clear():
    env = BlockBlastEnv(seed=0)
    env._tray = [SHAPE_BY_NAME["1x1"], SHAPE_BY_NAME["1x1"], None]
    env._board = [[False] * 8 for _ in range(8)]
    for c in range(1, 8):
        env._board[0][c] = True  # row 0 misses only (0,0)
    for c in range(7):
        env._board[5][c] = True  # filler -> no perfect clear
    res = env.step(0, 0, 0)
    text = render_step(res)
    assert "cleared 1 line" in text
    assert "combo x1" in text
    assert "+11 pts" in text


def test_board_rows_as_strings():
    env = BlockBlastEnv(seed=1)
    env.step(0, 0, 0)
    rows = board_rows_as_strings(env.observation().board)
    assert len(rows) == 8
    assert all(len(r) == 8 for r in rows)
    assert set("".join(rows)) <= {".", "X"}
