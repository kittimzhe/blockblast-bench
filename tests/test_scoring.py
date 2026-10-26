"""Tests for the scoring model."""

from blockblast.scoring import ScoringConfig, combo_multiplier, compute_move_points, line_points


def test_line_points_progressive():
    cfg = ScoringConfig()
    assert line_points(0, cfg) == 0
    assert line_points(1, cfg) == 10
    assert line_points(2, cfg) == 30
    assert line_points(3, cfg) == 60


def test_line_points_flat_mode():
    cfg = ScoringConfig(multi_line_progressive=False)
    assert line_points(3, cfg) == 30


def test_combo_multiplier():
    cfg = ScoringConfig()
    assert combo_multiplier(0, cfg) == 1.0
    assert combo_multiplier(1, cfg) == 1.0
    assert combo_multiplier(2, cfg) == 1.5
    assert combo_multiplier(7, cfg) == 4.0  # capped
    assert combo_multiplier(100, cfg) == 4.0


def test_move_points_without_clear():
    assert compute_move_points(cells=5, lines=0, combo=0, perfect_clear=False) == 5


def test_move_points_single_clear_starts_streak():
    # first clear of a streak: multiplier x1
    assert compute_move_points(cells=5, lines=1, combo=1, perfect_clear=False) == 15


def test_move_points_combo_clear():
    # 5 cells + 1 line (10) * 1.5 = 5 + 15
    assert compute_move_points(cells=5, lines=1, combo=2, perfect_clear=False) == 20


def test_move_points_multi_line():
    # 0 cells + 3 lines (60) * 1.0
    assert compute_move_points(cells=0, lines=3, combo=1, perfect_clear=False) == 60


def test_perfect_clear_bonus():
    cfg = ScoringConfig(perfect_clear_bonus=300)
    assert compute_move_points(cells=3, lines=1, combo=1, perfect_clear=True, cfg=cfg) == 313


def test_custom_config():
    cfg = ScoringConfig(points_per_cell=2, points_per_line=100)
    # 4 cells x 2 + 100 x 1 line x 1.0
    assert compute_move_points(cells=4, lines=1, combo=1, perfect_clear=False, cfg=cfg) == 108
