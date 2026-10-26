"""Tests for the piece catalog."""

from blockblast.pieces import SHAPE_BY_NAME, SHAPE_CATALOG, Shape


def test_catalog_size():
    assert len(SHAPE_CATALOG) == 19


def test_shapes_are_normalised():
    for shape in SHAPE_CATALOG:
        assert min(r for r, _ in shape.cells) == 0
        assert min(c for _, c in shape.cells) == 0


def test_shapes_have_unique_cells():
    for shape in SHAPE_CATALOG:
        assert len(set(shape.cells)) == len(shape.cells)
        assert len(shape.cells) >= 1


def test_bounding_boxes_match_cells():
    for shape in SHAPE_CATALOG:
        assert shape.rows == 1 + max(r for r, _ in shape.cells)
        assert shape.cols == 1 + max(c for _, c in shape.cells)
        assert shape.size == len(shape.cells)


def test_expected_shapes_present():
    expected = {"1x1", "2-line-h", "5-line-v", "2x2", "3x3", "L3-A", "L5-A"}
    assert expected.issubset(set(SHAPE_BY_NAME))
