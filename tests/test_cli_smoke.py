"""End-to-end smoke tests: every CLI subcommand over a real (tiny) run."""

from __future__ import annotations

import json

import pytest

from blockblast.cli import main


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """One tiny real eval run shared by all CLI tests in this module."""
    results = tmp_path_factory.mktemp("results")
    rc = main([
        "run", "--agent", "greedy", "--episodes", "2",
        "--seed-base", "2024", "--results-dir", str(results), "--tag", "smoke",
    ])
    assert rc == 0
    dirs = [p for p in results.iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_cli_demo(capsys):
    assert main(["demo", "--agent", "greedy", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "demo: agent=greedy" in out
    assert "final score" in out.lower() or "score" in out


def test_cli_shapes(capsys):
    assert main(["shapes"]) == 0
    out = capsys.readouterr().out
    assert "19 pieces" in out
    assert "1x1" in out and "3x3" in out
    assert out.count("X") >= 19  # every shape previews its cells


def test_cli_run_artifacts(run_dir):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["agent"] == "greedy"
    assert len(summary["episodes_detail"]) == 2
    episodes = sorted((run_dir / "episodes").glob("seed_*.json"))
    assert len(episodes) == 2
    runs_jsonl = run_dir.parent / "runs.jsonl"
    leaderboard = run_dir.parent / "leaderboard.md"
    assert runs_jsonl.exists() and leaderboard.exists()
    assert "greedy" in leaderboard.read_text(encoding="utf-8")


def test_cli_leaderboard(run_dir, capsys):
    assert main(["leaderboard", "--results-dir", str(run_dir.parent)]) == 0
    out = capsys.readouterr().out
    assert "greedy" in out and "|" in out


def test_cli_replay(run_dir, capsys):
    ep = sorted((run_dir / "episodes").glob("seed_*.json"))[0]
    assert main(["replay", str(ep)]) == 0
    out = capsys.readouterr().out
    assert "move" in out.lower()
    assert "score" in out.lower()


def test_cli_analyze(run_dir, capsys):
    assert main(["analyze", str(run_dir), "--json"]) == 0
    out = capsys.readouterr().out
    assert "greedy efficiency : 1.000" in out  # greedy is the single-step optimum
    report = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    assert report["aggregate"]["greedy_efficiency"] == 1.0


def test_cli_watch(run_dir, capsys):
    ep = sorted((run_dir / "episodes").glob("seed_*.json"))[0]
    assert main(["watch", str(ep), "--delay", "0"]) == 0
    out = capsys.readouterr().out
    assert "score" in out.lower()


def test_cli_gif(run_dir, tmp_path):
    ep = sorted((run_dir / "episodes").glob("seed_*.json"))[0]
    out_gif = tmp_path / "ep.gif"
    assert main(["gif", str(ep), "-o", str(out_gif)]) == 0
    data = out_gif.read_bytes()
    assert data[:6] in (b"GIF87a", b"GIF89a")
    assert len(data) > 500


def test_cli_analyze_rejects_non_run_dir(tmp_path, capsys):
    assert main(["analyze", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "summary.json" in err


def test_cli_resume_rejects_missing_run(tmp_path, capsys):
    rc = main([
        "run", "--agent", "greedy", "--episodes", "1",
        "--results-dir", str(tmp_path), "--resume", str(tmp_path / "nope"),
    ])
    assert rc == 2


def test_cli_report(run_dir, tmp_path, capsys):
    out = tmp_path / "site" / "index.html"
    assert main([
        "report", "--results-dir", str(run_dir.parent), "--output", str(out),
        "--gif", str(run_dir.parent.parent / "nonexistent.gif"),
    ]) == 0
    page = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in page
    assert "greedy" in page
    assert "<svg" in page  # per-episode chart present
    assert "暂无回放" in page or 'class="gif"' in page
