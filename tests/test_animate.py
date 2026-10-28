"""Tests for the GIF encoder and episode animation."""

import json

import pytest

from blockblast.agents import HeuristicAgent
from blockblast.animate import PALETTE, episode_frames, transcript_to_gif
from blockblast.gifenc import lzw_encode, write_gif
from blockblast.harness import EvalConfig, run_episode

PIL = pytest.importorskip("PIL.Image", reason="Pillow needed for GIF decode checks")


def _transcript(seed=2024):
    return run_episode(HeuristicAgent(), seed=seed, cfg=EvalConfig()).transcript


def test_write_gif_roundtrip_single_frame(tmp_path):
    # 4x2 image, 3 colours, deterministic pattern
    frame = [0, 1, 2, 1, 2, 0, 1, 0]
    path = tmp_path / "one.gif"
    write_gif(path, (4, 2), [(255, 0, 0), (0, 255, 0), (0, 0, 255)], [frame], [10])
    img = PIL.open(path)
    assert img.size == (4, 2)
    rgb = img.convert("RGB")
    expect = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
    for i, idx in enumerate(frame):
        assert rgb.getpixel((i % 4, i // 4)) == expect[idx]


def test_write_gif_animation_frames_and_loop(tmp_path):
    w, h = 6, 6
    frames = [[(f + x + y) % 4 for y in range(h) for x in range(w)] for f in range(5)]
    path = tmp_path / "anim.gif"
    write_gif(path, (w, h), PALETTE[:4], frames, [8] * 5, loop=0)
    img = PIL.open(path)
    assert getattr(img, "n_frames", 1) == 5
    assert img.info.get("loop") == 0
    # decode every frame and compare pixels
    for f in range(5):
        img.seek(f)
        rgb = img.convert("RGB")
        for y in range(h):
            for x in range(w):
                assert rgb.getpixel((x, y)) == PALETTE[frames[f][y * w + x]]


def test_lzw_long_stream_with_dict_reset(tmp_path):
    # enough distinct patterns to push past the 12-bit dictionary limit
    import random

    rng = random.Random(7)
    w, h = 64, 64
    frame = [rng.randrange(16) for _ in range(w * h)]
    path = tmp_path / "big.gif"
    write_gif(path, (w, h), [(i * 16, i * 8, 255 - i * 16) for i in range(16)], [frame], [10])
    img = PIL.open(path)
    rgb = img.convert("RGB")
    pal = [(i * 16, i * 8, 255 - i * 16) for i in range(16)]
    for y in range(h):
        for x in range(w):
            assert rgb.getpixel((x, y)) == pal[frame[y * w + x]]


def test_episode_frames_structure():
    transcript = _transcript()
    frames, delays, (w, h) = episode_frames(transcript)
    assert len(frames) == len(delays)
    # intro + >= 2 frames per move (placement + settle), clears add a third
    n_moves = len(transcript)
    n_clears = sum(1 for r in transcript if r.get("lines"))
    assert len(frames) == 1 + 2 * n_moves + n_clears
    assert all(len(f) == w * h for f in frames)
    assert all(0 <= v < len(PALETTE) for v in frames[0])


def test_transcript_to_gif(tmp_path):
    transcript = _transcript()
    out = tmp_path / "episode.gif"
    n_frames, (w, h) = transcript_to_gif(out, transcript, speed=4.0)
    assert out.exists() and out.stat().st_size > 1000
    img = PIL.open(out)
    assert img.size == (w, h)
    assert getattr(img, "n_frames", 1) == n_frames


def test_gif_cli_roundtrip(tmp_path):
    """End-to-end: run -> episode json -> gif subcommand."""
    from blockblast.cli import main
    from blockblast.harness import run_eval
    from blockblast.report import write_run

    cfg = EvalConfig(episodes=1, seed_base=2024, results_dir=tmp_path)
    run = run_eval(lambda: HeuristicAgent(), cfg, agent_name="heuristic")
    run_dir = write_run(run, cfg.results_dir)
    ep_json = run_dir / "episodes" / "seed_2024.json"
    out_gif = tmp_path / "demo.gif"
    assert main(["gif", str(ep_json), "-o", str(out_gif), "--speed", "2"]) == 0
    assert out_gif.exists()
    img = PIL.open(out_gif)
    assert getattr(img, "n_frames", 1) > 10


def test_extreme_delay_is_clamped_not_crashing(tmp_path):
    """GIF stores delays in 16 bits; overflow must clamp, not raise struct.error."""
    from blockblast.gifenc import MAX_DELAY_CS, write_gif

    out = tmp_path / "slow.gif"
    # a delay far beyond the 16-bit field, plus a zero/negative one
    write_gif(out, (2, 2), [(0, 0, 0), (255, 255, 255)],
              [[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]],
              [999_999, 0, -5])
    assert out.stat().st_size > 0

    raw = out.read_bytes()
    assert raw.startswith(b"GIF89a")
    # every graphic-control extension must carry a representable delay
    assert MAX_DELAY_CS == 0xFFFF
    with PIL.open(out) as im:  # decoded delays must stay representable
        delays = []
        for frame in range(im.n_frames):
            im.seek(frame)
            delays.append(im.info.get("duration", 0))
        assert all(0 <= d <= MAX_DELAY_CS * 10 for d in delays)


def test_gif_cli_rejects_absurd_speed(tmp_path, capsys):
    """--speed 0.0001 used to surface a raw struct.error from the encoder."""
    from blockblast.cli import main

    ep = tmp_path / "ep.json"
    ep.write_text(json.dumps({
        "agent": "greedy", "seed": 1, "score": 0, "transcript": [],
    }), encoding="utf-8")

    assert main(["gif", str(ep), "-o", str(tmp_path / "x.gif"), "--speed", "0.0001"]) == 2
    assert "--speed must be between" in capsys.readouterr().err
