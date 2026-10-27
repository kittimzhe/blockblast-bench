"""Command line interface.

Usage (from the project root, no install needed)::

    python3 -m blockblast demo --agent heuristic --seed 42
    python3 -m blockblast run  --agent heuristic --episodes 10
    python3 -m blockblast run  --agent llm --model deepseek-chat \
        --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY
    python3 -m blockblast leaderboard
    python3 -m blockblast replay results/<run>/episodes/seed_2024.json
    python3 -m blockblast shapes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .agents import AGENT_REGISTRY, LLMAgent, LLMConfig, agent_names, make_agent
from .harness import EvalConfig, aggregates, resume_eval, run_episode, run_eval
from .pieces import SHAPE_CATALOG
from .render import render_shape
from .report import leaderboard_md, load_runs, refresh_leaderboard, write_run


def _pretty_board(rows: List[str]) -> str:
    header = "    " + " ".join(str(c) for c in range(len(rows[0])))
    body = [f"{r}   " + " ".join(row) for r, row in enumerate(rows)]
    return "\n".join([header] + body)


def _move_line(rec: dict) -> str:
    bits = [
        f"mv {rec['move']:>4}",
        f"piece [{rec['piece']}] {rec['piece_name']}",
        f"at (row={rec['row']}, col={rec['col']})",
        f"+{rec['cells']} cells",
    ]
    if rec.get("lines"):
        where = []
        if rec.get("cleared_rows"):
            where.append("rows " + ",".join(map(str, rec["cleared_rows"])))
        if rec.get("cleared_cols"):
            where.append("cols " + ",".join(map(str, rec["cleared_cols"])))
        bits.append(f"cleared {rec['lines']} ({'; '.join(where)}) combo x{rec['combo']}")
    if rec.get("perfect_clear"):
        bits.append("PERFECT CLEAR")
    bits.append(f"+{rec['points']} pts -> {rec['score']}")
    return "  ".join(bits)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_demo(args) -> int:
    cfg = EvalConfig(
        max_moves=args.max_moves,
        invalid_action_retries=args.invalid_retries,
    )
    kwargs = {}
    if args.agent in ("greedy", "heuristic"):
        kwargs["scoring"] = cfg.scoring
    agent = make_agent(args.agent, **kwargs)

    print(f"# demo: agent={agent.name} seed={args.seed}")
    print()

    def on_move(rec: dict) -> None:
        print(_move_line(rec))

    result = run_episode(agent, args.seed, cfg, on_move=on_move)

    print()
    last_board = None
    for rec in result.transcript:
        if rec.get("board_after"):
            last_board = rec["board_after"]
    if last_board:
        print("Final board:")
        print(_pretty_board(last_board))
        print()
    if result.error:
        print(f"ended with error: {result.error}")
    print(
        f"result: score={result.score} moves={result.moves} lines={result.lines_cleared} "
        f"max_combo={result.max_combo} perfect_clears={result.perfect_clears} "
        f"invalid={result.invalid_actions} reason={result.reason} time={result.duration_s}s"
    )
    return 0


def cmd_run(args) -> int:
    cfg = EvalConfig(
        episodes=args.episodes,
        seed_base=args.seed_base,
        max_moves=args.max_moves,
        invalid_action_retries=args.invalid_retries,
        results_dir=Path(args.results_dir),
        save_transcripts=not args.no_transcripts,
        save_dialog=args.save_dialog,
        concurrency=args.concurrency,
        tag=args.tag,
    )

    model: Optional[str] = None
    if args.agent == "llm":
        if not os.environ.get(args.api_key_env, "").strip():
            print(
                f"error: missing API key — set ${args.api_key_env} (or pass "
                f"--api-key-env <VAR>)",
                file=sys.stderr,
            )
            return 2
        system_prompt: Optional[str] = None
        if args.system_prompt_file:
            system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
        llm_cfg = LLMConfig(
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            timeout=args.timeout,
            max_network_attempts=args.network_retries,
            max_tokens=args.max_tokens,
            system_prompt=system_prompt,
            allow_reasoning=args.allow_reasoning,
        )

        def factory() -> LLMAgent:
            return LLMAgent(llm_cfg, record_dialog=cfg.save_dialog)

        model = args.model
    else:
        kwargs = {}
        if args.agent in ("greedy", "heuristic"):
            kwargs["scoring"] = cfg.scoring
        if args.agent == "beam":
            kwargs["scoring"] = cfg.scoring
            if getattr(args, "beam_width", None):
                kwargs["beam_width"] = args.beam_width

        def factory():
            return make_agent(args.agent, **kwargs)

    print(
        f"# run: agent={args.agent}" + (f" model={model}" if model else "")
        + f" episodes={cfg.episodes} seeds={cfg.seed_base}..{cfg.seed_base + cfg.episodes - 1}"
        + (f" tag={cfg.tag}" if cfg.tag else "")
    )
    if args.resume:
        prior = Path(args.resume)
        if not (prior / "summary.json").exists():
            print(f"error: {prior} is not a run directory (no summary.json)", file=sys.stderr)
            return 2
        print(f"# resuming unfinished episodes from {prior}")
        run = resume_eval(factory, prior, cfg, agent_name=args.agent, model=model)
    else:
        run = run_eval(factory, cfg, agent_name=args.agent, model=model)
    run_dir = write_run(run, cfg.results_dir, save_transcripts=cfg.save_transcripts)

    print(f"\n{'seed':>6} {'score':>7} {'moves':>6} {'lines':>6} {'combo':>6} {'inv':>4} {'reason':>14} {'time':>8}")
    for e in run.results:
        print(
            f"{e.seed:>6} {e.score:>7} {e.moves:>6} {e.lines_cleared:>6} "
            f"{e.max_combo:>6} {e.invalid_actions:>4} {e.reason:>14} {e.duration_s:>7.2f}s"
        )
    agg = aggregates(run)
    print()
    for key in (
        "mean_score",
        "median_score",
        "max_score",
        "mean_moves",
        "mean_lines",
        "best_combo",
        "perfect_clears",
        "invalid_per_move",
        "valid_completion_rate",
        "tokens_per_move",
        "mean_duration_s",
    ):
        print(f"  {key}: {agg[key]}")
    print(f"\nrun directory : {run_dir}")
    print(f"results dir   : {cfg.results_dir}")
    print(f"leaderboard   : {Path(cfg.results_dir) / 'leaderboard.md'}")
    return 0


def cmd_leaderboard(args) -> int:
    rows = load_runs(Path(args.results_dir), collapse_resumes=not args.all_runs)
    text = leaderboard_md(rows)
    print(text, end="")
    return 0


def cmd_replay(args) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2
    data = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"# replay: agent={data.get('agent')} seed={data.get('seed')} "
        f"score={data.get('score')} moves={data.get('moves')} reason={data.get('reason')}"
    )
    print()
    every = args.board_every
    last_board = None
    for rec in data.get("transcript", []):
        if rec.get("terminal_error"):
            print(f"!! terminal: {rec['terminal_error']}")
            continue
        print(_move_line(rec))
        if rec.get("board_after"):
            last_board = rec["board_after"]
            if every and rec["move"] % every == 0:
                print(_pretty_board(last_board))
                print()
    if last_board and not every:
        print()
        print("Final board:")
        print(_pretty_board(last_board))
    return 0


def cmd_shapes(args) -> int:
    print(f"# shape catalog ({len(SHAPE_CATALOG)} pieces)\n")
    for shape in SHAPE_CATALOG:
        print(f"{shape.name}  rows={shape.rows} cols={shape.cols} cells={shape.size}")
        print(render_shape(shape))
        print()
    return 0


def _load_transcript(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    transcript = data.get("transcript", [])
    if not transcript:
        raise SystemExit(f"error: {path} contains no transcript (ran with --no-transcripts?)")
    return data, transcript


def cmd_gif(args) -> int:
    from .animate import transcript_to_gif

    path = Path(args.path)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2
    data, transcript = _load_transcript(path)
    out = Path(args.out) if args.out else path.with_suffix(".gif")
    n_frames, (w, h) = transcript_to_gif(out, transcript, speed=args.speed)
    print(
        f"wrote {out} ({w}x{h}, {n_frames} frames) — "
        f"agent={data.get('agent')} seed={data.get('seed')} score={data.get('score')}"
    )
    return 0


def cmd_report(args) -> int:
    from .report_html import generate_report

    out = generate_report(args.results_dir, args.output, gifs=args.gif)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


def cmd_analyze(args) -> int:
    from .analysis import analyze_run, render_analysis

    run_dir = Path(args.run_dir)
    if not (run_dir / "summary.json").exists():
        print(f"error: {run_dir} is not a run directory (no summary.json)", file=sys.stderr)
        return 2
    report = analyze_run(run_dir)
    print(render_analysis(report))
    if args.json:
        out = run_dir / "analysis.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


def cmd_watch(args) -> int:
    """Animate a saved episode in the terminal (ANSI redraw)."""
    import time as _time

    path = Path(args.path)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2
    data, transcript = _load_transcript(path)
    delay = max(0.01, args.delay)
    print(
        f"# watch: agent={data.get('agent')} seed={data.get('seed')} "
        f"score={data.get('score')} moves={data.get('moves')}\n"
    )
    lines_per_frame = 8 + 1 + 2  # board + header + status lines
    first = True
    for rec in transcript:
        if rec.get("terminal_error"):
            break
        if not first:
            # move the cursor up and redraw in place
            sys.stdout.write(f"\x1b[{lines_per_frame}A")
        first = False
        print(_pretty_board(rec["board_after"]))
        status = (
            f"mv {rec['move']:>4}  {rec['piece_name']:<10} at ({rec['row']},{rec['col']})"
            f"  score {rec['score']:<7}"
        )
        if rec.get("lines"):
            status += f"  CLEAR x{rec['combo']}"
        print(f"\x1b[2K{status}")
        print("\x1b[2K", end="")
        sys.stdout.flush()
        _time.sleep(delay)
    print(f"\nfinal: score={data.get('score')} moves={data.get('moves')} reason={data.get('reason')}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-moves", type=int, default=3000, help="safety cap on moves per episode")
    p.add_argument("--invalid-retries", type=int, default=2, help="retries after an illegal/unparseable move")
    p.add_argument("--beam-width", type=int, default=40, help="beam width for the beam agent")


def _add_llm(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default="https://api.openai.com/v1", help="OpenAI-compatible base URL")
    p.add_argument("--model", default="gpt-4o-mini", help="model name")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the API key")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--network-retries", type=int, default=6,
                   help="network attempts per move (exponential backoff, growing timeout)")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--system-prompt-file", default=None, help="optional file with a custom system prompt")
    p.add_argument("--allow-reasoning", action="store_true",
                   help="invite free-form reasoning before the final JSON line (prompt variant)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blockblast",
        description="Block Blast benchmark: deterministic game env + AI-agent evaluation harness",
    )
    parser.add_argument("--version", action="version", version=f"blockblast-bench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="play one episode with live move-by-move output")
    demo.add_argument("--agent", default="heuristic", choices=agent_names())
    demo.add_argument("--seed", type=int, default=42)
    _add_common(demo)
    demo.set_defaults(func=cmd_demo)

    run = sub.add_parser("run", help="run a full evaluation")
    run.add_argument("--agent", required=True, choices=agent_names())
    run.add_argument("--episodes", type=int, default=10)
    run.add_argument("--seed-base", type=int, default=2024)
    run.add_argument("--tag", default="", help="label stored with the run")
    run.add_argument("--results-dir", default="results")
    run.add_argument("--concurrency", type=int, default=1, help="parallel episodes")
    run.add_argument("--no-transcripts", action="store_true", help="skip per-episode transcript files")
    run.add_argument("--save-dialog", action="store_true", help="store raw LLM request/reply per move")
    run.add_argument("--resume", default=None, metavar="RUN_DIR",
                     help="finish an interrupted run: keep its completed episodes, continue the rest")
    _add_common(run)
    _add_llm(run)
    run.set_defaults(func=cmd_run)

    board = sub.add_parser("leaderboard", help="print the markdown leaderboard")
    board.add_argument("--results-dir", default="results")
    board.add_argument("--all-runs", action="store_true",
                       help="also list superseded resume-chain snapshots")
    board.set_defaults(func=cmd_leaderboard)

    replay = sub.add_parser("replay", help="pretty-print a saved episode transcript")
    replay.add_argument("path", help="path to episodes/seed_N.json")
    replay.add_argument("--board-every", type=int, default=0, help="print a board snapshot every N moves")
    replay.set_defaults(func=cmd_replay)

    gif = sub.add_parser("gif", help="render a saved episode into an animated GIF")
    gif.add_argument("path", help="path to episodes/seed_N.json")
    gif.add_argument("-o", "--out", default=None, help="output .gif path (default: alongside the JSON)")
    gif.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    gif.set_defaults(func=cmd_gif)

    watch = sub.add_parser("watch", help="animate a saved episode in the terminal")
    watch.add_argument("path", help="path to episodes/seed_N.json")
    watch.add_argument("--delay", type=float, default=0.15, help="seconds per move")
    watch.set_defaults(func=cmd_watch)

    analyze = sub.add_parser("analyze", help="decision-quality analysis of a finished run")
    analyze.add_argument("run_dir", help="path to results/<run_id>")
    analyze.add_argument("--json", action="store_true", help="also write analysis.json into the run dir")
    analyze.set_defaults(func=cmd_analyze)

    report = sub.add_parser("report", help="generate a self-contained HTML report")
    report.add_argument("--results-dir", type=Path, default=Path("results"))
    report.add_argument("--output", type=Path, default=Path("docs/index.html"))
    report.add_argument("--gif", action="append", type=Path, default=None,
                        help="GIF to embed (repeatable; paths relative to repo root)")
    report.set_defaults(func=cmd_report)

    shapes = sub.add_parser("shapes", help="print the piece catalog")
    shapes.set_defaults(func=cmd_shapes)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
