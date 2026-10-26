"""Self-contained HTML report generator for blockblast-bench.

Reads ``results/runs.jsonl`` and the referenced run directories, then writes
ONE HTML file (no external assets, no JavaScript dependencies) containing:

- the leaderboard table,
- per-run cards with aggregate metrics and a per-episode score chart
  (pure inline SVG),
- GIF replays shipped in the repo.

Usage: ``python3 -m blockblast report --output docs/index.html``
"""

from __future__ import annotations

import datetime
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .report import load_runs


def _e(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _fmt(x: Any, nd: int = 1) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return f"{v:.{nd}f}" if nd else str(int(round(v)))


def _seeds_of(run: Dict[str, Any]) -> List[int]:
    base = run.get("seed_base")
    n = run.get("episodes") or run.get("aggregates", {}).get("episodes") or 0
    if base is None or not n:
        return []
    return [base + i for i in range(int(n))]


def _episode_scores(run_dir: Path, seeds: List[int]) -> List[int]:
    scores: List[int] = []
    for seed in seeds:
        ep = run_dir / "episodes" / f"seed_{seed}.json"
        if ep.exists():
            try:
                scores.append(int(json.loads(ep.read_text(encoding="utf-8"))["score"]))
            except Exception:
                pass
    return scores


def _bar_chart_svg(scores: List[int], width: int = 460, height: int = 90) -> str:
    """Tiny pure-SVG bar chart of per-episode scores."""
    if not scores:
        return ""
    n = len(scores)
    top = max(max(scores), 1)
    pad = 2
    bw = max(2.0, (width - pad * (n + 1)) / n)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">']
    for i, s in enumerate(scores):
        h = max(1.0, (s / top) * (height - 18))
        x = pad + i * (bw + pad)
        y = height - 14 - h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="1.5" fill="#4c8dff"/>'
        )
        parts.append(
            f'<text x="{x + bw / 2:.1f}" y="{height - 3}" font-size="8" text-anchor="middle" fill="#8892a4">{s}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _run_card(run: Dict[str, Any], results_dir: Path, rank: int) -> str:
    run_id = run.get("run_id", "?")
    agg = run.get("aggregates", {})
    run_dir = results_dir / run_id
    scores = _episode_scores(run_dir, _seeds_of(run))
    agent = run.get("agent", "?")
    model = run.get("model")
    tag = _e(agent) + (f" · {_e(model)}" if model else "")

    eff = ""
    analysis_path = run_dir / "analysis.json"
    if analysis_path.exists():
        try:
            eff_val = json.loads(analysis_path.read_text(encoding="utf-8"))["aggregate"]["greedy_efficiency"]
            eff = f'<div class="kpi"><span>{eff_val:.3f}</span>决策效率</div>'
        except Exception:
            pass

    chart = _bar_chart_svg(scores)
    tok = agg.get("tokens_per_move") or 0
    tok_html = f'<div class="kpi"><span>{_fmt(tok)}</span>tok/步</div>' if tok else ""
    completed = agg.get("valid_completion_rate")
    seeds = _seeds_of(run)

    return f"""
<section class="card">
  <h3>#{rank} {tag}</h3>
  <div class="kpis">
    <div class="kpi big"><span>{_fmt(agg.get("mean_score"))}</span>均分</div>
    <div class="kpi"><span>{_fmt(agg.get("max_score"), 0)}</span>最佳</div>
    <div class="kpi"><span>{_fmt(agg.get("mean_moves"))}</span>步数</div>
    <div class="kpi"><span>{_fmt(agg.get("mean_lines"))}</span>消行</div>
    <div class="kpi"><span>{_fmt(agg.get("best_combo"), 0)}</span>最高combo</div>
    {tok_html}
    {eff}
  </div>
  {chart}
  <p class="meta">run <code>{_e(run_id)}</code> · {len(seeds)} 局 · 种子 {seeds[0] if seeds else "?"}…{seeds[-1] if seeds else "?"}
   {f' · 完整终局率 <b>{completed:.0%}</b>' if completed is not None else ''}</p>
</section>"""


_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Block Blast Benchmark — 评测报告</title>
<style>
  :root {{ --fg:#1f2430; --muted:#8892a4; --line:#e3e8ef; --bg:#f6f8fb; --card:#fff; --accent:#4c8dff; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font:15px/1.6 -apple-system,"PingFang SC","Segoe UI",sans-serif; color:var(--fg); background:var(--bg); }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 40px 20px 80px; }}
  h1 {{ font-size: 26px; margin:0 0 4px; }}
  h2 {{ font-size: 19px; margin: 34px 0 12px; }}
  .sub {{ color: var(--muted); margin:0 0 28px; }}
  table {{ border-collapse: collapse; width:100%; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; font-size:14px; }}
  th, td {{ padding:8px 12px; text-align:right; border-bottom:1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ background:#eef2f8; font-weight:600; }}
  tr:last-child td {{ border-bottom:none; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px 20px; margin:16px 0; }}
  .card h3 {{ margin:0 0 10px; font-size:16px; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:10px 22px; margin-bottom:10px; }}
  .kpi {{ min-width:64px; font-size:12px; color:var(--muted); }}
  .kpi span {{ display:block; font-size:20px; font-weight:700; color:var(--accent); }}
  .kpi.big span {{ font-size:28px; }}
  .meta {{ color:var(--muted); font-size:12.5px; margin:8px 0 0; }}
  code {{ background:#eef2f8; padding:1px 6px; border-radius:4px; font-size:12px; }}
  .gifs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }}
  .gif {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; text-align:center; }}
  .gif img {{ max-width:100%; height:auto; border-radius:6px; }}
  .gif p {{ margin:8px 0 0; font-size:13px; color:var(--muted); }}
  footer {{ margin-top:40px; color:var(--muted); font-size:13px; }}
  a {{ color: var(--accent); text-decoration:none; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Block Blast Benchmark</h1>
  <p class="sub">确定性 8×8 方块消除环境上的 AI Agent 评测 · 同种子集公平对比 · 生成于 {generated}</p>

  <h2>排行榜</h2>
  <table>
    <tr><th>#</th><th>Agent</th><th>Model</th><th>Eps</th><th>Mean</th><th>Median</th><th>Max</th><th>Moves</th><th>Lines</th><th>Combo</th><th>Inv/mv</th><th>Tok/mv</th></tr>
    {rows}
  </table>

  <h2>各 run 画像</h2>
  {cards}

  <h2>精彩回放</h2>
  <div class="gifs">
    {gifs}
  </div>

  <footer>由 <a href="https://github.com/kittimzhe/blockblast-bench">blockblast-bench</a> 生成 ·
  完整数据见仓库 <code>results/</code> 目录 · 分数只在同种子集与同计分配置下可比</footer>
</div>
</body>
</html>
"""


def generate_report(
    results_dir: Path, output: Path, gifs: Optional[List[Path]] = None
) -> Path:
    """Write a single self-contained HTML report. Returns the output path.

    GIFs are copied next to the output file and referenced by name, so the
    whole output directory is relocatable (and works as a GitHub Pages root).
    """
    import shutil

    results_dir = Path(results_dir)
    output = Path(output)
    runs = sorted(
        load_runs(results_dir),
        key=lambda r: float(r.get("aggregates", {}).get("mean_score") or 0),
        reverse=True,
    )

    rows = []
    for i, r in enumerate(runs, 1):
        agg = r.get("aggregates", {})
        tok = agg.get("tokens_per_move") or 0
        rows.append(
            f"<tr><td>{i}</td><td>{_e(r.get('agent'))}</td><td>{_e(r.get('model') or '–')}</td>"
            f"<td>{agg.get('episodes', '–')}</td><td><b>{_fmt(agg.get('mean_score'))}</b></td>"
            f"<td>{_fmt(agg.get('median_score'))}</td><td>{_fmt(agg.get('max_score'), 0)}</td>"
            f"<td>{_fmt(agg.get('mean_moves'))}</td><td>{_fmt(agg.get('mean_lines'))}</td>"
            f"<td>{_fmt(agg.get('best_combo'), 0)}</td><td>{_fmt(agg.get('invalid_per_move'), 3)}</td>"
            f"<td>{_fmt(tok) if tok else '–'}</td></tr>"
        )

    cards = "".join(_run_card(r, results_dir, i) for i, r in enumerate(runs, 1))

    gif_html = []
    for g in gifs or []:
        g = Path(g)
        if g.exists():
            shutil.copy2(g, output.parent / g.name)
            caption = g.name
            gif_html.append(
                f'<div class="gif"><img src="{_e(g.name)}" alt="{_e(caption)}"><p>{_e(caption)}</p></div>'
            )
    if not gif_html:
        gif_html.append('<div class="gif"><p>暂无回放（运行 blockblast gif 生成）</p></div>')

    page = _PAGE.format(
        generated=datetime.date.today().isoformat(),
        rows="".join(rows) or "<tr><td colspan='12'>还没有评测数据</td></tr>",
        cards=cards or "<p>还没有评测数据</p>",
        gifs="".join(gif_html),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output
