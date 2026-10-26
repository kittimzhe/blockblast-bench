"""LLM agent: any OpenAI-compatible chat-completions endpoint plays the game.

Works with OpenAI, DeepSeek, GLM, Qwen (DashScope compatible-mode), local
vLLM/Ollama with an OpenAI-compatible server, etc. — anything that speaks
``POST {base_url}/chat/completions``.

The agent is stateless per move: each call sends the system prompt plus one
user message containing the full text observation. Retries after a rejected
move append the error as feedback in the same turn's retry call.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..env import Observation
from ..render import render_observation
from .base import Action, Agent, AgentError, AgentParseError

DEFAULT_SYSTEM_PROMPT = """\
You are playing Block Blast, an 8x8 block puzzle game.

RULES
- The board is an 8x8 grid. '.' = empty, 'X' = occupied.
- Your tray holds up to 3 pieces; you place ONE piece per turn.
- Pieces are given exactly as drawn: they can NOT be rotated or flipped.
- When all tray pieces are used, the tray refills with 3 new pieces.
- A completely filled row or column is cleared (several can clear at once).
- Consecutive clearing turns build a combo multiplier for higher scores.
- The game ends when none of the remaining tray pieces fits anywhere.
- Goal: maximise total score. Clear lines often, keep the board open and
  connected, and avoid leaving scattered small holes.

COORDINATES
- Rows are numbered 0-7 top to bottom, columns 0-7 left to right.
- An action places the chosen piece so that the top-left cell of its
  bounding box lands at (row, col). Use the indices printed around the
  board and the mini-grid previews in the tray.

OUTPUT
- Reply with ONLY a JSON object, no extra text:
  {"piece": <tray index 0-2>, "row": <0-7>, "col": <0-7>}"""

RETRY_SUFFIX = (
    "\n\nRETRY: {feedback}\n"
    "Answer again. Choose a piece still present in the tray and a legal "
    "(row, col) for its top-left cell. Reply with ONLY the JSON object."
)


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    timeout: float = 60.0
    #: hard ceiling the per-request timeout may grow to after read timeouts
    timeout_cap: float = 600.0
    max_network_attempts: int = 6
    #: exponential backoff between network retries: base * 2^attempt, capped
    backoff_base: float = 2.0
    backoff_cap: float = 60.0
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)


#: HTTP statuses that indicate a transient server-side condition worth retrying.
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


class TransientEndpointError(Exception):
    """A 200 response whose body is not a usable completion (gateway hiccup)."""


def _post_chat(
    cfg: LLMConfig, api_key: str, messages: List[Dict[str, str]], timeout: Optional[float] = None
) -> Tuple[str, Dict[str, Any]]:
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
    }
    if cfg.max_tokens is not None:
        payload["max_tokens"] = cfg.max_tokens
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        **(cfg.extra_headers or {}),
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout or cfg.timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise TransientEndpointError(f"endpoint returned non-JSON body: {raw[:200]!r}")
    choices = body.get("choices") or []
    if not choices:
        raise TransientEndpointError(f"endpoint returned no choices: {str(body)[:200]}")
    content = (choices[0].get("message") or {}).get("content") or ""
    usage = body.get("usage") or {}
    return content, usage


def _is_timeout(exc: Exception) -> bool:
    """Whether a network error is (or wraps) a read/connect timeout."""
    if isinstance(exc, TimeoutError):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, TimeoutError):
        return True
    return "timed out" in str(exc).lower()


def parse_action(text: str) -> Action:
    """Best-effort extraction of {"piece", "row", "col"} from a model reply."""
    t = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    if fenced:
        t = fenced.group(1)
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        t = t[start : end + 1]
    try:
        obj = json.loads(t)
    except Exception:
        raise AgentParseError(f"cannot parse JSON from reply: {text[:200]!r}")
    try:
        return Action(int(obj["piece"]), int(obj["row"]), int(obj["col"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise AgentParseError(f"reply has missing/invalid fields ({exc}): {text[:200]!r}")


class LLMAgent(Agent):
    """One LLM endpoint as a game player."""

    def __init__(self, config: LLMConfig, record_dialog: bool = False) -> None:
        self.config = config
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"missing API key: set the {config.api_key_env} environment variable "
                f"(LLMConfig.api_key_env) before running the llm agent"
            )
        self._api_key = api_key
        self.name = f"llm:{config.model}"
        self._record_dialog = record_dialog
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._stats: Dict[str, Any] = {
            "requests": 0,
            "network_errors": 0,
            "parse_errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_s": 0.0,
        }
        self._dialog: List[Dict[str, Any]] = []

    def reset(self, seed: int) -> None:
        self._reset_stats()

    def act(self, obs: Observation, feedback: Optional[str] = None) -> Action:
        user = render_observation(obs)
        if feedback:
            user += RETRY_SUFFIX.format(feedback=feedback)
        messages = [
            {"role": "system", "content": self.config.system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

        last_error: Optional[Exception] = None
        content = ""
        timeout = self.config.timeout
        attempts = max(1, self.config.max_network_attempts)
        for attempt in range(attempts):
            try:
                t0 = time.monotonic()
                content, usage = _post_chat(self.config, self._api_key, messages, timeout=timeout)
                self._stats["requests"] += 1
                self._stats["latency_s"] += time.monotonic() - t0
                self._stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                self._stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                self._stats["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
                last_error = None
                break
            except urllib.error.HTTPError as exc:
                if exc.code in RETRYABLE_HTTP:  # transient server-side condition
                    self._stats["network_errors"] += 1
                    last_error = exc
                else:  # non-retryable client error — fail the episode loudly
                    detail = ""
                    try:
                        detail = exc.read().decode("utf-8", "replace")[:200]
                    except Exception:
                        pass
                    raise AgentError(f"HTTP {exc.code} from endpoint: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self._stats["network_errors"] += 1
                last_error = exc
                # a read timeout suggests the model needs longer: grow the budget
                if _is_timeout(exc):
                    timeout = min(timeout * 2, self.config.timeout_cap)
            except TransientEndpointError as exc:  # 200 with a broken body
                self._stats["network_errors"] += 1
                last_error = exc
            if attempt + 1 < attempts:
                backoff = min(
                    self.config.backoff_base * (2 ** attempt), self.config.backoff_cap
                )
                time.sleep(backoff)
        if last_error is not None:
            raise AgentError(
                f"network failure after {attempts} attempts: {last_error}"
            )

        try:
            action = parse_action(content)
        except AgentParseError:
            self._stats["parse_errors"] += 1
            if self._record_dialog:
                self._dialog.append({"messages": messages, "reply": content, "ok": False})
            raise
        if self._record_dialog:
            self._dialog.append(
                {
                    "messages": messages,
                    "reply": content,
                    "action": action.as_tuple(),
                    "ok": True,
                }
            )
        return action

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    @property
    def dialog(self) -> List[Dict[str, Any]]:
        return self._dialog
