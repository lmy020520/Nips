#!/usr/bin/env python3
"""Fail-fast DeepSeek connectivity and authentication preflight."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing or blank")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly OK.",
            }
        ],
        "temperature": 0.0,
        "max_tokens": 8,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"DeepSeek preflight HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, ConnectionError, urllib.error.URLError) as exc:
        raise RuntimeError(f"DeepSeek preflight connection failed: {exc}") from exc

    choices = payload.get("choices") if isinstance(payload, dict) else None
    content = ""
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = str(message.get("content") or "").strip()
    if not content:
        raise RuntimeError("DeepSeek preflight returned an empty response")

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    print(
        json.dumps(
            {
                "status": "PASS",
                "model": model,
                "latency_seconds": round(time.time() - started, 3),
                "total_tokens": int(usage.get("total_tokens") or 0),
                "response_nonempty": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
