#!/usr/bin/env python3
"""Tiny local DeepSeek proxy for WSL environments where Windows curl has network access.

The project builders speak OpenAI-style HTTP to DEEPSEEK_BASE_URL. In some WSL
setups, DNS resolves to a fake-ip range that Python cannot route, while
Windows curl can still reach the endpoint through the desktop proxy/TUN. This
proxy keeps the builder unchanged: point DEEPSEEK_BASE_URL at localhost and it
forwards POST bodies to DeepSeek using cmd.exe curl.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "DeepSeekWindowsCurlProxy/0.1"

    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok\n")

    def do_POST(self) -> None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            self.send_error(500, "DEEPSEEK_API_KEY is not set")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        target = "https://api.deepseek.com" + self.path

        cmd = [
            "cmd.exe",
            "/c",
            "curl",
            "-sS",
            "-m",
            str(getattr(self.server, "upstream_timeout", 120)),
            target,
            "-H",
            f"Authorization: Bearer {key}",
            "-H",
            "Content-Type: application/json",
            "-d",
            "@-",
        ]
        try:
            result = subprocess.run(
                cmd,
                input=body,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=getattr(self.server, "subprocess_timeout", 150),
            )
        except subprocess.TimeoutExpired:
            self.send_error(504, "upstream timeout")
            return

        if result.returncode != 0:
            detail = result.stderr.decode("gbk", errors="replace")[:1000]
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write((f'{{"error":"upstream curl failed","detail":{detail!r}}}').encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(result.stdout)

    def log_message(self, fmt: str, *args: object) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--upstream-timeout", type=int, default=120)
    parser.add_argument("--subprocess-timeout", type=int, default=150)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    server.upstream_timeout = args.upstream_timeout
    server.subprocess_timeout = args.subprocess_timeout
    server.quiet = args.quiet
    print(f"DeepSeek Windows curl proxy listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
