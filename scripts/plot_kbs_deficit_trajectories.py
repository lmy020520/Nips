#!/usr/bin/env python3
"""Plot predicted/teacher typed-deficit trajectories for successful and failed qids."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROLES = ("d_br", "d_dis", "d_sup", "d_der")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.records))
    by_qid = defaultdict(list)
    for record in records:
        by_qid[str(record.get("qid") or "")].append(record)

    status = {}
    for qid, items in by_qid.items():
        correctness = [item.get("step_correct") for item in items]
        status[qid] = "success" if correctness and all(value is True for value in correctness) else "failure"

    buckets = defaultdict(lambda: defaultdict(list))
    for record in records:
        group = status[str(record.get("qid") or "")]
        t = int(record.get("t", 0))
        buckets[(group, t)]["pred_mean"].append(float(record["pred_mean"]))
        buckets[(group, t)]["teacher_mean"].append(float(record["teacher_mean"]))
        for role in ROLES:
            role_error = (record.get("role_errors") or {}).get(role) or {}
            if role_error.get("pred") is not None:
                buckets[(group, t)][f"pred_{role}"].append(float(role_error["pred"]))
            if role_error.get("teacher") is not None:
                buckets[(group, t)][f"teacher_{role}"].append(float(role_error["teacher"]))

    rows = []
    for (group, t), values in sorted(buckets.items()):
        row = {"group": group, "t": t, "count": len(values["pred_mean"])}
        for key, items in sorted(values.items()):
            row[key] = round(mean(items), 6) if items else None
        rows.append(row)

    summary = {
        "records": len(records),
        "qids": len(by_qid),
        "trajectory_success_qids": sum(value == "success" for value in status.values()),
        "trajectory_failure_qids": sum(value == "failure" for value in status.values()),
        "success_definition": "All teacher-state evidence-selection steps have positive_rank=1.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.tsv_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"group", "t", "count"}, key))
    with args.tsv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # Dependency-free SVG keeps the analysis runnable on offline servers.
    width, height = 1000, 420
    panel_width, top, bottom = 430, 55, 350
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.pred{stroke:#c84b31;fill:none;stroke-width:3}.teacher{stroke:#26738f;fill:none;stroke-width:3}</style>',
    ]
    for panel, group in enumerate(("success", "failure")):
        left = 65 + panel * 500
        right = left + panel_width
        group_rows = sorted((row for row in rows if row["group"] == group), key=lambda row: row["t"])
        max_t = max((row["t"] for row in group_rows), default=1)
        max_t = max(1, max_t)
        for tick in range(6):
            value = tick / 5
            y = bottom - value * (bottom - top)
            svg.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
            if panel == 0:
                svg.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="13">{value:.1f}</text>')
        svg.extend(
            [
                f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
                f'<text x="{(left+right)/2:.1f}" y="30" text-anchor="middle" font-size="19" font-weight="bold">{group.title()} trajectories</text>',
                f'<text x="{(left+right)/2:.1f}" y="395" text-anchor="middle" font-size="15">Evidence acquisition step t</text>',
            ]
        )
        for t in range(max_t + 1):
            x = left + t / max_t * panel_width
            svg.append(f'<text x="{x:.1f}" y="373" text-anchor="middle" font-size="13">{t}</text>')

        for key, css_class in (("pred_mean", "pred"), ("teacher_mean", "teacher")):
            points = []
            for row in group_rows:
                value = row.get(key)
                if value is None:
                    continue
                x = left + row["t"] / max_t * panel_width
                y = bottom - max(0.0, min(1.0, float(value))) * (bottom - top)
                points.append((x, y))
            if points:
                coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                svg.append(f'<polyline class="{css_class}" points="{coordinates}"/>')
                for x, y in points:
                    svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="{css_class}" fill="white"/>')
    svg.extend(
        [
            '<text x="15" y="210" text-anchor="middle" font-size="15" transform="rotate(-90 15 210)">Mean typed deficit</text>',
            '<line class="pred" x1="390" y1="407" x2="425" y2="407"/><text x="435" y="412" font-size="14">Predicted</text>',
            '<line class="teacher" x1="535" y1="407" x2="570" y2="407"/><text x="580" y="412" font-size="14">Teacher</text>',
            '</svg>',
        ]
    )
    args.figure_output.parent.mkdir(parents=True, exist_ok=True)
    args.figure_output.write_text("\n".join(svg), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"figure: {args.figure_output}")


if __name__ == "__main__":
    main()
