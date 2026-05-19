#!/usr/bin/env python3
"""Build HotpotQA target role labels with LLM calls in parallel.

This is a narrow utility for auditing/rebuilding T_q_raw role labels from an
external query subset. It writes only targets and llm_role_cache under the
chosen output root; it does not touch trajectories/samples/debug.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rebuild_hotpotqa_targets_v2 import (
    build_parent_chunk_id,
    classify_role_with_deepseek,
    find_sentence_text,
    normalize_supporting_facts,
    read_jsonl,
    write_jsonl,
)


SPLITS = ["train", "val", "test"]


def load_queries(query_dir: Path, split: str) -> Dict[str, dict]:
    candidates = [
        query_dir / f"{split}.jsonl",
        query_dir / f"{split}.full.bak.jsonl",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"找不到 query 文件: tried={candidates}")
    rows: Dict[str, dict] = {}
    for row in read_jsonl(path):
        qid = str(row.get("qid", "")).strip()
        if not qid:
            raise ValueError(f"query 缺少 qid: file={path}")
        rows[qid] = row
    return rows


def load_processed(processed_path: Path, allowed_qids: set[str]) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    for row in read_jsonl(processed_path):
        qid = str(row.get("qid", "")).strip()
        if qid in allowed_qids:
            rows[qid] = row
    missing = allowed_qids - set(rows)
    if missing:
        sample = sorted(missing)[:5]
        raise RuntimeError(f"processed 缺少 query qid: count={len(missing)}, sample={sample}")
    return rows


def collect_target_jobs(sample: dict) -> List[dict]:
    required = ["qid", "question", "supporting_facts", "context"]
    for field in required:
        if field not in sample:
            raise ValueError(f"processed 样本缺少字段: field={field}")

    qid = str(sample["qid"])
    question = str(sample["question"]).strip()
    answer = str(sample.get("answer", "")).strip()
    supporting_pairs = normalize_supporting_facts(sample["supporting_facts"], qid=qid)

    chunk_supports: Dict[str, dict] = {}
    for doc_id, sent_id in supporting_pairs:
        chunk_id = build_parent_chunk_id(qid, doc_id)
        text = find_sentence_text(sample["context"], title=doc_id, sent_id=sent_id, qid=qid)
        item = chunk_supports.setdefault(chunk_id, {"doc_id": doc_id, "texts": []})
        item["texts"].append(text)

    jobs = []
    for chunk_id, item in chunk_supports.items():
        text = " ".join(x.strip() for x in item["texts"] if str(x).strip())
        jobs.append(
            {
                "qid": qid,
                "question": question,
                "answer": answer,
                "unit_id": chunk_id,
                "chunk_id": chunk_id,
                "doc_id": item["doc_id"],
                "text": text,
            }
        )
    if not jobs:
        raise ValueError(f"T_q_raw 为空: qid={qid}")
    return jobs


def label_one(job: dict, *, api_key: str, base_url: str, model: str, cache_dir: Path) -> dict:
    role, source = classify_role_with_deepseek(
        api_key=api_key,
        base_url=base_url,
        model=model,
        cache_dir=cache_dir,
        qid=job["qid"],
        question=job["question"],
        answer=job["answer"],
        unit_id=job["unit_id"],
        text=job["text"],
        doc_id=job["doc_id"],
    )
    return {
        "qid": job["qid"],
        "target": {
            "unit_id": job["unit_id"],
            "chunk_id": job["chunk_id"],
            "text": job["text"],
            "doc_id": job["doc_id"],
            "parent_chunk_id": job["chunk_id"],
            "span_start": None,
            "span_end": None,
            "provenance": "raw",
            "weight": 1.0,
            "primary_role": role,
            "role_label_source": source,
        },
    }


def build_split(
    split: str,
    *,
    query_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    api_key: str,
    base_url: str,
    model: str,
    workers: int,
    max_qids: int = 0,
    retry_rounds: int = 6,
) -> Tuple[int, int]:
    queries = load_queries(query_dir, split)
    if max_qids > 0:
        queries = dict(list(queries.items())[:max_qids])
    processed = load_processed(processed_dir / f"{split}.jsonl", set(queries))

    jobs = []
    qid_order = []
    for qid in queries:
        qid_order.append(qid)
        jobs.extend(collect_target_jobs(processed[qid]))

    by_qid: Dict[str, List[dict]] = {qid: [] for qid in qid_order}
    done = 0
    started = time.time()
    pending = list(jobs)
    round_idx = 0
    last_errors: List[str] = []
    while pending and round_idx < max(1, retry_rounds):
        round_idx += 1
        failed: List[dict] = []
        if round_idx > 1:
            print(
                f"[{split}] retry round {round_idx}/{retry_rounds}: pending={len(pending)}",
                flush=True,
            )
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_job = {
                ex.submit(
                    label_one,
                    job,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    cache_dir=cache_dir,
                ): job
                for job in pending
            }
            for fut in as_completed(future_to_job):
                job = future_to_job[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    failed.append(job)
                    last_errors.append(
                        f"qid={job.get('qid')} unit_id={job.get('unit_id')} error={exc}"
                    )
                    continue
                by_qid[result["qid"]].append(result["target"])
                done += 1
                if done % 100 == 0 or done == len(jobs):
                    elapsed = time.time() - started
                    print(
                        f"[{split}] labeled {done}/{len(jobs)} target units elapsed={elapsed:.1f}s",
                        flush=True,
                    )
        pending = failed
        if pending:
            time.sleep(min(30, 2 * round_idx))

    if pending:
        sample_errors = "\n".join(last_errors[-10:])
        raise RuntimeError(
            f"{split} 仍有 {len(pending)} 个 target units 在 {retry_rounds} 轮后失败。\n"
            f"最近错误:\n{sample_errors}"
        )

    records = []
    for qid in qid_order:
        target_items = by_qid[qid]
        target_items.sort(key=lambda x: x["chunk_id"])
        q = queries[qid]
        records.append(
            {
                "qid": qid,
                "question": q.get("question", processed[qid].get("question", "")),
                "T_q_raw": target_items,
            }
        )

    out_path = output_dir / f"{split}.jsonl"
    count = write_jsonl(records, out_path)
    if count != len(queries):
        raise RuntimeError(f"写入数量不一致: split={split}, queries={len(queries)}, written={count}")
    return count, len(jobs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-dir", type=Path, default=Path("/home/lmy/study/queries"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/hotpotqa_distractor/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("data/hotpotqa_distractor_v5_llm_roles"))
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("ROLE_LLM_WORKERS", "8")))
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--retry-rounds", type=int, default=6)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
    os.environ["HOTPOTQA_REQUIRE_LLM_ROLES"] = "1"
    os.environ["HOTPOTQA_ROLE_LABEL_MODE"] = "auto"

    output_root = args.output_root
    targets_dir = output_root / "targets"
    cache_dir = output_root / "llm_role_cache"
    targets_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        count, target_count = build_split(
            split,
            query_dir=args.query_dir,
            processed_dir=args.processed_dir,
            output_dir=targets_dir,
            cache_dir=cache_dir,
            api_key=api_key,
            base_url=base_url,
            model=model,
            workers=max(1, args.workers),
            max_qids=max(0, args.max_qids),
            retry_rounds=max(1, args.retry_rounds),
        )
        stats[split] = {"queries": count, "targets": target_count}

    manifest = {
        "source_query_dir": str(args.query_dir),
        "processed_dir": str(args.processed_dir),
        "model": model,
        "base_url": base_url,
        "workers": args.workers,
        "stats": stats,
    }
    (output_root / "build_manifest_llm_roles.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("LLM role targets built:")
    for split, item in stats.items():
        print(f"  {split}: queries={item['queries']} targets={item['targets']} -> {targets_dir / f'{split}.jsonl'}")
    print(f"  role cache: {cache_dir}")


if __name__ == "__main__":
    main()
