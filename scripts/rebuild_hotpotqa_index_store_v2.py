import os
import json
from pathlib import Path
from typing import Iterable, Dict, List, Tuple


SPLITS = ["train", "val", "test"]

DEFAULT_INPUT_BASE = "data/hotpotqa_distractor/processed"
DEFAULT_OUTPUT_BASE = os.environ.get("HOTPOTQA_INDEX_STORE_BASE", os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v2") + "/index_store")
DEFAULT_QUERY_BASE = os.environ.get("HOTPOTQA_QUERY_BASE", os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v2") + "/queries")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSONL 解析失败: file={path}, line={line_idx}, error={e}"
                ) from e


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_allowed_qids(path: Path) -> set:
    if not path.exists():
        raise FileNotFoundError(f"找不到 queries 文件: {path}")

    allowed = set()
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record:
            raise ValueError(f"queries 缺少 qid: file={path}, row={row_idx}")
        allowed.add(str(record["qid"]))

    if not allowed:
        raise ValueError(f"queries 为空: {path}")
    return allowed


def build_chunk_id(qid: str, doc_id: str) -> str:
    return f"{qid}::{doc_id}"


def build_atom_id(qid: str, doc_id: str, sent_id: int) -> str:
    return f"{qid}::{doc_id}::{sent_id}"


def normalize_sentences(sentences: list, qid: str, doc_id: str) -> List[Tuple[int, str]]:
    if not isinstance(sentences, list):
        raise ValueError(f"sentences 必须是 list: qid={qid}, doc_id={doc_id}")

    kept = []
    for sent_id, text in enumerate(sentences):
        if not isinstance(text, str):
            raise ValueError(
                f"sentence 必须是 str: qid={qid}, doc_id={doc_id}, sent_id={sent_id}"
            )
        text = text.strip()
        if not text:
            continue
        kept.append((sent_id, text))
    return kept


def build_chunk_text(sentence_pairs: List[Tuple[int, str]]) -> str:
    return " ".join(text for _, text in sentence_pairs).strip()


def build_summary_text(sentence_pairs: List[Tuple[int, str]], max_sentences: int = 2, max_chars: int = 200) -> str:
    """
    当前先用轻量规则构造 summary view：
    - 取前 1~2 句
    - 最长截到 max_chars
    """
    summary = " ".join(text for _, text in sentence_pairs[:max_sentences]).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip()
    return summary


def build_records_from_sample(sample: dict) -> Tuple[List[dict], List[dict]]:
    required_fields = ["qid", "context"]
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"processed 样本缺少字段: {field}")

    qid = str(sample["qid"])
    context = sample["context"]

    if not isinstance(context, list):
        raise ValueError(f"context 必须是 list: qid={qid}")

    chunks = []
    atoms = []

    seen_chunk_ids = set()
    seen_atom_ids = set()

    for block_idx, block in enumerate(context):
        if not isinstance(block, dict):
            raise ValueError(f"context[{block_idx}] 必须是 dict: qid={qid}")

        if "title" not in block or "sentences" not in block:
            raise ValueError(f"context[{block_idx}] 缺少 title 或 sentences: qid={qid}")

        doc_id = str(block["title"])
        sentence_pairs = normalize_sentences(block["sentences"], qid=qid, doc_id=doc_id)

        if not sentence_pairs:
            continue

        chunk_id = build_chunk_id(qid, doc_id)
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"发现重复 chunk_id: qid={qid}, chunk_id={chunk_id}")
        seen_chunk_ids.add(chunk_id)

        chunk_text = build_chunk_text(sentence_pairs)
        summary_text = build_summary_text(sentence_pairs)

        if not chunk_text:
            raise ValueError(f"chunk_text 为空: qid={qid}, chunk_id={chunk_id}")
        if not summary_text:
            raise ValueError(f"summary_text 为空: qid={qid}, chunk_id={chunk_id}")

        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_text": chunk_text,
                "summary_text": summary_text,
            }
        )

        for sent_id, text in sentence_pairs:
            atom_id = build_atom_id(qid, doc_id, sent_id)
            if atom_id in seen_atom_ids:
                raise ValueError(f"发现重复 atom_id: qid={qid}, atom_id={atom_id}")
            seen_atom_ids.add(atom_id)

            atoms.append(
                {
                    "atom_id": atom_id,
                    "parent_chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "atom_text": text,
                    "span_start": None,
                    "span_end": None,
                }
            )

    if not chunks:
        raise ValueError(f"该样本没有可写入的 chunks: qid={qid}")
    if not atoms:
        raise ValueError(f"该样本没有可写入的 atoms: qid={qid}")

    return chunks, atoms


def convert_split(input_path: Path, query_path: Path, chunks_output_path: Path, atoms_output_path: Path) -> Tuple[int, int]:
    seen_global_chunk_ids = set()
    seen_global_atom_ids = set()
    allowed_qids = load_allowed_qids(query_path)
    kept_qids = set()

    def chunk_generator():
        for row_idx, sample in enumerate(read_jsonl(input_path), start=1):
            qid = str(sample.get("qid", ""))
            if qid not in allowed_qids:
                continue

            kept_qids.add(qid)

            try:
                chunks, _ = build_records_from_sample(sample)
            except Exception as e:
                raise ValueError(
                    f"构建 chunks 失败: file={input_path}, row={row_idx}, qid={sample.get('qid', 'UNKNOWN')}, error={e}"
                ) from e

            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                if chunk_id in seen_global_chunk_ids:
                    raise ValueError(f"split 内出现重复 chunk_id: file={input_path}, chunk_id={chunk_id}")
                seen_global_chunk_ids.add(chunk_id)
                yield chunk

    def atom_generator():
        for row_idx, sample in enumerate(read_jsonl(input_path), start=1):
            qid = str(sample.get("qid", ""))
            if qid not in allowed_qids:
                continue

            kept_qids.add(qid)

            try:
                _, atoms = build_records_from_sample(sample)
            except Exception as e:
                raise ValueError(
                    f"构建 atoms 失败: file={input_path}, row={row_idx}, qid={sample.get('qid', 'UNKNOWN')}, error={e}"
                ) from e

            for atom in atoms:
                atom_id = atom["atom_id"]
                if atom_id in seen_global_atom_ids:
                    raise ValueError(f"split 内出现重复 atom_id: file={input_path}, atom_id={atom_id}")
                seen_global_atom_ids.add(atom_id)
                yield atom

    chunk_count = write_jsonl(chunk_generator(), chunks_output_path)
    atom_count = write_jsonl(atom_generator(), atoms_output_path)

    if kept_qids != allowed_qids:
        missing = sorted(allowed_qids - kept_qids)
        raise RuntimeError(
            f"index_store 未覆盖全部 queries qid: "
            f"need={len(allowed_qids)}, got={len(kept_qids)}, missing={missing[:5]}"
        )

    return chunk_count, atom_count


def main():
    project_root = Path(__file__).resolve().parent.parent

    input_base = project_root / DEFAULT_INPUT_BASE
    output_base = project_root / DEFAULT_OUTPUT_BASE
    query_base = project_root / DEFAULT_QUERY_BASE
    output_base.mkdir(parents=True, exist_ok=True)

    chunk_name_map = {
        "train": "chunks_train.jsonl",
        "val": "chunks_val.jsonl",
        "test": "chunks_test.jsonl",
    }
    atom_name_map = {
        "train": "atoms_train.jsonl",
        "val": "atoms_val.jsonl",
        "test": "atoms_test.jsonl",
    }

    stats: Dict[str, Dict[str, int]] = {}

    for split in SPLITS:
        input_path = input_base / f"{split}.jsonl"
        query_path = query_base / f"{split}.jsonl"
        chunks_output_path = output_base / chunk_name_map[split]
        atoms_output_path = output_base / atom_name_map[split]

        if not input_path.exists():
            raise FileNotFoundError(f"找不到 processed 文件: {input_path}")

        chunk_count, atom_count = convert_split(
            input_path=input_path,
            query_path=query_path,
            chunks_output_path=chunks_output_path,
            atoms_output_path=atoms_output_path,
        )
        stats[split] = {
            "chunks": chunk_count,
            "atoms": atom_count,
        }

    print("index_store v2 重建完成：")
    for split in SPLITS:
        print(
            f"  {split}: "
            f"chunks={stats[split]['chunks']} -> {output_base / chunk_name_map[split]}, "
            f"atoms={stats[split]['atoms']} -> {output_base / atom_name_map[split]}"
        )


if __name__ == "__main__":
    main()
