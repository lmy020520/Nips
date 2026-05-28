import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


SEED = 42
TRAIN_SIZE = 5000
VAL_SIZE = 500
TEST_SIZE = 500

KEEP_FIELDS = [
    "question",
    "answer",
    "type",
    "level",
    "supporting_facts",
    "context",
]


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_sample_id(sample: dict) -> str:
    if "id" in sample:
        return str(sample["id"])
    if "_id" in sample:
        return str(sample["_id"])
    raise ValueError("样本中既没有 id 也没有 _id，无法生成 qid。")


def stable_score(sample_id: str, seed: int) -> str:
    key = f"{seed}::{sample_id}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def sort_stably(samples, seed: int):
    return sorted(samples, key=lambda x: stable_score(get_sample_id(x), seed))


def project_fields(sample: dict):
    result = {
        "qid": get_sample_id(sample)
    }
    for k in KEEP_FIELDS:
        if k not in sample:
            raise ValueError(f"样本缺少字段: {k}")
        result[k] = sample[k]
    return result


def main():
    parser = argparse.ArgumentParser(description="Download and split HotpotQA distractor.")
    parser.add_argument("--out-base", type=Path, default=None)
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=VAL_SIZE)
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    out_base = args.out_base or (project_root / "data" / "hotpotqa_distractor")
    if not out_base.is_absolute():
        out_base = project_root / out_base
    out_train_path = out_base / "raw" / "train.json"
    out_val_path = out_base / "raw" / "val.json"
    out_test_path = out_base / "raw" / "test.json"
    out_meta_path = out_base / "splits" / "split_meta.json"
    cache_dir = out_base / "cache"

    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "raw").mkdir(parents=True, exist_ok=True)
    (out_base / "splits").mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("正在从 Hugging Face 下载 HotpotQA distractor...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")

    raw_train = [dict(x) for x in ds["train"]]
    raw_valid = [dict(x) for x in ds["validation"]]

    if len(raw_train) < args.train_size + args.val_size:
        raise ValueError(
            f"train split 样本不足，需要至少 {args.train_size + args.val_size} 条，实际只有 {len(raw_train)} 条"
        )
    if len(raw_valid) < args.test_size:
        raise ValueError(
            f"validation split 样本不足，需要至少 {args.test_size} 条，实际只有 {len(raw_valid)} 条"
        )

    sorted_train = sort_stably(raw_train, args.seed)
    sorted_valid = sort_stably(raw_valid, args.seed)

    train_raw = sorted_train[:args.train_size]
    val_raw = sorted_train[args.train_size:args.train_size + args.val_size]
    test_raw = sorted_valid[:args.test_size]

    train_data = [project_fields(x) for x in train_raw]
    val_data = [project_fields(x) for x in val_raw]
    test_data = [project_fields(x) for x in test_raw]

    save_json(train_data, out_train_path)
    save_json(val_data, out_val_path)
    save_json(test_data, out_test_path)

    meta = {
        "dataset_name": "HotpotQA distractor",
        "version": "v1_fixed_baseline",
        "source": {
            "provider": "Hugging Face datasets",
            "dataset_id": "hotpotqa/hotpot_qa",
            "config": "distractor"
        },
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_sizes": {
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size
        },
        "kept_fields_in_raw": ["qid"] + KEEP_FIELDS,
        "file_paths": {
            "train": str(out_train_path),
            "val": str(out_val_path),
            "test": str(out_test_path),
            "meta": str(out_meta_path)
        }
    }
    save_json(meta, out_meta_path)

    print("完成：固定版基准数据集已生成")
    print(f"train -> {out_train_path}")
    print(f"val   -> {out_val_path}")
    print(f"test  -> {out_test_path}")
    print(f"meta  -> {out_meta_path}")


if __name__ == "__main__":
    main()
