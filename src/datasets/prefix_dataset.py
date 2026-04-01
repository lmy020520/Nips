import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

from torch.utils.data import Dataset


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


class PrefixDataset(Dataset):
    """
    只读取 samples/{train,val,test}.jsonl，
    输出 ranker 当前阶段需要的最小字段：

    {
      "qid": ...,
      "question": ...,
      "K_t": ...,
      "candidates": ...,
      "positive_unit_id": ...
    }
    """

    REQUIRED_FIELDS = [
        "qid",
        "question",
        "K_t",
        "candidates",
        "positive_unit_id",
    ]

    def __init__(self, samples_path: str):
        self.samples_path = Path(samples_path)
        if not self.samples_path.exists():
            raise FileNotFoundError(f"找不到 samples 文件: {self.samples_path}")

        self.samples: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        for row_idx, record in enumerate(read_jsonl(self.samples_path), start=1):
            for field in self.REQUIRED_FIELDS:
                if field not in record:
                    raise ValueError(
                        f"samples 缺少字段: file={self.samples_path}, row={row_idx}, field={field}"
                    )

            qid = str(record["qid"])
            question = str(record["question"]).strip()
            k_t = str(record["K_t"])
            candidates = record["candidates"]
            positive_unit_id = str(record["positive_unit_id"]).strip()

            if not question:
                raise ValueError(f"question 为空: file={self.samples_path}, row={row_idx}, qid={qid}")

            if not isinstance(candidates, list) or len(candidates) == 0:
                raise ValueError(
                    f"candidates 必须是非空 list: file={self.samples_path}, row={row_idx}, qid={qid}"
                )

            normalized_candidates = []
            seen = set()
            for unit_id in candidates:
                unit_id = str(unit_id)
                if unit_id in seen:
                    raise ValueError(
                        f"candidates 中存在重复 unit_id: file={self.samples_path}, row={row_idx}, qid={qid}, unit_id={unit_id}"
                    )
                seen.add(unit_id)
                normalized_candidates.append(unit_id)

            if positive_unit_id not in normalized_candidates:
                raise ValueError(
                    f"positive_unit_id 不在 candidates 中: file={self.samples_path}, row={row_idx}, qid={qid}, positive_unit_id={positive_unit_id}"
                )

            self.samples.append(
                {
                    "qid": qid,
                    "question": question,
                    "K_t": k_t,
                    "candidates": normalized_candidates,
                    "positive_unit_id": positive_unit_id,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def prefix_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {
        "qid": [item["qid"] for item in batch],
        "question": [item["question"] for item in batch],
        "K_t": [item["K_t"] for item in batch],
        "candidates": [item["candidates"] for item in batch],
        "positive_unit_id": [item["positive_unit_id"] for item in batch],
    }