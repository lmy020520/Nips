#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.datasets.prefix_dataset import (
    ROLE_TO_ID,
    PrefixRankingDataset,
    prefix_ranking_collate_fn,
)
from src.models.ranker import CrossEncoderRanker


ID_TO_ROLE = {value: key for key, value in ROLE_TO_ID.items()}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def move_to_device(tokens: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in tokens.items()}


def load_doc_role_map(path: Path) -> Dict[str, Dict[str, str]]:
    role_map: Dict[str, Dict[str, str]] = {}
    for record in read_jsonl(path):
        qid = str(record["qid"])
        qid_roles = role_map.setdefault(qid, {})
        for unit in record.get("T_q_raw") or []:
            if not isinstance(unit, dict):
                continue
            doc_id = str(unit.get("doc_id") or "").strip()
            role = str(unit.get("primary_role") or "").strip()
            if doc_id and role in ROLE_TO_ID:
                qid_roles[doc_id] = role
    return role_map


def update_metric(metrics: Dict[str, Counter], group: str, key: str, correct: bool):
    metric = metrics[group][key]
    metric["total"] += 1
    metric["correct"] += int(correct)


def finalize_metrics(metrics: Dict[str, Dict[str, Counter]]) -> dict:
    result = {}
    for group, values in metrics.items():
        result[group] = {}
        for key, metric in sorted(values.items()):
            total = metric["total"]
            correct = metric["correct"]
            result[group][key] = {
                "total": total,
                "correct": correct,
                "accuracy": round(correct / max(total, 1), 6),
            }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze HotpotQA ranker errors by trajectory buckets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-errors", type=int, default=50)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    args = build_parser().parse_args()
    config = load_yaml(Path(args.config))
    split = args.split
    data = config["data"]
    samples_path = Path(data[f"{split}_samples"])
    memory_path = Path(data[f"{split}_memory"])
    targets_path = Path(data[f"{split}_role_targets"])
    max_length = int(config["train"]["max_length"])
    pretrained_name = str(config["model"]["pretrained_name"])
    dropout = float(config["model"].get("dropout", 0.1))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    dataset = PrefixRankingDataset(
        samples_path=str(samples_path),
        memory_path=str(memory_path),
        role_targets_path=str(targets_path),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=prefix_ranking_collate_fn,
    )
    doc_role_map = load_doc_role_map(targets_path)

    model = CrossEncoderRanker(pretrained_name=pretrained_name, dropout=dropout).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    metrics = defaultdict(lambda: defaultdict(Counter))
    confusion = Counter()
    errors: List[dict] = []
    total = 0
    correct_total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"analyze-{split}"):
            tokens = tokenizer(
                batch["flat_text_a"],
                batch["flat_text_b"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = move_to_device(tokens, device)
            flat_scores, _ = model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                token_type_ids=tokens.get("token_type_ids"),
            )
            packed_scores, _ = CrossEncoderRanker.pack_scores(flat_scores, batch["candidate_counts"])
            predictions = packed_scores.argmax(dim=-1).tolist()

            for index, prediction in enumerate(predictions):
                label = int(batch["labels"][index])
                candidate_ids = batch["candidate_unit_ids"][index]
                positive_id = candidate_ids[label]
                predicted_id = candidate_ids[prediction]
                qid = batch["qids"][index]
                t = int(batch["ts"][index])
                count = int(batch["candidate_counts"][index])
                is_correct = prediction == label
                positive_role = ID_TO_ROLE.get(int(batch["positive_role_ids"][index]), "unlabeled")

                predicted_memory = dataset.memory_map.get(predicted_id, {})
                predicted_doc_id = str(predicted_memory.get("title") or "")
                predicted_role = doc_role_map.get(qid, {}).get(predicted_doc_id, "unlabeled")

                total += 1
                correct_total += int(is_correct)
                update_metric(metrics, "step", f"t={t}", is_correct)
                update_metric(metrics, "positive_role", positive_role, is_correct)
                update_metric(metrics, "candidate_count", str(count), is_correct)

                if not is_correct:
                    confusion[(positive_role, predicted_role)] += 1
                    positive_score = float(packed_scores[index, label].item())
                    predicted_score = float(packed_scores[index, prediction].item())
                    valid_scores = packed_scores[index, :count]
                    ranked_indices = valid_scores.argsort(descending=True).tolist()
                    positive_rank = ranked_indices.index(label) + 1
                    top_candidates = []
                    for rank, candidate_index in enumerate(ranked_indices[: max(args.top_candidates, 0)], start=1):
                        candidate_id = candidate_ids[candidate_index]
                        candidate_memory = dataset.memory_map.get(candidate_id, {})
                        candidate_doc_id = str(candidate_memory.get("title") or "")
                        candidate_role = doc_role_map.get(qid, {}).get(candidate_doc_id, "unlabeled")
                        top_candidates.append(
                            {
                                "rank": rank,
                                "unit_id": candidate_id,
                                "doc_id": candidate_doc_id,
                                "role": candidate_role,
                                "score": round(float(valid_scores[candidate_index].item()), 6),
                            }
                        )
                    errors.append(
                        {
                            "qid": qid,
                            "t": t,
                            "candidate_count": count,
                            "positive_unit_id": positive_id,
                            "predicted_unit_id": predicted_id,
                            "positive_role": positive_role,
                            "predicted_role": predicted_role,
                            "positive_score": round(positive_score, 6),
                            "predicted_score": round(predicted_score, 6),
                            "score_gap": round(predicted_score - positive_score, 6),
                            "positive_rank": positive_rank,
                            "top_candidates": top_candidates,
                        }
                    )

    errors.sort(key=lambda item: item["score_gap"], reverse=True)
    report = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "split": split,
        "summary": {
            "total": total,
            "correct": correct_total,
            "accuracy": round(correct_total / max(total, 1), 6),
            "errors": total - correct_total,
        },
        "metrics": finalize_metrics(metrics),
        "error_role_confusion": {
            f"{positive_role}->{predicted_role}": count
            for (positive_role, predicted_role), count in confusion.most_common()
        },
        "checkpoint_load": {
            "missing_keys": list(missing_keys),
            "unexpected_keys": list(unexpected_keys),
        },
        "top_errors": errors[: args.top_errors],
    }

    output_path = Path(args.output or f"outputs/analysis/ranker_errors_{split}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("RANKER ERROR ANALYSIS")
    print(f"split: {split}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"accuracy: {correct_total}/{total} = {report['summary']['accuracy']:.4f}")
    for group, values in report["metrics"].items():
        print(f"{group}:")
        for key, metric in values.items():
            print(
                f"  {key}: correct={metric['correct']} total={metric['total']} "
                f"acc={metric['accuracy']:.4f}"
            )
    print("error_role_confusion:")
    for key, count in report["error_role_confusion"].items():
        print(f"  {key}: {count}")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
