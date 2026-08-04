import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.datasets.prefix_dataset import PrefixRankingDataset, prefix_ranking_collate_fn
from src.models.ranker import CrossEncoderRanker, DualEncoderStateRanker


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_loader(
    samples_path: str,
    memory_path: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    role_targets_path: str = None,
    require_labeled_positive: bool = False,
    context_mode: str = "full_state",
):
    dataset = PrefixRankingDataset(
        samples_path=samples_path,
        memory_path=memory_path,
        role_targets_path=role_targets_path,
        require_labeled_positive=require_labeled_positive,
        context_mode=context_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=prefix_ranking_collate_fn,
    )
    return dataset, loader


def move_to_device(batch_tokens: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch_tokens.items()}


def forward_ranker(
    model,
    batch: dict,
    tokenizer,
    device: torch.device,
    architecture: str,
    max_length: int,
    state_max_length: int,
    candidate_max_length: int,
    return_deficit: bool,
    return_contribution: bool,
):
    if architecture == "dual_state_interaction":
        state_tokens = move_to_device(
            tokenizer(
                batch["state_texts"],
                padding=True,
                truncation=True,
                max_length=state_max_length,
                return_tensors="pt",
            ),
            device,
        )
        candidate_tokens = move_to_device(
            tokenizer(
                batch["flat_candidate_questions"],
                batch["flat_text_b"],
                padding=True,
                truncation=True,
                max_length=candidate_max_length,
                return_tensors="pt",
            ),
            device,
        )
        return model(
            state_input_ids=state_tokens["input_ids"],
            state_attention_mask=state_tokens["attention_mask"],
            state_token_type_ids=state_tokens.get("token_type_ids"),
            candidate_input_ids=candidate_tokens["input_ids"],
            candidate_attention_mask=candidate_tokens["attention_mask"],
            candidate_token_type_ids=candidate_tokens.get("token_type_ids"),
            candidate_counts=batch["candidate_counts"],
            return_deficit=return_deficit,
            return_contribution=return_contribution,
        )

    tokens = move_to_device(
        tokenizer(
            batch["flat_text_a"],
            batch["flat_text_b"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ),
        device,
    )
    return model(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        token_type_ids=tokens.get("token_type_ids"),
        return_deficit=return_deficit,
        return_contribution=return_contribution,
    )


def get_positive_flat_indices(candidate_counts, labels, device):
    offsets = []
    cursor = 0
    for count, label in zip(candidate_counts, labels.tolist()):
        offsets.append(cursor + label)
        cursor += count
    return torch.tensor(offsets, dtype=torch.long, device=device)


def compute_margin_loss(packed_scores, labels, margin: float):
    positive_scores = packed_scores.gather(1, labels.unsqueeze(1)).squeeze(1)
    negative_scores = packed_scores.clone()
    negative_scores.scatter_(1, labels.unsqueeze(1), torch.finfo(packed_scores.dtype).min)
    hardest_negative_scores = negative_scores.max(dim=1).values
    return F.relu(margin - positive_scores + hardest_negative_scores).mean()


def compute_acquired_negative_margin_loss(
    packed_scores: torch.Tensor,
    labels: torch.Tensor,
    acquired_candidate_indices: List[List[int]],
    margin: float,
):
    losses = []
    for row_idx, acquired_indices in enumerate(acquired_candidate_indices):
        positive_score = packed_scores[row_idx, labels[row_idx]]
        for candidate_idx in acquired_indices:
            losses.append(
                F.relu(
                    margin
                    - positive_score
                    + packed_scores[row_idx, int(candidate_idx)]
                )
            )
    if not losses:
        return None
    return torch.stack(losses).mean()


def acquired_pair_metrics(
    packed_scores: torch.Tensor,
    labels: torch.Tensor,
    acquired_candidate_indices: List[List[int]],
):
    correct = 0
    total = 0
    for row_idx, acquired_indices in enumerate(acquired_candidate_indices):
        positive_score = packed_scores[row_idx, labels[row_idx]]
        for candidate_idx in acquired_indices:
            correct += int(positive_score > packed_scores[row_idx, int(candidate_idx)])
            total += 1
    return correct, total


def load_compatible_state_dict(model, state_dict: dict):
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    skipped = sorted(set(state_dict) - set(compatible))
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return list(missing), list(unexpected), skipped


def mean_pack_vectors(flat_vectors: torch.Tensor, candidate_counts) -> torch.Tensor:
    rows = []
    cursor = 0
    for count in candidate_counts:
        rows.append(flat_vectors[cursor: cursor + count].mean(dim=0))
        cursor += count
    if cursor != flat_vectors.size(0):
        raise ValueError(
            "flat vector count does not match candidate_counts: "
            f"flat={flat_vectors.size(0)}, packed={cursor}"
        )
    return torch.stack(rows, dim=0)


def compute_deficit_loss(deficit_preds, deficit_labels):
    mask = deficit_labels != -100.0
    if not mask.any():
        return None
    return F.mse_loss(deficit_preds[mask], deficit_labels[mask])


def compute_deficit_mae(deficit_preds, deficit_labels):
    mask = deficit_labels != -100.0
    if not mask.any():
        return 0.0, 0
    return torch.abs(deficit_preds[mask] - deficit_labels[mask]).sum().item(), int(mask.sum().item())


def compute_contribution_loss(contribution_preds, contribution_labels):
    mask = contribution_labels != -100.0
    if not mask.any():
        return None
    return F.mse_loss(contribution_preds[mask], contribution_labels[mask])


def compute_contribution_mae(contribution_preds, contribution_labels):
    mask = contribution_labels != -100.0
    if not mask.any():
        return 0.0, 0
    return torch.abs(contribution_preds[mask] - contribution_labels[mask]).sum().item(), int(mask.sum().item())


def compute_candidate_role_loss(flat_role_logits, flat_candidate_role_ids):
    role_mask = flat_candidate_role_ids != -100
    if not role_mask.any():
        return None
    return F.cross_entropy(flat_role_logits[role_mask], flat_candidate_role_ids[role_mask])


def compute_candidate_role_metrics(flat_role_logits, flat_candidate_role_ids):
    role_mask = flat_candidate_role_ids != -100
    if not role_mask.any():
        return 0, 0
    correct = (
        flat_role_logits[role_mask].argmax(dim=-1) == flat_candidate_role_ids[role_mask]
    ).sum().item()
    return int(correct), int(role_mask.sum().item())


def run_eval(
    model,
    loader,
    tokenizer,
    device,
    max_length: int,
    architecture: str = "cross_encoder",
    state_max_length: int = 320,
    candidate_max_length: int = 192,
    role_aux_weight: float = 0.0,
    deficit_aux_weight: float = 0.0,
    candidate_role_aux_weight: float = 0.0,
    contribution_aux_weight: float = 0.0,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct = 0
    role_correct = 0
    role_total = 0
    candidate_role_correct = 0
    candidate_role_total = 0
    deficit_abs_error = 0.0
    deficit_total = 0
    contribution_abs_error = 0.0
    contribution_total = 0
    acquired_pair_correct = 0
    acquired_pair_total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)

            model_outputs = forward_ranker(
                model=model,
                batch=batch,
                tokenizer=tokenizer,
                device=device,
                architecture=architecture,
                max_length=max_length,
                state_max_length=state_max_length,
                candidate_max_length=candidate_max_length,
                return_deficit=deficit_aux_weight > 0.0,
                return_contribution=contribution_aux_weight > 0.0,
            )
            flat_scores, flat_role_logits = model_outputs[0], model_outputs[1]
            output_cursor = 2
            if deficit_aux_weight > 0.0:
                flat_deficit_preds = model_outputs[output_cursor]
                output_cursor += 1
            if contribution_aux_weight > 0.0:
                flat_contribution_preds = model_outputs[output_cursor]
            packed_scores, _ = CrossEncoderRanker.pack_scores(flat_scores, batch["candidate_counts"])
            pair_correct, pair_total = acquired_pair_metrics(
                packed_scores,
                labels,
                batch["acquired_candidate_indices"],
            )
            acquired_pair_correct += pair_correct
            acquired_pair_total += pair_total

            loss = F.cross_entropy(packed_scores, labels)
            role_labels = torch.tensor(batch["positive_role_ids"], dtype=torch.long, device=device)
            role_mask = role_labels != -100
            if role_aux_weight > 0.0 and role_mask.any():
                positive_flat_indices = get_positive_flat_indices(batch["candidate_counts"], labels, device)
                positive_role_logits = flat_role_logits[positive_flat_indices]
                role_loss = F.cross_entropy(positive_role_logits[role_mask], role_labels[role_mask])
                loss = loss + role_aux_weight * role_loss
                role_correct += (
                    positive_role_logits[role_mask].argmax(dim=-1) == role_labels[role_mask]
                ).sum().item()
                role_total += role_mask.sum().item()
            if candidate_role_aux_weight > 0.0:
                flat_candidate_role_ids = torch.tensor(
                    batch["flat_candidate_role_ids"], dtype=torch.long, device=device
                )
                candidate_role_loss = compute_candidate_role_loss(flat_role_logits, flat_candidate_role_ids)
                if candidate_role_loss is not None:
                    loss = loss + candidate_role_aux_weight * candidate_role_loss
                    batch_correct, batch_total = compute_candidate_role_metrics(
                        flat_role_logits, flat_candidate_role_ids
                    )
                    candidate_role_correct += batch_correct
                    candidate_role_total += batch_total
            if deficit_aux_weight > 0.0:
                deficit_labels = torch.tensor(batch["deficit_labels"], dtype=torch.float, device=device)
                deficit_preds = mean_pack_vectors(flat_deficit_preds, batch["candidate_counts"])
                deficit_loss = compute_deficit_loss(deficit_preds, deficit_labels)
                if deficit_loss is not None:
                    loss = loss + deficit_aux_weight * deficit_loss
                    batch_abs_error, batch_total = compute_deficit_mae(deficit_preds, deficit_labels)
                    deficit_abs_error += batch_abs_error
                    deficit_total += batch_total
            if contribution_aux_weight > 0.0:
                contribution_labels = torch.tensor(
                    batch["positive_contribution_labels"], dtype=torch.float, device=device
                )
                positive_flat_indices = get_positive_flat_indices(batch["candidate_counts"], labels, device)
                positive_contribution_preds = flat_contribution_preds[positive_flat_indices]
                contribution_loss = compute_contribution_loss(positive_contribution_preds, contribution_labels)
                if contribution_loss is not None:
                    loss = loss + contribution_aux_weight * contribution_loss
                    batch_abs_error, batch_total = compute_contribution_mae(
                        positive_contribution_preds, contribution_labels
                    )
                    contribution_abs_error += batch_abs_error
                    contribution_total += batch_total
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite eval loss detected. "
                    f"qids={batch['qids'][:3]}, ts={batch['ts'][:3]}"
                )
            preds = packed_scores.argmax(dim=-1)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    acc = correct / max(total_samples, 1)

    return {
        "loss": avg_loss,
        "acc": acc,
        "role_acc": role_correct / max(role_total, 1),
        "role_labeled": role_total,
        "candidate_role_acc": candidate_role_correct / max(candidate_role_total, 1),
        "candidate_role_labeled": candidate_role_total,
        "deficit_mae": deficit_abs_error / max(deficit_total, 1),
        "deficit_labeled": deficit_total,
        "contribution_mae": contribution_abs_error / max(contribution_total, 1),
        "contribution_labeled": contribution_total,
        "acquired_pair_acc": acquired_pair_correct / max(acquired_pair_total, 1),
        "acquired_pairs": acquired_pair_total,
    }


def train_one_epoch(
    model,
    loader,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    device,
    max_length: int,
    architecture: str,
    state_max_length: int,
    candidate_max_length: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    use_fp16: bool,
    log_every: int,
    role_aux_weight: float,
    margin_loss_weight: float,
    margin: float,
    deficit_aux_weight: float,
    candidate_role_aux_weight: float,
    contribution_aux_weight: float,
    acquired_negative_margin_weight: float,
    acquired_negative_margin: float,
):
    model.train()

    total_loss = 0.0
    total_samples = 0
    correct = 0
    role_correct = 0
    role_total = 0
    candidate_role_correct = 0
    candidate_role_total = 0
    deficit_abs_error = 0.0
    deficit_total = 0
    contribution_abs_error = 0.0
    contribution_total = 0
    acquired_pair_correct = 0
    acquired_pair_total = 0

    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="train", leave=False)
    for step_idx, batch in enumerate(progress, start=1):
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", enabled=use_fp16):
            model_outputs = forward_ranker(
                model=model,
                batch=batch,
                tokenizer=tokenizer,
                device=device,
                architecture=architecture,
                max_length=max_length,
                state_max_length=state_max_length,
                candidate_max_length=candidate_max_length,
                return_deficit=deficit_aux_weight > 0.0,
                return_contribution=contribution_aux_weight > 0.0,
            )
            flat_scores, flat_role_logits = model_outputs[0], model_outputs[1]
            output_cursor = 2
            if deficit_aux_weight > 0.0:
                flat_deficit_preds = model_outputs[output_cursor]
                output_cursor += 1
            if contribution_aux_weight > 0.0:
                flat_contribution_preds = model_outputs[output_cursor]
            packed_scores, _ = CrossEncoderRanker.pack_scores(flat_scores, batch["candidate_counts"])
            loss = F.cross_entropy(packed_scores, labels)
            if margin_loss_weight > 0.0:
                loss = loss + margin_loss_weight * compute_margin_loss(packed_scores, labels, margin)
            if acquired_negative_margin_weight > 0.0:
                acquired_margin_loss = compute_acquired_negative_margin_loss(
                    packed_scores,
                    labels,
                    batch["acquired_candidate_indices"],
                    acquired_negative_margin,
                )
                if acquired_margin_loss is not None:
                    loss = loss + acquired_negative_margin_weight * acquired_margin_loss

            role_labels = torch.tensor(batch["positive_role_ids"], dtype=torch.long, device=device)
            role_mask = role_labels != -100
            if role_aux_weight > 0.0 and role_mask.any():
                positive_flat_indices = get_positive_flat_indices(batch["candidate_counts"], labels, device)
                positive_role_logits = flat_role_logits[positive_flat_indices]
                role_loss = F.cross_entropy(positive_role_logits[role_mask], role_labels[role_mask])
                loss = loss + role_aux_weight * role_loss
            if candidate_role_aux_weight > 0.0:
                flat_candidate_role_ids = torch.tensor(
                    batch["flat_candidate_role_ids"], dtype=torch.long, device=device
                )
                candidate_role_loss = compute_candidate_role_loss(flat_role_logits, flat_candidate_role_ids)
                if candidate_role_loss is not None:
                    loss = loss + candidate_role_aux_weight * candidate_role_loss
            if deficit_aux_weight > 0.0:
                deficit_labels = torch.tensor(batch["deficit_labels"], dtype=torch.float, device=device)
                deficit_preds = mean_pack_vectors(flat_deficit_preds, batch["candidate_counts"])
                deficit_loss = compute_deficit_loss(deficit_preds, deficit_labels)
                if deficit_loss is not None:
                    loss = loss + deficit_aux_weight * deficit_loss
            if contribution_aux_weight > 0.0:
                contribution_labels = torch.tensor(
                    batch["positive_contribution_labels"], dtype=torch.float, device=device
                )
                positive_flat_indices = get_positive_flat_indices(batch["candidate_counts"], labels, device)
                positive_contribution_preds = flat_contribution_preds[positive_flat_indices]
                contribution_loss = compute_contribution_loss(positive_contribution_preds, contribution_labels)
                if contribution_loss is not None:
                    loss = loss + contribution_aux_weight * contribution_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite train loss detected. "
                    f"step={step_idx}, qids={batch['qids'][:3]}, ts={batch['ts'][:3]}"
                )
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if step_idx % grad_accum_steps == 0 or step_idx == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            preds = packed_scores.argmax(dim=-1)
            bs = labels.size(0)
            total_loss += loss.item() * grad_accum_steps * bs
            total_samples += bs
            correct += (preds == labels).sum().item()
            pair_correct, pair_total = acquired_pair_metrics(
                packed_scores,
                labels,
                batch["acquired_candidate_indices"],
            )
            acquired_pair_correct += pair_correct
            acquired_pair_total += pair_total
            if role_aux_weight > 0.0 and role_mask.any():
                role_correct += (
                    positive_role_logits[role_mask].argmax(dim=-1) == role_labels[role_mask]
                ).sum().item()
                role_total += role_mask.sum().item()
            if candidate_role_aux_weight > 0.0:
                batch_correct, batch_total = compute_candidate_role_metrics(
                    flat_role_logits, flat_candidate_role_ids
                )
                candidate_role_correct += batch_correct
                candidate_role_total += batch_total
            if deficit_aux_weight > 0.0:
                batch_abs_error, batch_total = compute_deficit_mae(deficit_preds, deficit_labels)
                deficit_abs_error += batch_abs_error
                deficit_total += batch_total
            if contribution_aux_weight > 0.0:
                batch_abs_error, batch_total = compute_contribution_mae(
                    positive_contribution_preds, contribution_labels
                )
                contribution_abs_error += batch_abs_error
                contribution_total += batch_total

        if step_idx % log_every == 0:
            progress.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                acc=f"{correct / max(total_samples, 1):.4f}",
            )

    avg_loss = total_loss / max(total_samples, 1)
    acc = correct / max(total_samples, 1)

    return {
        "loss": avg_loss,
        "acc": acc,
        "role_acc": role_correct / max(role_total, 1),
        "role_labeled": role_total,
        "candidate_role_acc": candidate_role_correct / max(candidate_role_total, 1),
        "candidate_role_labeled": candidate_role_total,
        "deficit_mae": deficit_abs_error / max(deficit_total, 1),
        "deficit_labeled": deficit_total,
        "contribution_mae": contribution_abs_error / max(contribution_total, 1),
        "contribution_labeled": contribution_total,
        "acquired_pair_acc": acquired_pair_correct / max(acquired_pair_total, 1),
        "acquired_pairs": acquired_pair_total,
    }


def save_checkpoint(output_dir: str, model, tokenizer, config: dict, epoch: int, val_metrics: dict):
    ensure_dir(output_dir)

    ckpt_path = os.path.join(output_dir, "best_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        ckpt_path,
    )

    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    save_json(val_metrics, os.path.join(output_dir, "best_val_metrics.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_ranker.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(config["seed"])
    output_dir = str(config["output_dir"])
    ensure_dir(output_dir)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_name = config["model"]["pretrained_name"]
    dropout = float(config["model"].get("dropout", 0.1))
    architecture = str(config["model"].get("architecture", "cross_encoder"))
    if architecture not in {"cross_encoder", "dual_state_interaction"}:
        raise ValueError(f"unsupported model architecture: {architecture}")

    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"]["num_workers"])
    epochs = int(config["train"]["epochs"])
    lr = float(config["train"]["lr"])
    weight_decay = float(config["train"]["weight_decay"])
    warmup_ratio = float(config["train"]["warmup_ratio"])
    max_length = int(config["train"]["max_length"])
    state_max_length = int(config["train"].get("state_max_length", max_length))
    candidate_max_length = int(config["train"].get("candidate_max_length", max_length))
    grad_accum_steps = int(config["train"]["grad_accum_steps"])
    max_grad_norm = float(config["train"]["max_grad_norm"])
    log_every = int(config["train"]["log_every"])
    use_fp16 = bool(config["train"]["fp16"]) and device.type == "cuda"
    role_aux_weight = float(config["train"].get("role_aux_weight", 0.0))
    candidate_role_aux_weight = float(config["train"].get("candidate_role_aux_weight", 0.0))
    contribution_aux_weight = float(config["train"].get("contribution_aux_weight", 0.0))
    margin_loss_weight = float(config["train"].get("margin_loss_weight", 0.0))
    margin = float(config["train"].get("margin", 0.2))
    deficit_aux_weight = float(config["train"].get("deficit_aux_weight", 0.0))
    acquired_negative_margin_weight = float(
        config["train"].get("acquired_negative_margin_weight", 0.0)
    )
    acquired_negative_margin = float(
        config["train"].get("acquired_negative_margin", margin)
    )
    context_mode = str(config["data"].get("context_mode", "full_state"))

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    train_dataset, train_loader = build_loader(
        samples_path=config["data"]["train_samples"],
        memory_path=config["data"]["train_memory"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        role_targets_path=config["data"].get("train_role_targets"),
        require_labeled_positive=bool(config["data"].get("train_require_labeled_positive", False)),
        context_mode=str(config["data"].get("train_context_mode", context_mode)),
    )
    val_dataset, val_loader = build_loader(
        samples_path=config["data"]["val_samples"],
        memory_path=config["data"]["val_memory"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        role_targets_path=config["data"].get("val_role_targets"),
        context_mode=str(config["data"].get("val_context_mode", context_mode)),
    )
    test_dataset, test_loader = build_loader(
        samples_path=config["data"]["test_samples"],
        memory_path=config["data"]["test_memory"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        role_targets_path=config["data"].get("test_role_targets"),
        context_mode=str(config["data"].get("test_context_mode", context_mode)),
    )

    if architecture == "dual_state_interaction":
        model = DualEncoderStateRanker(
            pretrained_name=pretrained_name,
            dropout=dropout,
            projection_dim=int(config["model"].get("projection_dim", 256)),
        ).to(device)
    else:
        model = CrossEncoderRanker(
            pretrained_name=pretrained_name,
            dropout=dropout,
        ).to(device)
    init_checkpoint = str(config["train"].get("init_checkpoint", "") or "").strip()
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing_keys, unexpected_keys, skipped_keys = load_compatible_state_dict(
            model,
            state_dict,
        )
        print(f"Loaded init checkpoint: {init_checkpoint}")
        if missing_keys:
            print(f"  missing_keys: {missing_keys}")
        if unexpected_keys:
            print(f"  unexpected_keys: {unexpected_keys}")
        if skipped_keys:
            print(f"  skipped_incompatible_keys: {skipped_keys}")

    head_lr = float(config["train"].get("head_lr", lr))
    if architecture == "dual_state_interaction" and head_lr != lr:
        optimizer = AdamW(
            [
                {"params": model.encoder.parameters(), "lr": lr},
                {
                    "params": [
                        parameter
                        for name, parameter in model.named_parameters()
                        if not name.startswith("encoder.")
                    ],
                    "lr": head_lr,
                },
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / max(grad_accum_steps, 1))
    total_train_steps = num_update_steps_per_epoch * epochs
    warmup_steps = int(total_train_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    print("===== dataset stats =====")
    print(f"train samples: {len(train_dataset)}")
    print(f"train skipped unlabeled positives: {train_dataset.skipped_unlabeled_positive}")
    print(f"context mode: {context_mode}")
    print(f"model architecture: {architecture}")
    print(f"val samples:   {len(val_dataset)}")
    print(f"test samples:  {len(test_dataset)}")
    print("===== training =====")

    history = []
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            max_length=max_length,
            architecture=architecture,
            state_max_length=state_max_length,
            candidate_max_length=candidate_max_length,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            use_fp16=use_fp16,
            log_every=log_every,
            role_aux_weight=role_aux_weight,
            margin_loss_weight=margin_loss_weight,
            margin=margin,
            deficit_aux_weight=deficit_aux_weight,
            candidate_role_aux_weight=candidate_role_aux_weight,
            contribution_aux_weight=contribution_aux_weight,
            acquired_negative_margin_weight=acquired_negative_margin_weight,
            acquired_negative_margin=acquired_negative_margin,
        )

        val_metrics = run_eval(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            architecture=architecture,
            state_max_length=state_max_length,
            candidate_max_length=candidate_max_length,
            role_aux_weight=role_aux_weight,
            deficit_aux_weight=deficit_aux_weight,
            candidate_role_aux_weight=candidate_role_aux_weight,
            contribution_aux_weight=contribution_aux_weight,
        )

        print(
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f}"
            f" val_role_acc={val_metrics['role_acc']:.4f}"
            f" val_candidate_role_acc={val_metrics['candidate_role_acc']:.4f}"
            f" val_deficit_mae={val_metrics['deficit_mae']:.4f}"
            f" val_contribution_mae={val_metrics['contribution_mae']:.4f}"
            f" val_acquired_pair_acc={val_metrics['acquired_pair_acc']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            save_checkpoint(
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                config=config,
                epoch=epoch,
                val_metrics=val_metrics,
            )

    save_json(history, os.path.join(output_dir, "train_history.json"))

    print(f"\nBest epoch: {best_epoch}, best val acc: {best_val_acc:.4f}")

    best_ckpt = torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = run_eval(
        model=model,
        loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        architecture=architecture,
        state_max_length=state_max_length,
        candidate_max_length=candidate_max_length,
        role_aux_weight=role_aux_weight,
        deficit_aux_weight=deficit_aux_weight,
        candidate_role_aux_weight=candidate_role_aux_weight,
        contribution_aux_weight=contribution_aux_weight,
    )
    save_json(test_metrics, os.path.join(output_dir, "test_metrics.json"))

    print(
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['acc']:.4f}"
    )


if __name__ == "__main__":
    main()
