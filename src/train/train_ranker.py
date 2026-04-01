import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.datasets.prefix_dataset import PrefixRankingDataset, prefix_ranking_collate_fn
from src.models.ranker import CrossEncoderRanker


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


def build_loader(samples_path: str, memory_path: str, batch_size: int, num_workers: int, shuffle: bool):
    dataset = PrefixRankingDataset(samples_path=samples_path, memory_path=memory_path)
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


def run_eval(model, loader, tokenizer, device, max_length: int) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            tokens = tokenizer(
                batch["flat_text_a"],
                batch["flat_text_b"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = move_to_device(tokens, device)
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)

            flat_scores = model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                token_type_ids=tokens.get("token_type_ids"),
            )
            packed_scores, _ = CrossEncoderRanker.pack_scores(flat_scores, batch["candidate_counts"])

            loss = F.cross_entropy(packed_scores, labels)
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
    grad_accum_steps: int,
    max_grad_norm: float,
    use_fp16: bool,
    log_every: int,
):
    model.train()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="train", leave=False)
    for step_idx, batch in enumerate(progress, start=1):
        tokens = tokenizer(
            batch["flat_text_a"],
            batch["flat_text_b"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = move_to_device(tokens, device)
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", enabled=use_fp16):
            flat_scores = model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                token_type_ids=tokens.get("token_type_ids"),
            )
            packed_scores, _ = CrossEncoderRanker.pack_scores(flat_scores, batch["candidate_counts"])
            loss = F.cross_entropy(packed_scores, labels)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if step_idx % grad_accum_steps == 0:
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

    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"]["num_workers"])
    epochs = int(config["train"]["epochs"])
    lr = float(config["train"]["lr"])
    weight_decay = float(config["train"]["weight_decay"])
    warmup_ratio = float(config["train"]["warmup_ratio"])
    max_length = int(config["train"]["max_length"])
    grad_accum_steps = int(config["train"]["grad_accum_steps"])
    max_grad_norm = float(config["train"]["max_grad_norm"])
    log_every = int(config["train"]["log_every"])
    use_fp16 = bool(config["train"]["fp16"]) and device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    train_dataset, train_loader = build_loader(
        samples_path=config["data"]["train_samples"],
        memory_path=config["data"]["train_memory"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dataset, val_loader = build_loader(
        samples_path=config["data"]["val_samples"],
        memory_path=config["data"]["val_memory"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    test_dataset, test_loader = build_loader(
        samples_path=config["data"]["test_samples"],
        memory_path=config["data"]["test_memory"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    model = CrossEncoderRanker(
        pretrained_name=pretrained_name,
        dropout=dropout,
    ).to(device)

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
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            use_fp16=use_fp16,
            log_every=log_every,
        )

        val_metrics = run_eval(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )

        print(
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f}"
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
    )
    save_json(test_metrics, os.path.join(output_dir, "test_metrics.json"))

    print(
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['acc']:.4f}"
    )


if __name__ == "__main__":
    main()