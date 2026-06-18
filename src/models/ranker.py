from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class CrossEncoderRanker(nn.Module):
    def __init__(
        self,
        pretrained_name: str,
        dropout: float = 0.1,
        num_roles: int = 3,
        num_deficits: int = 4,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        # Some local mirrors/checkpoints may resolve to fp16 weights. Keep the
        # trainable base model in fp32; autocast will handle mixed precision when
        # explicitly enabled by the training config.
        self.encoder.float()
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(hidden_size, 1)
        self.role_classifier = nn.Linear(hidden_size, num_roles)
        self.deficit_regressor = nn.Linear(hidden_size, num_deficits)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        return_deficit: bool = False,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        cls_state = outputs.last_hidden_state[:, 0, :]
        cls_state = self.dropout(cls_state)
        flat_scores = self.scorer(cls_state).squeeze(-1)  # [num_pairs]
        role_logits = self.role_classifier(cls_state)  # [num_pairs, num_roles]
        if return_deficit:
            deficit_preds = torch.sigmoid(self.deficit_regressor(cls_state))  # [num_pairs, num_deficits]
            return flat_scores, role_logits, deficit_preds
        return flat_scores, role_logits

    @staticmethod
    def pack_scores(
        flat_scores: torch.Tensor,
        candidate_counts: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(candidate_counts)
        max_candidates = max(candidate_counts)

        # Use the minimum representable value for the current dtype to stay fp16-safe.
        neg_large = torch.finfo(flat_scores.dtype).min

        padded_scores = flat_scores.new_full((batch_size, max_candidates), neg_large)
        mask = torch.zeros(
            (batch_size, max_candidates),
            dtype=torch.bool,
            device=flat_scores.device,
        )

        cursor = 0
        for i, count in enumerate(candidate_counts):
            padded_scores[i, :count] = flat_scores[cursor: cursor + count]
            mask[i, :count] = True
            cursor += count

        if cursor != flat_scores.size(0):
            raise ValueError(
                "flat_scores count does not match candidate_counts: "
                f"flat={flat_scores.size(0)}, packed={cursor}"
            )

        return padded_scores, mask
