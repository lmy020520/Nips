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
        num_contributions: int = 4,
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
        self.contribution_regressor = nn.Linear(hidden_size, num_contributions)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        return_deficit: bool = False,
        return_contribution: bool = False,
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
        outputs = [flat_scores, role_logits]
        if return_deficit:
            deficit_preds = torch.sigmoid(self.deficit_regressor(cls_state))  # [num_pairs, num_deficits]
            outputs.append(deficit_preds)
        if return_contribution:
            contribution_preds = torch.sigmoid(self.contribution_regressor(cls_state))  # [num_pairs, num_contributions]
            outputs.append(contribution_preds)
        return tuple(outputs)

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


class DualEncoderStateRanker(nn.Module):
    """Shared-backbone dual tower with explicit state-candidate interaction."""

    def __init__(
        self,
        pretrained_name: str,
        dropout: float = 0.1,
        projection_dim: int = 256,
        num_roles: int = 3,
        num_deficits: int = 4,
        num_contributions: int = 4,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        self.encoder.float()
        hidden_size = self.encoder.config.hidden_size
        self.state_projection = nn.Linear(hidden_size, projection_dim)
        self.candidate_projection = nn.Linear(hidden_size, projection_dim)
        self.state_norm = nn.LayerNorm(projection_dim)
        self.candidate_norm = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(dropout)
        interaction_size = projection_dim * 4
        self.interaction = nn.Sequential(
            nn.Linear(interaction_size, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scorer = nn.Linear(projection_dim, 1)
        self.role_classifier = nn.Linear(projection_dim, num_roles)
        self.deficit_regressor = nn.Linear(projection_dim, num_deficits)
        self.contribution_regressor = nn.Linear(projection_dim, num_contributions)

    def _encode(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        state_input_ids: torch.Tensor,
        state_attention_mask: torch.Tensor,
        candidate_input_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        candidate_counts: List[int],
        state_token_type_ids: torch.Tensor = None,
        candidate_token_type_ids: torch.Tensor = None,
        return_deficit: bool = False,
        return_contribution: bool = False,
    ) -> torch.Tensor:
        state_hidden = self._encode(
            state_input_ids,
            state_attention_mask,
            state_token_type_ids,
        )
        candidate_hidden = self._encode(
            candidate_input_ids,
            candidate_attention_mask,
            candidate_token_type_ids,
        )
        state_vector = self.state_norm(self.state_projection(state_hidden))
        candidate_vector = self.candidate_norm(
            self.candidate_projection(candidate_hidden)
        )
        counts = torch.as_tensor(
            candidate_counts,
            dtype=torch.long,
            device=state_vector.device,
        )
        repeated_state = torch.repeat_interleave(state_vector, counts, dim=0)
        if repeated_state.size(0) != candidate_vector.size(0):
            raise ValueError(
                "candidate count mismatch: "
                f"state-expanded={repeated_state.size(0)}, "
                f"candidates={candidate_vector.size(0)}"
            )
        interaction_input = torch.cat(
            [
                repeated_state,
                candidate_vector,
                repeated_state * candidate_vector,
                torch.abs(repeated_state - candidate_vector),
            ],
            dim=-1,
        )
        interaction = self.interaction(self.dropout(interaction_input))
        flat_scores = self.scorer(interaction).squeeze(-1)
        role_logits = self.role_classifier(interaction)
        outputs = [flat_scores, role_logits]
        if return_deficit:
            state_deficit = torch.sigmoid(self.deficit_regressor(state_vector))
            outputs.append(torch.repeat_interleave(state_deficit, counts, dim=0))
        if return_contribution:
            outputs.append(torch.sigmoid(self.contribution_regressor(interaction)))
        return tuple(outputs)

    pack_scores = staticmethod(CrossEncoderRanker.pack_scores)
