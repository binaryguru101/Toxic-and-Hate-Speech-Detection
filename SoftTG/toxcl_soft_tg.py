
import os
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers.modeling_outputs import BaseModelOutput


def extract_target_groups_from_augmented_text(text: str) -> List[str]:
    """
    Expected format from the original repo data:
        "Target: {TG} Post: {raw_text}"
    TG may contain one or more comma-separated groups.
    """
    if text is None:
        return []

    if not isinstance(text, str):
        text = str(text)

    match = re.search(r"Target\s*:\s*(.*?)\s*Post\s*:", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []

    tg_text = match.group(1).strip()
    if tg_text == "" or tg_text.lower() in {"none", "[none]", "nan"}:
        return []

    groups = []
    for part in re.split(r"\s*,\s*", tg_text):
        clean = part.strip().lower()
        if clean:
            groups.append(clean)
    return groups


class ToxclDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def _label_to_int(self, value):
        if isinstance(value, (int, float)):
            return int(value)
        s = str(value).strip().lower()
        return 1 if s in {"1", "toxic", "implicit hate speech", "hate", "yes", "true"} else 0

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = self._label_to_int(item["label"])
        return {
            "document": item["document"],
            "summary": item["summary"],
            "student_cls_labels": label,
            "teacher_cls_labels": label,
            "target_groups": extract_target_groups_from_augmented_text(item["document"]),
        }


class ToXCL(nn.Module):
    """
    Soft target-group version:
    - classifier uses pooled encoder state + pooled target-group embedding
    - decoder receives an extra prefix memory token built from target groups
    """
    def __init__(
        self,
        decoder_model,
        num_target_groups: int,
        pad_token_id: int,
        distill_alpha: float = 1.0,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()
        self.decoder_model = decoder_model
        self.hidden_size = decoder_model.config.d_model
        self.pad_token_id = pad_token_id
        self.distill_alpha = distill_alpha
        self.max_grad_norm = max_grad_norm

        self.tg_embedding = nn.Embedding(max(num_target_groups, 1), self.hidden_size)
        self.tg_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier_fuse = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, 2)

        self.dropout = nn.Dropout(0.1)
        self.cls_loss_fn = nn.CrossEntropyLoss()

    def encode_target_groups(self, tg_ids: torch.Tensor, tg_mask: torch.Tensor) -> torch.Tensor:
        tg_emb = self.tg_embedding(tg_ids)
        tg_mask_f = tg_mask.unsqueeze(-1).float()
        denom = tg_mask_f.sum(dim=1).clamp(min=1.0)
        pooled = (tg_emb * tg_mask_f).sum(dim=1) / denom
        return torch.tanh(self.tg_proj(pooled))

    def _build_encoder_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tg_ids: torch.Tensor,
        tg_mask: torch.Tensor,
    ) -> Tuple[BaseModelOutput, torch.Tensor, torch.Tensor]:
        encoder = self.decoder_model.get_encoder()
        enc = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = enc.last_hidden_state

        mask_f = attention_mask.unsqueeze(-1).float()
        pooled_enc = (hidden_states * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        tg_vec = self.encode_target_groups(tg_ids, tg_mask)

        fused_cls = torch.tanh(self.classifier_fuse(torch.cat([pooled_enc, tg_vec], dim=-1)))
        fused_cls = self.dropout(fused_cls)

        tg_memory = tg_vec.unsqueeze(1)
        modified_hidden = torch.cat([tg_memory, hidden_states], dim=1)
        modified_mask = torch.cat(
            [
                torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),
                attention_mask,
            ],
            dim=1,
        )

        return BaseModelOutput(last_hidden_state=modified_hidden), fused_cls, modified_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lm_labels: torch.Tensor,
        cls_labels: torch.Tensor,
        tg_ids: torch.Tensor,
        tg_mask: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
    ):
        encoder_outputs, fused_cls, modified_attention_mask = self._build_encoder_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tg_ids=tg_ids,
            tg_mask=tg_mask,
        )

        cls_logits = self.classifier(fused_cls)
        cls_loss = self.cls_loss_fn(cls_logits, cls_labels)

        seq_outputs = self.decoder_model(
            attention_mask=modified_attention_mask,
            encoder_outputs=encoder_outputs,
            labels=lm_labels,
            return_dict=True,
        )
        lm_loss = seq_outputs.loss

        kl_loss = torch.tensor(0.0, device=input_ids.device)
        if teacher_logits is not None:
            student_log_probs = F.log_softmax(cls_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * self.distill_alpha

        return cls_logits, lm_loss, cls_loss, kl_loss

    @torch.no_grad()
    def generate_expl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tg_ids: torch.Tensor,
        tg_mask: torch.Tensor,
        **gen_kwargs,
    ):
        encoder_outputs, _, modified_attention_mask = self._build_encoder_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tg_ids=tg_ids,
            tg_mask=tg_mask,
        )
        return self.decoder_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=modified_attention_mask,
            **gen_kwargs,
        )

    def save_checkpoint(self, output_dir, is_best=False, optimizer=None, scheduler=None, training_stats=None):
        os.makedirs(output_dir, exist_ok=True)

        ckpt = {
            "model_state_dict": self.state_dict(),
            "is_best": is_best,
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        if training_stats is not None:
            ckpt["training_stats"] = training_stats

        torch.save(ckpt, os.path.join(output_dir, "soft_tg_ckpt.pt"))
        self.decoder_model.save_pretrained(output_dir)

    def load_checkpoint(self, output_dir, optimizer=None, scheduler=None):
        ckpt_path = os.path.join(output_dir, "soft_tg_ckpt.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt


def build_target_group_vocab(examples: List[dict]) -> dict:
    vocab = {}
    for ex in examples:
        for tg in extract_target_groups_from_augmented_text(ex["document"]):
            if tg not in vocab:
                vocab[tg] = len(vocab) + 1
    return vocab


def encode_target_groups_batch(batch_target_groups: List[List[str]], tg_vocab: dict, top_k: int):
    tg_ids = []
    tg_mask = []
    for groups in batch_target_groups:
        ids = [tg_vocab[g] for g in groups if g in tg_vocab][:top_k]
        mask = [1] * len(ids)

        while len(ids) < top_k:
            ids.append(0)
            mask.append(0)

        tg_ids.append(ids)
        tg_mask.append(mask)

    return torch.tensor(tg_ids, dtype=torch.long), torch.tensor(tg_mask, dtype=torch.long)
