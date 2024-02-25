from typing import Iterable

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel


def mean_pool(token_embeds: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool


class Sbert(torch.nn.Module):
    def __init__(
        self,
        device,
        max_length: int = 128,
    ):
        super().__init__()
        self.max_length = max_length
        self.device = device
        self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size * 3, 3)

    def forward(self, data) -> torch.tensor:
        premise_input_ids = data["premise_input_ids"].to(self.device)
        premise_attention_mask = data["premise_attention_mask"].to(self.device)
        hypothesis_input_ids = data["hypothesis_input_ids"].to(self.device)
        hypothesis_attention_mask = data["hypothesis_attention_mask"].to(self.device)

        out_premise = self.bert_model(premise_input_ids, premise_attention_mask)
        out_hypothesis = self.bert_model(
            hypothesis_input_ids, hypothesis_attention_mask
        )
        premise_embeds = out_premise.last_hidden_state
        hypothesis_embeds = out_hypothesis.last_hidden_state

        pooled_premise_embeds = mean_pool(premise_embeds, premise_attention_mask)
        pooled_hypotheses_embeds = mean_pool(
            hypothesis_embeds, hypothesis_attention_mask
        )

        embeds = torch.cat(
            [
                pooled_premise_embeds,
                pooled_hypotheses_embeds,
                torch.abs(pooled_premise_embeds - pooled_hypotheses_embeds),
            ],
            dim=-1,
        )
        return self.linear(embeds)


class CrossEncoderBert(torch.nn.Module):
    def __init__(self, max_length: int = 128):
        super().__init__()
        self.max_length = max_length
        self.bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the CLS token's output
        return self.linear(pooled_output)


class StsDataset(Dataset):
    def __init__(self, tokens: dict, labels: list[float]):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, ix: int) -> dict[str, torch.tensor]:
        return {
            "input_ids": torch.tensor(self.tokens["input_ids"][ix], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.tokens["attention_mask"][ix], dtype=torch.long
            ),
            "labels": torch.tensor(
                self.labels[ix], dtype=torch.float
            ),  # Use float for regression
        }

    def __len__(self) -> int:
        return len(self.tokens["input_ids"])


class SnliDataset(Dataset):
    def __init__(
        self, premise_tokens: dict, hypothesis_tokens: dict, labels: Iterable[str]
    ):
        self.premise_tokens = premise_tokens
        self.hypothesis_tokens = hypothesis_tokens
        self.labels = labels
        self._init_data()

    def _init_data(self) -> None:
        self.data = []
        for pt_ids, pt_am, ht_ids, ht_am, label in zip(
            self.premise_tokens["input_ids"],
            self.premise_tokens["attention_mask"],
            self.hypothesis_tokens["input_ids"],
            self.hypothesis_tokens["attention_mask"],
            self.labels,
        ):
            data = {}
            data["premise_input_ids"] = torch.tensor(pt_ids, dtype=torch.long)
            data["premise_attention_mask"] = torch.tensor(pt_am, dtype=torch.long)
            data["hypothesis_input_ids"] = torch.tensor(ht_ids, dtype=torch.long)
            data["hypothesis_attention_mask"] = torch.tensor(ht_am, dtype=torch.long)
            data["label"] = torch.tensor(label, dtype=torch.long)
            self.data.append(data)

    def __getitem__(self, ix: int) -> dict[str, torch.tensor]:
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)
