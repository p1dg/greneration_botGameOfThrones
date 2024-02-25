import numpy as np
import random
from tqdm import tqdm
import datasets

import matplotlib.pyplot as plt

from typing import Callable

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup

from utils.warning_catcher import catch_warnings
from model.model_class import Sbert, SnliDataset, mean_pool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode(
    input_texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str = "cpu",
) -> torch.tensor:
    model.eval()
    tokenized_texts = tokenizer(
        input_texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    token_embeds = model(
        tokenized_texts["input_ids"].to(DEVICE),
        tokenized_texts["attention_mask"].to(device),
    ).last_hidden_state
    pooled_embeds = mean_pool(
        token_embeds, tokenized_texts["attention_mask"].to(DEVICE)
    )
    return pooled_embeds


def get_train_step_fn(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn: torch.nn.CrossEntropyLoss,
) -> Callable[[torch.tensor, torch.tensor], float]:
    def train_step_fn(x: torch.tensor, y: torch.tensor) -> float:
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step_fn


def get_val_step_fn(
    model: torch.nn.Module, loss_fn: torch.nn.CrossEntropyLoss
) -> Callable[[torch.tensor, torch.tensor], float]:
    def val_step_fn(x: torch.tensor, y: torch.tensor) -> float:
        model.eval()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        return loss.item()

    return val_step_fn


def mini_batch(
    dataloader: DataLoader,
    step_fn: Callable[[torch.tensor, torch.tensor], float],
    is_training: bool = True,
    batch_size: int = 16,
) -> tuple[np.array, list[float]]:
    mini_batch_losses = []

    if is_training:
        desc = "training"
    else:
        desc = "validatin"
    n_steps = len(dataloader)
    progress_bar = tqdm(total=n_steps, desc=desc)
    progress_bar.set_postfix({"loss": None})
    for i, data in enumerate(dataloader):
        loss = step_fn(data, data["label"].to(DEVICE))
        mini_batch_losses.append(loss)
        if i % (batch_size * 10) == 0:
            progress_bar.set_postfix({"loss": f"{np.mean(mini_batch_losses): .3f}"})

        progress_bar.update(1)

    return np.mean(mini_batch_losses), mini_batch_losses


@catch_warnings(log_path="./logs/warnings_bi_encoder.log")
def train_bi_encoder_model(
    data,
    limit_context=200,
    max_length=128,
    seed=2101,
    window_size=32,
    plot_path="./model/plots/bi_encoder_training.png",
    model_path=None,
    save_path="./models_storage/sbert_trained",
):
    if model_path is not None:
        return Sbert(device=DEVICE, max_length=max_length).bert_model.from_pretrained(
            model_path
        )

    print(f"!!!training bi_encoder on {DEVICE}!!!")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ##
    talk_dataset = {"premise": [], "hypothesis": [], "label": []}
    cols = ["query", "close_reply", "neutral_reply", "bad_reply", "context"]
    for index, row in data[cols].iterrows():
        close_reply = row["close_reply"]
        neutral_reply = row["neutral_reply"]
        bad_reply = row["bad_reply"]

        hypothesis = close_reply + neutral_reply + bad_reply
        labels = (
            [1 for _ in range(len(close_reply))]
            + [0 for _ in range(len(neutral_reply))]
            + [2 for _ in range(len(bad_reply))]
        )

        premise = [
            row["query"] + "[SEP]" + row["context"][:limit_context]
            for _ in range(len(hypothesis))
        ]

        if len(premise) != len(labels) != len(hypothesis):
            print(f"mistake with row {index}")
            break

        else:
            talk_dataset["premise"].extend(premise)
            talk_dataset["hypothesis"].extend(hypothesis)
            talk_dataset["label"].extend(labels)

    dataset = datasets.Dataset.from_dict(talk_dataset)

    ##

    filtered_data = []
    for data in tqdm(dataset, total=len(dataset), desc="filtered_data"):
        if random.random() < 0.1:
            filtered_data.append(data)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tokenized_premises = tokenizer(
        [data["premise"] for data in filtered_data],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        verbose=True,
    )

    tokenized_hypothesis = tokenizer(
        [data["hypothesis"] for data in filtered_data],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        verbose=True,
    )

    snli_dataset = SnliDataset(
        tokenized_premises,
        tokenized_hypothesis,
        (data["label"] for data in filtered_data),
    )

    train_ratio = 0.8

    n_total = len(snli_dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(snli_dataset, [n_train, n_val])

    batch_size = 16  # mentioned in the paper
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    bi_encoder = Sbert(device=DEVICE, max_length=max_length)
    bi_encoder.to(DEVICE)

    optimizer = torch.optim.AdamW(bi_encoder.parameters(), lr=2e-6)

    total_steps = len(train_dataset) // batch_size
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps - warmup_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    n_epochs = 15  # mentioned in the paper

    train_step_fn = get_train_step_fn(bi_encoder, optimizer, scheduler, loss_fn)
    val_step_fn = get_val_step_fn(bi_encoder, loss_fn)

    train_losses, train_mini_batch_losses = [], []
    val_losses, val_mini_batch_losses = [], []

    for _ in tqdm(range(1, n_epochs + 1), desc="model bi_encoder training/validation"):
        train_loss, _train_mini_batch_losses = mini_batch(
            train_dataloader, train_step_fn, batch_size=batch_size
        )
        train_mini_batch_losses += _train_mini_batch_losses
        train_losses.append(train_loss)

        with torch.no_grad():
            val_loss, _val_mini_batch_losses = mini_batch(
                val_dataloader, val_step_fn, is_training=False
            )
            val_mini_batch_losses += _val_mini_batch_losses
            val_losses.append(val_loss)

    bi_encoder.bert_model.save_pretrained(save_path)

    # рисуем график обучения
    train_mb_running_loss = []
    for i in range(len(train_mini_batch_losses) - window_size):
        train_mb_running_loss.append(
            np.mean(train_mini_batch_losses[i : i + window_size])
        )
    val_mb_running_loss = []
    for i in range(len(val_mini_batch_losses) - window_size):
        val_mb_running_loss.append(np.mean(val_mini_batch_losses[i : i + window_size]))

    _, ax = plt.subplots(figsize=(14, 10))
    ax.plot(range(len(train_mb_running_loss)), train_mb_running_loss)
    plt.title("Bi-encoder training")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    return bi_encoder
