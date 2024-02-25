import numpy as np
import random
from tqdm import tqdm
import datasets

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup

from utils.warning_catcher import catch_warnings
from model.model_class import StsDataset, CrossEncoderBert

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_step_fn(model, optimizer, scheduler, loss_fn, batch):
    model.train()
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    optimizer.zero_grad()
    logits = model(input_ids, attention_mask)
    loss = loss_fn(logits.squeeze(-1), labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


def val_step_fn(model, loss_fn, batch):
    model.eval()
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    loss = loss_fn(logits.squeeze(-1), labels)
    return loss.item()


def mini_batch(
    dataloader,
    model,
    optimizer,
    scheduler,
    loss_fn,
    step_fn,
    batch_size,
    is_training=True,
):
    if is_training:
        desc = "training"
    else:
        desc = "validatin"
    mini_batch_losses = []
    progress_bar = tqdm(total=len(dataloader), desc=desc)
    for i, batch in enumerate(dataloader):
        if is_training:
            loss = step_fn(model, optimizer, scheduler, loss_fn, batch)
        else:
            loss = step_fn(model, loss_fn, batch)
        mini_batch_losses.append(loss)
        if i % (batch_size * 4) == 0:
            progress_bar.set_postfix({"loss": f"{np.mean(mini_batch_losses): .3f}"})

        progress_bar.update(1)
    return np.mean(mini_batch_losses), mini_batch_losses


@catch_warnings(log_path="./logs/warnings_cross_encoder.log")
def train_cross_encoder_model(
    sbert_path="./models_storage/sbert_trained",
    max_length=128,
    seed=2101,
    plot_path="./model/plots/cross_encoder_training.png",
    model_path=None,
    save_path="./models_storage/cross_encoder_trained.pth",
):
    if model_path is not None:
        model = CrossEncoderBert()
        model.load_state_dict(torch.load(model_path))
        return model
    print(f"!!!training cross_encoder on {DEVICE}!!!")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.from_pretrained(sbert_path)

    dataset = datasets.load_dataset("glue", "stsb", split="train")

    tokenized_texts = tokenizer(
        [data["sentence1"] for data in dataset],
        [data["sentence2"] for data in dataset],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        verbose=True,
    )

    sts_dataset = StsDataset(tokenized_texts, [data["label"] for data in dataset])

    train_ratio = 0.8
    n_total = len(sts_dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(sts_dataset, [n_train, n_val])

    batch_size = 16  # mentioned in the paper
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    cross_encoder = CrossEncoderBert(max_length=max_length).to(DEVICE)

    optimizer = torch.optim.AdamW(cross_encoder.parameters(), lr=3e-5)

    total_steps = len(train_dataset) // batch_size
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps - warmup_steps,
    )

    loss_fn = torch.nn.MSELoss()

    n_epochs = 5
    train_losses, train_mini_batch_losses = [], []
    val_losses, val_mini_batch_losses = [], []

    for _ in tqdm(range(1, n_epochs + 1), desc="model cross_encoder training"):
        train_loss, _train_mini_batch_losses = mini_batch(
            dataloader=train_dataloader,
            model=cross_encoder,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            step_fn=train_step_fn,
            batch_size=batch_size,
            is_training=True,
        )
        train_mini_batch_losses.extend(_train_mini_batch_losses)
        train_losses.append(train_loss)

        with torch.no_grad():
            val_loss, _val_mini_batch_losses = mini_batch(
                dataloader=val_dataloader,
                model=cross_encoder,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                step_fn=val_step_fn,
                batch_size=batch_size,
                is_training=False,
            )
            val_mini_batch_losses.extend(_val_mini_batch_losses)
            val_losses.append(val_loss)

    torch.save(cross_encoder.state_dict(), save_path)

    # рисуем график обучения
    window_size = 32

    train_mb_running_loss = []
    for i in range(len(train_mini_batch_losses) - window_size):
        train_mb_running_loss.append(
            np.mean(train_mini_batch_losses[i : i + window_size])
        )

    val_mb_running_loss = []
    for i in range(len(val_mini_batch_losses) - window_size):
        val_mb_running_loss.append(np.mean(val_mini_batch_losses[i : i + window_size]))

    _, ax = plt.subplots(figsize=(14, 8))
    ax.plot(range(len(train_mb_running_loss)), train_mb_running_loss)

    plt.title("Cross-encoder training")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    return cross_encoder
