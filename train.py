"""This is draft file. Maynot work."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, get_scheduler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model import DistilBERTClass


import torchmetrics
import seaborn as sns

from dataset import ToxicityDataset
from utils import get_device

device = get_device()

MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-05


def prepare_data(dataset_path):
    # train_df = pd.read_csv("./data/train.csv")
    train_df = pd.read_csv(dataset_path)
    train_df.drop(["id"], inplace=True, axis=1)
    train_df["labels"] = train_df.iloc[:, 1:].values.tolist()
    train_df.drop(train_df.columns.values[1:-1].tolist(), inplace=True, axis=1)
    train_df["comment_text"] = train_df["comment_text"].str.lower()

    train_df["comment_text"] = (
        train_df["comment_text"]
        .str.replace("\xa0", " ", regex=False)
        .str.split()
        .str.join(" ")
    )
    comments = train_df.comment_text.tolist()

    labels = train_df.labels.to_list()
    train_text, eval_text, train_labels, eval_labels = train_test_split(
        comments, labels, test_size=0.15
    )

    train_text, test_text, train_labels, test_labels = train_test_split(
        train_text, train_labels, test_size=0.20
    )

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", truncation=True, do_lower_case=True
    )

    training_dataset = ToxicityDataset(
        tokenizer, train_text, labels=train_labels, max_len=MAX_LEN
    )
    eval_dataset = ToxicityDataset(
        tokenizer, eval_text, labels=eval_labels, max_len=MAX_LEN
    )
    test_dataset = ToxicityDataset(
        tokenizer, test_text, labels=test_labels, max_len=MAX_LEN
    )

    train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": True}

    training_loader = DataLoader(training_dataset, **train_params)
    eval_loader = DataLoader(eval_dataset, **train_params)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE)
    return training_loader, eval_loader, test_loader


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train_model(model, data, epoch, optimizer, lr_scheduler):
    model.train()
    loss_batch = []
    accuracy_batch = []
    with tqdm(data, desc=f"Epoch {epoch+1}") as pbar:
        for idx, data in enumerate(data, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(
                device, dtype=torch.long
            )
            targets = data["targets"].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            acc = accuracy(outputs, targets)
            f1 = f1_score(outputs, targets)
            pbar.set_postfix(
                {
                    "Loss": loss.item(),
                    "Accuracy": acc.item(),
                    "F1-Score": f1.item(),
                }
            )

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            pbar.update()


def evaluate_model(model, data_loader):
    model.eval()
    target_vals = []
    preds = []
    with tqdm(data_loader) as pbar:
        for idx, data in enumerate(data_loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(
                device, dtype=torch.long
            )
            targets = data["targets"].to(device, dtype=torch.float)
            with torch.no_grad():
                outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            acc = accuracy(outputs, targets)
            f1 = f1_score(outputs, targets)
            pbar.set_postfix(
                {
                    "Loss": loss.item(),
                    "Accuracy": acc.item(),
                    "F1-Score": f1.item(),
                }
            )
            preds.append(outputs)
            target_vals.append(targets)

            pbar.update()
    preds = torch.cat(preds)
    target_vals = torch.cat(target_vals)
    print(
        f"\nAccuracy: {accuracy(preds, target_vals)} F1-Score: {f1_score(preds, target_vals)}"
    )


def train(model, train_loader, eval_loader):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    for epoch in range(EPOCHS):
        train_model(model, train_loader, epoch, optimizer, lr_scheduler)
        evaluate_model(eval_loader)


if __name__ == "__main__":
    model = DistilBERTClass()
    model.to(get_device())
    dataset_path = "data/train.csv"
    training_loader, eval_loader, test_loader = prepare_data(dataset_path)
    task = "multilabel"
    num_labels = 6

    accuracy = torchmetrics.Accuracy(task=task, num_labels=num_labels).to(
        device
    )
    f1_score = torchmetrics.F1Score(task, num_labels=num_labels).to(device)
    train(model, training_loader, eval_loader)
