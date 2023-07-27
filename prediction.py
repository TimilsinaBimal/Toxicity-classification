import os
import torch
from transformers import DistilBertTokenizer
from dataset import ToxicityDataset
from torch.utils.data import DataLoader
from model import DistilBERTClass
from utils import get_device
import gdown

device = get_device()


def prepare_data(text):
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", truncation=True, do_lower_case=True
    )

    test_dataset = ToxicityDataset(
        tokenizer,
        comments=[text],
        labels=None,
        max_len=512,
        evaluation_mode=True,
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader


def load_model(model_path):
    model = DistilBERTClass()
    model.to(get_device())
    os.makedirs("/".join(model_path.split("/")[:-1]), exist_ok=True)
    if not os.path.exists(model_path):
        print("Downloading Pretrained Model!")
        url = (
            "https://drive.google.com/uc?id=1-78xAf_LWNG5WnQZVHgym6ULqcXgpp6Q"
        )
        gdown.download(url, model_path, quiet=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def predict(text):
    test_loader = prepare_data(text)
    model = load_model("./models/model_state_dict.pt")
    model.eval()
    for data in test_loader:
        with torch.no_grad():
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(
                device, dtype=torch.long
            )
            preds = model(ids, mask, token_type_ids)
    preds = torch.nn.Sigmoid()(preds)
    probas = preds
    labels = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    pred = probas.cpu().tolist()
    pred_df = {
        label: round(p, 2) for label, p in zip(labels, pred[0]) if p >= 0.5
    }
    return pred_df
