from torch.utils.data import Dataset
import torch


class ToxicityDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        comments,
        labels=None,
        max_len=512,
        evaluation_mode=True,
    ):
        self.tokenizer = tokenizer
        self.text = comments
        self.eval_mode = evaluation_mode
        if not self.eval_mode:
            self.targets = labels
        self.max_len = max_len

    def __getitem__(self, idx):
        text = str(self.text[idx])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        out = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }
        if not self.eval_mode:
            out["targets"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return out

    def __len__(self):
        return len(self.text)
