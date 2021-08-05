from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_scheduler
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
import multiprocessing


class SimilarityDataset(Dataset):
    def __init__(self, tokenizer, data_list, max_token_length=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

        token_ids_arr = map(
            self.tokenizer.encode,
            [item["sentence1"] for item in data_list],
            [item["sentence2"] for item in data_list],
        )
        labels_arr = [item["labels"]["binary-label"] for item in data_list]
        self.data_list = list(zip(token_ids_arr, labels_arr))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        tokens = self.data_list[idx][0]
        if len(tokens) < self.max_token_length:
            tokens += [self.tokenizer.pad_token_id] * (
                self.max_token_length - len(tokens)
            )

        tokens, labels = (
            torch.LongTensor(tokens[: self.max_token_length]),
            torch.LongTensor([self.data_list[idx][1]]),
        )
        return tokens, labels


class SimilarityModel(pl.LightningModule):
    def __init__(self, backbone_size="small", lr=1e-4, max_token_length=256):
        super().__init__()

        self.loss_func = nn.CrossEntropyLoss()
        self.lr = lr
        self.max_token_length = max_token_length
        self.class_num = 2  # binary classification

        # model & tokenizer prepare
        self.backbone_size = backbone_size
        if self.backbone_size == "small":
            self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "klue/roberta-small"
            )
        elif self.backbone_size == "base":
            self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "klue/roberta-base"
            )
        elif self.backbone_size == "large":
            self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "klue/roberta-large"
            )
        else:
            raise ValueError("backbone size should be one of [small, base, large]")

        nn.init.xavier_uniform_(self.model.classifier.out_proj.weight)

        self.save_hyperparameters()

    def forward(self, x):
        pred = self.model(x).logits
        return pred

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def prepare_token_ids(self, query1, query2=None):
        if query2 is None:
            tokens = self.tokenizer.encode(query1)
        else:
            tokens = self.tokenizer.encode(query1, query2)

        if len(tokens) < self.max_token_length:
            tokens += [self.tokenizer.pad_token_id] * (
                self.max_token_length - len(tokens)
            )

        return torch.LongTensor(tokens[: self.max_token_length]).unsqueeze(0)

    def predict(self, query1, query2):
        with torch.no_grad():
            result = self.forward(self.prepare_token_ids(query1, query2))
            result = nn.Softmax(dim=1)(result).squeeze(0)
            confidence = (max(result) - min(result)).item()
            class_idx = torch.argmax(result).item()

            return {'class_idx': class_idx, 'confidence': confidence}

    def training_step(self, batch, batch_idx):
        self.model.train()
        token_ids, labels = batch
        pred_ids = self.forward(token_ids)

        loss = self.loss_func(pred_ids, labels.squeeze(1))
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        with torch.no_grad():
            token_ids, labels = batch
            pred_ids = self.forward(token_ids)

            loss = self.loss_func(pred_ids, labels.squeeze(1))
            self.log("val_loss", loss, prog_bar=True)

            return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    # model preparation
    model = SimilarityModel(lr=args.lr)

    # data preparation
    similarity_dataset = load_dataset("klue", "sts")
    train_data = similarity_dataset["train"]
    valid_data = similarity_dataset["validation"]

    train_dataset = SimilarityDataset(model.tokenizer, train_data)
    valid_dataset = SimilarityDataset(model.tokenizer, valid_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count(),
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count(),
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath="./",
        filename="klue_textual_similarity",
    )

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        progress_bar_refresh_rate=1,
        accelerator="ddp",
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, valid_loader)
