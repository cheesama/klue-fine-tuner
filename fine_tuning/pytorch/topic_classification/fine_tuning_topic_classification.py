from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_scheduler
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm
from tensordash.torchdash import Torchdash

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
import multiprocessing
import os, sys


class TopicDataset(Dataset):
    def __init__(self, tokenizer, data_list, max_token_length=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

        token_ids_arr = map(
            self.tokenizer.encode, [item["title"] for item in data_list]
        )
        labels_arr = [item["label"] for item in data_list]
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


class TopicModel(pl.LightningModule):
    def __init__(
        self,
        backbone_size="small",
        lr=1e-4,
        max_token_length=256,
        topic_class_num=7,
        max_epochs=8,
        class_dict=None,
    ):
        super().__init__()

        self.loss_func = nn.CrossEntropyLoss()
        self.lr = lr
        self.max_token_length = max_token_length
        self.topic_class_num = topic_class_num  # based on KLUE ynat dataset
        self.max_epochs = max_epochs
        self.class_dict = None

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

        self.model.classifier.out_proj = nn.Linear(
            self.model.classifier.out_proj.in_features, self.topic_class_num
        )
        nn.init.xavier_uniform_(self.model.classifier.out_proj.weight)

        self.save_hyperparameters()

        if (
            os.getenv("TENSORDASH_EMAIL") is not None
            and os.getenv("TENSORDASH_PWD") is not None
        ):
            self.tensordash_callback = Torchdash(
                ModelName=f"klue-roberta-{self.backbone_size}-fine-tune-topic_classification",
                email=os.getenv("TENSORDASH_EMAIL"),
                password=os.getenv("TENSORDASH_PWD"),
            )
        else:
            self.tensordash_callback = None

    def forward(self, x):
        pred = self.model(x).logits
        return pred

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def prepare_token_ids(self, input_data):
        if type(input_data) == list:
            input_data = "".join(input_data)

        tokens = self.tokenizer.encode(input_data)
        if len(tokens) < self.max_token_length:
            tokens += [self.tokenizer.pad_token_id] * (
                self.max_token_length - len(tokens)
            )

        return torch.LongTensor(tokens[: self.max_token_length]).unsqueeze(0)

    def predict(self, text):
        pred = self.forward(self.prepare_token_ids(tokens))[0]
        class_idx = torch.argmax(pred).item()

        if self.class_dict is not None:
            label = self.class_dict[class_idx]
            return {"label": label, "class_idx": class_idx}

        return {"class_idx": class_idx}

    def training_step(self, batch, batch_idx):
        self.model.train()
        token_ids, labels = batch
        pred_ids = self.forward(token_ids)

        loss = self.loss_func(pred_ids, labels.squeeze(1))
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        if self.tensordash_callback is not None:
            self.tensordash_callback.sendLoss(loss=avg_loss.item(), epoch=self.current_epoch, total_epochs=self.max_epochs)

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
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--epochs", default=8, type=int)
    # parser.add_argument("--batch_size", default=64, type=int) # small model, 12GB GPU based
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--model_size", default='small', choices=['small','base','large'])
    args = parser.parse_args()

    # model preparation
    model = TopicModel(lr=args.lr, backbone_size=args.model_size, max_epochs=args.epochs)

    # data preparation
    topic_dataset = load_dataset("klue", "ynat")
    train_data = topic_dataset["train"]
    valid_data = topic_dataset["validation"]

    train_dataset = TopicDataset(model.tokenizer, train_data)
    valid_dataset = TopicDataset(model.tokenizer, valid_data)

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

    callback_list = []
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath="./",
        filename="klue_topic_classification",
    )
    callback_list.append(checkpoint_callback)

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        progress_bar_refresh_rate=1,
        accelerator="ddp",
        max_epochs=args.epochs,
        callbacks=callback_list,
    )

    trainer.fit(model, train_loader, valid_loader)
