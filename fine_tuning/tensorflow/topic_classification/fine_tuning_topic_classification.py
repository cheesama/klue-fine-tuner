from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from transformers import TFAutoModelForSequenceClassification
from transformers import AdamW, get_scheduler
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer

from tqdm.auto import tqdm

import argparse
import multiprocessing
import tensorflow as tf

def create_topic_dataset(tokenizer, data_list, max_seq_length=256):
    x = []
    y = []
    for data in tqdm(data_list, desc='generating dataset ...'):
        tokens = tokenizer.encode(data['title'])
        if len(tokens) < max_seq_length:
            tokens += [tokenizer.pad_token_id] * (max_seq_length - len(tokens))
        label = data['label']

        x.append(tokens)
        y.append(label)

    return x, y

def create_topic_model(backbone_size="small", lr=5e-5, max_token_length=256, topic_class_num=7, class_dict=None):
        # model & tokenizer prepare
        if backbone_size == "small":
            tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
            encoder = TFAutoModelForSequenceClassification.from_pretrained(
                "klue/roberta-small"
            )
        elif backbone_size == "base":
            tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
            encoder = TFAutoModelForSequenceClassification.from_pretrained(
                "klue/roberta-base"
            )
        elif backbone_size == "large":
            tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
            encoder = TFAutoModelForSequenceClassification.from_pretrained(
                "klue/roberta-large"
            )
        else:
            raise ValueError("backbone size should be one of [small, base, large]")

        # input layer
        input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)

        embedding = encoder(input_ids)[0]
        pred = layers.Dense(topic_class_num, name='pred_layer')(embedding)
        model = keras.Model(inputs=[input_ids], outputs=[pred])

        loss = keras.losses.SparseCategoricalCrossentropy()
        optimizer = keras.optimizers.Adam(lr=lr)
        model.compile(optimizer=optimizer, loss=[loss])

        return model




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    # parser.add_argument("--batch_size", default=64, type=int) # small model, 12GB GPU based
    parser.add_argument("--batch_size", default=2, type=int)
    args = parser.parse_args()

    # model preparation
    model = TopicModel(lr=args.lr)

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

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath="./",
        filename="klue_topic_classification",
    )
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        progress_bar_refresh_rate=1,
        accelerator="ddp",
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, valid_loader)
