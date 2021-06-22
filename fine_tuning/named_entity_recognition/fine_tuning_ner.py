from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification
# dataset prepare
ner_dataset = load_dataset('klue', 'ner')

# model & tokenizer prepare
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
model = AutoModelForTokenClassification.from_pretrained("klue/roberta-small")

#tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
#model = AutoModelForMaskedLM.from_pretrained("klue/roberta-base")

#tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
#model = AutoModelForMaskedLM.from_pretrained("klue/roberta-large")

print ('model load done')

train_dataset = ner_dataset['train']
valid_dataset = ner_dataset['validation']

training_args = TrainingArguments("test_trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

trainer.train()



