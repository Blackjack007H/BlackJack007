from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from datasets import load_dataset
import torch
from sklearn.metrics import classification_report

model_name = "results/my_finetuned_bert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

print("Loading and tokenizing test dataset...")
dataset = load_dataset("imdb")
tokenized_datasets = dataset.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=512), batched=True)

trainer = Trainer(model=model)

print("Evaluating model...")
predictions = trainer.predict(tokenized_datasets["test"])

predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1)
true_labels = predictions.label_ids

report = classification_report(true_labels, predicted_labels, target_names=["negative", "positive"])
print(report)
