from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import os

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

print("Loading dataset...")
dataset = load_dataset("imdb")

def tokenize_data(example):
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=512)

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_data, batched=True)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",      
    save_strategy="epoch",           
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,              
    load_best_model_at_end=True
)

last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    checkpoints = [os.path.join(training_args.output_dir, d) for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getctime)  
        print(f"Resuming training from checkpoint {last_checkpoint}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  
)

print("Starting training...")
trainer.train(resume_from_checkpoint=last_checkpoint)

model.save_pretrained("results/my_finetuned_bert")
tokenizer.save_pretrained("results/my_finetuned_bert")

print("Model training completed and saved.")
