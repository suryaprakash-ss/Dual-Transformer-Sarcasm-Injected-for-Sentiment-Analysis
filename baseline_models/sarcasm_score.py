# Import Necessary Libraries

from datasets import load_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
import torch
from transformers import Trainer
from transformers import AutoModelForSequenceClassification, TrainingArguments
from sklearn.metrics import accuracy_score

import numpy as np

# Load the dataset
sarcasm_ds = load_dataset("raquiba/Sarcasm_News_Headline")
sarcasm_ds

#Preprocessing
def preprocess_sarcasm(batch):
    return tokenizer(batch["headline"], truncation=True, padding="max_length", max_length=128)

sarcasm_ds = sarcasm_ds.map(preprocess_sarcasm, batched=True)

sarcasm_ds = sarcasm_ds.rename_column("is_sarcastic", "label")

sarcasm_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


import os

# Model checkpoint path
checkpoint_dir = "./sarcasm_results/checkpoint-final"

# Load model from checkpoint if exists, else train
if os.path.exists(checkpoint_dir):
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    print("Loaded sarcasm model from checkpoint.")
else:
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

#Training args
sarcasm_training_args = TrainingArguments(
    output_dir="./sarcasm_results",
    do_eval=True,
    save_steps=1000,
    eval_steps=1000,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_dir="./sarcasm_logs",
    logging_steps=50,
    report_to="none"
)

#Compute Metrics
def compute_metrics_sarcasm(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    return {"accuracy": float(accuracy_score(pred.label_ids, preds))}

sarcasm_trainer = Trainer(
    model=sarcasm_model,
    args=sarcasm_training_args,
    train_dataset=sarcasm_ds["train"].select(range(10000)),   # subset for speed
    eval_dataset=sarcasm_ds["test"].select(range(2000)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_sarcasm
)


# Train sarcasm model only if not already trained
if not os.path.exists(checkpoint_dir):
    sarcasm_trainer.train()
    sarcasm_trainer.save_model(checkpoint_dir)
    print("Model trained and saved.")

#Evaluate the model
sarcasm_metrics = sarcasm_trainer.evaluate()
print("Sarcasm Model Evaluation:", sarcasm_metrics)

#Sample Run
texts = [
    "Yeah, I *really* needed my flight to be delayed for 5 hours.",
    "I love sunny days and warm weather!",
    "Oh great, another Monday morning..."
]
inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

sarcasm_model.eval()

# Move to cuda if available
if torch.cuda.is_available():
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

# Remove token_type_ids for DistilBERT compatibility (always)
inputs.pop("token_type_ids", None)

sarcasm_model.eval()
with torch.no_grad():
    outputs = sarcasm_model(**inputs)
    logits = outputs.logits

probabilities = torch.softmax(logits, dim=1)
sarcasm_scores = probabilities[:, 1]
print(sarcasm_scores)

#Probability Scores
sarcasm_scores_list = sarcasm_scores.cpu().numpy().tolist()
print(sarcasm_scores_list)


