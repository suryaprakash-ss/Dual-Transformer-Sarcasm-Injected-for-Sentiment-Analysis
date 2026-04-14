# Importing the Necessary Libraries
import os
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Compatibility shim: some versions of `accelerate` have Accelerator.unwrap_model
# that accepts `keep_torch_compile`; older versions do not. Trainer in transformers
# may call unwrap_model(..., keep_torch_compile=False). If the installed
# accelerate lacks that parameter, we provide a small wrapper that accepts and
# ignores it to maintain compatibility without requiring package changes.
try:
    # Import lazily; if accelerate is not installed this will fail harmlessly
    from accelerate import Accelerator
    import inspect

    # Only patch if unwrap_model exists and does not accept keep_torch_compile
    if hasattr(Accelerator, "unwrap_model"):
        sig = inspect.signature(Accelerator.unwrap_model)
        if "keep_torch_compile" not in sig.parameters:
            _orig_unwrap = Accelerator.unwrap_model

            def _compat_unwrap(self, *args, **kwargs):
                # Pop unsupported kwarg if present and call original
                kwargs.pop("keep_torch_compile", None)
                return _orig_unwrap(self, *args, **kwargs)

            Accelerator.unwrap_model = _compat_unwrap
except Exception:
    # If anything goes wrong (accelerate missing or unexpected API), skip patch.
    pass

os.environ["WANDB_DISABLED"] = "true"

# Load the Dataset
sentiment_ds = load_dataset("yelp_polarity")
emotion_ds = load_dataset("go_emotions")
sarcasm_ds = load_dataset("raquiba/Sarcasm_News_Headline")

# Loading the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization Function
def tokenize_fn(batch, text_field="text"):
    return tokenizer(batch[text_field], padding="max_length", truncation=True, max_length=128)

# Apply Tokenization
sentiment_ds = sentiment_ds.map(lambda x: tokenize_fn(x, "text"), batched=True)
emotion_ds = emotion_ds.map(lambda x: tokenize_fn(x, "text"), batched=True)
sarcasm_ds = sarcasm_ds.map(lambda x: tokenize_fn(x, "headline"), batched=True)

# Fixing Sarcasm Dataset Labels
sarcasm_ds = sarcasm_ds.rename_column("is_sarcastic", "label")

# Formating Datasets for PyTorch
sentiment_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
emotion_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
sarcasm_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# print("Preprocessing complete!")
# print("Sentiment example:", sentiment_ds["train"][0])
# print("Emotion example:", emotion_ds["train"][0])
# print("Sarcasm example:", sarcasm_ds["train"][0])

# Loading the DistilBERT Model for Sequence Classification
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./sentiment_results",
    do_eval=True,
    save_steps=5000,
    eval_steps=5000,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100
)

# Initialize Trainer
trainer = Trainer(
    model=sentiment_model,
    args=training_args,
    train_dataset=sentiment_ds["train"].select(range(20000)),
    eval_dataset=sentiment_ds["test"].select(range(5000)),
    tokenizer=tokenizer
)

# Train the Model
trainer.train()

# Evaluate the Model
metrics = trainer.evaluate()
print("Baseline Sentiment Model Evaluation:", metrics)

# Predictions
predictions = trainer.predict(sentiment_ds["test"].select(range(5000)))
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

# Calculate Accuracy
accuracy = accuracy_score(y_true, y_pred)