# ==========================================
# 🔹 DUAL TRANSFORMER SARCASTIC SENTIMENT+EMOTION MODEL (Increased Data/Epochs)
# ==========================================
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    DistilBertModel, Trainer, TrainingArguments
)
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1️⃣ Dataset Loading and Preprocessing
# ==========================================
sentiment_ds = load_dataset("yelp_polarity")
emotion_ds = load_dataset("go_emotions")
sarcasm_ds_loaded = load_dataset("raquiba/Sarcasm_News_Headline")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

sentiment_ds = sentiment_ds.map(preprocess_function, batched=True)
sentiment_ds = sentiment_ds.rename_column("label", "labels")
sentiment_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

emotion_ds = emotion_ds.map(preprocess_function, batched=True)
emotion_ds = emotion_ds.rename_column("labels", "label_list")

# Convert multi-label to single (argmax)
def simplify_labels(example):
    example["labels"] = int(torch.tensor(example["label_list"]).argmax())
    return example

emotion_ds = emotion_ds.map(simplify_labels)
emotion_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Preprocess sarcasm dataset
def preprocess_sarcasm(examples):
    return tokenizer(examples["headline"], truncation=True, padding="max_length", max_length=128)

sarcasm_ds_loaded = sarcasm_ds_loaded.map(preprocess_sarcasm, batched=True)
sarcasm_ds_loaded = sarcasm_ds_loaded.rename_column("is_sarcastic", "labels")
sarcasm_ds_loaded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Assuming num_emotions is needed for the emotion model
num_emotions = len(set(emotion_ds["train"]["labels"]))

# ==========================================
# 2️⃣ Baseline Sentiment Model (Increased Data/Epochs)
# ==========================================
sentiment_model_large = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
).to(device)

sentiment_args_large = TrainingArguments(
    output_dir="./sentiment_results_large",
    num_train_epochs=3, # Increased epochs
    per_device_train_batch_size=16, # Increased batch size
    per_device_eval_batch_size=16, # Increased batch size
    report_to="none"
)

sentiment_trainer_large = Trainer(
    model=sentiment_model_large,
    args=sentiment_args_large,
    train_dataset=sentiment_ds["train"].select(range(10000)), # Increased dataset size
    eval_dataset=sentiment_ds["test"].select(range(2000)) # Increased dataset size
)

print("🧠 Training baseline sentiment model (larger data)...")
sentiment_trainer_large.train()
sentiment_eval_large = sentiment_trainer_large.evaluate()
print("✅ Baseline Sentiment Eval (larger data):", sentiment_eval_large)

# ==========================================
# 3️⃣ Baseline Emotion Model (Increased Data/Epochs)
# ==========================================
# Assuming num_emotions is already defined from the previous cell
emotion_model_large = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_emotions
).to(device)

emotion_args_large = TrainingArguments(
    output_dir="./emotion_results_large",
    num_train_epochs=3, # Increased epochs
    per_device_train_batch_size=16, # Increased batch size
    per_device_eval_batch_size=16, # Increased batch size
    report_to="none"
)

emotion_trainer_large = Trainer(
    model=emotion_model_large,
    args=emotion_args_large,
    train_dataset=emotion_ds["train"].select(range(15000)), # Increased dataset size
    eval_dataset=emotion_ds["test"].select(range(2000)) # Increased dataset size
)

print("🎭 Training baseline emotion model (larger data)...")
emotion_trainer_large.train()
emotion_eval_large = emotion_trainer_large.evaluate()
print("✅ Baseline Emotion Eval (larger data):", emotion_eval_large)

# ==========================================
# 4️⃣ Sarcasm Detection Model (Increased Data/Epochs)
# ==========================================
sarcasm_model_large = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
).to(device)

sarcasm_args_large = TrainingArguments(
    output_dir="./sarcasm_results_large",
    num_train_epochs=3, # Increased epochs
    per_device_train_batch_size=16, # Increased batch size
    per_device_eval_batch_size=16, # Increased batch size
    report_to="none"
)

sarcasm_trainer_large = Trainer(
    model=sarcasm_model_large,
    args=sarcasm_args_large,
    train_dataset=sarcasm_ds_loaded["train"].select(range(10000)), # Increased dataset size
    eval_dataset=sarcasm_ds_loaded["test"].select(range(2000)) # Increased dataset size
)

print("🤨 Training sarcasm detection model (larger data)...")
sarcasm_trainer_large.train()
sarcasm_eval_large = sarcasm_trainer_large.evaluate()
print("✅ Sarcasm Model Eval (larger data):", sarcasm_eval_large)


# ==========================================
# 5️⃣ Define Dual-Branch Fusion Model (Using newly trained sarcasm model)
# ==========================================
class DualTransformerModelLarge(nn.Module):
    def __init__(self, sarcasm_model, num_sent_labels=2, num_emotion_labels=num_emotions):
        super(DualTransformerModelLarge, self).__init__()
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.sarcasm_model = sarcasm_model
        for p in self.sarcasm_model.parameters():
            p.requires_grad = False

        self.sent_fusion = nn.Linear(768 + 2, 768)
        self.emo_fusion = nn.Linear(768 + 2, 768)

        self.sent_classifier = nn.Linear(768, num_sent_labels)
        self.emo_classifier = nn.Linear(768, num_emotion_labels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            sarcasm_logits = self.sarcasm_model(input_ids=input_ids, attention_mask=attention_mask).logits
            sarcasm_probs = torch.softmax(sarcasm_logits, dim=1)

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = text_outputs.last_hidden_state[:, 0, :]

        sent_fused = self.relu(self.sent_fusion(torch.cat((cls_emb, sarcasm_probs), dim=1)))
        emo_fused = self.relu(self.emo_fusion(torch.cat((cls_emb, sarcasm_probs), dim=1)))

        sent_logits = self.sent_classifier(self.dropout(sent_fused))
        emo_logits = self.emo_classifier(self.dropout(emo_fused))

        return sent_logits, emo_logits

# ==========================================
# 6️⃣ Train the Dual-Branch Model (Increased Data/Epochs)
# ==========================================
class CombinedDataset(Dataset):
    def __init__(self, text_data):
        self.text_data = text_data

    def __getitem__(self, idx):
        return {
            "input_ids": self.text_data[idx]["input_ids"],
            "attention_mask": self.text_data[idx]["attention_mask"],
            "sentiment_labels": self.text_data[idx]["labels"], # Renamed for clarity
        }

    def __len__(self):
        return len(self.text_data)

dual_train_large = CombinedDataset(sentiment_ds["train"].select(range(15000))) # Increased dataset size
dual_test_large = CombinedDataset(sentiment_ds["test"].select(range(2000))) # Increased dataset size


fusion_model_large = DualTransformerModelLarge(sarcasm_model_large).to(device)
optimizer_large = torch.optim.AdamW(fusion_model_large.parameters(), lr=2e-5)
loss_fn_large = nn.CrossEntropyLoss()

print("🧩 Training Dual-Branch Fusion Model (larger data)...")
fusion_model_large.train()
for epoch in range(3): # Increased epochs
    total_loss = 0
    for i in range(0, len(dual_train_large), 8): # Manual batching for simplicity
        batch = [dual_train_large[j] for j in range(i, min(i+8, len(dual_train_large)))]
        input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
        attn = torch.stack([item["attention_mask"] for item in batch]).to(device)
        labels = torch.tensor([item["sentiment_labels"] for item in batch]).to(device) # Using renamed labels


        sent_logits, emo_logits = fusion_model_large(input_ids, attn)

        sent_loss = loss_fn_large(sent_logits, labels)
        total = sent_loss # Emotion fine-tuning can be added later

        optimizer_large.zero_grad()
        total.backward()
        optimizer_large.step()
        total_loss += total.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/(len(dual_train_large)/8):.4f}")


# ==========================================
# 7️⃣ Evaluation (Larger Data)
# ==========================================
fusion_model_large.eval()
correct_large = 0
with torch.no_grad():
    for i in range(0, len(dual_test_large), 8):
        batch = [dual_test_large[j] for j in range(i, min(i+8, len(dual_test_large)))]
        input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
        attn = torch.stack([item["attention_mask"] for item in batch]).to(device)
        labels = torch.tensor([item["sentiment_labels"] for item in batch]).to(device) # Using renamed labels


        sent_logits, _ = fusion_model_large(input_ids, attn)
        preds = torch.argmax(sent_logits, dim=1)
        correct_large += (preds == labels).sum().item()

accuracy_large = correct_large / len(dual_test_large)
print(f"✅ Final Dual-Transformer Accuracy (larger data): {accuracy_large*100:.2f}%")