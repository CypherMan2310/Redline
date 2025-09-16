# ========================
# 1. Imports
# ========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# ========================
# 2. Load CSVs
# ========================
train_1mg = pd.read_csv("C:/Users/somas/PycharmProjects/Redline/data/IHQID-1mg/train.csv")
test_1mg = pd.read_csv("C:/Users/somas/PycharmProjects/Redline/data/IHQID-1mg/test.csv")

train_webmd = pd.read_csv("C:/Users/somas/PycharmProjects/Redline/data/IHQID-webmd/train.csv")
test_webmd = pd.read_csv("C:/Users/somas/PycharmProjects/Redline/data/IHQID-webmd/test.csv")

# Rename columns
for df in [train_1mg, test_1mg, train_webmd, test_webmd]:
    df.rename(columns={"question_english": "text", "Manual_Intent": "label"}, inplace=True)

# Merge datasets
train_df = pd.concat([train_1mg, train_webmd], ignore_index=True)
test_df = pd.concat([test_1mg, test_webmd], ignore_index=True)

print("Train sample:\n", train_df.head())
print("Test sample:\n", test_df.head())

# ========================
# 3. Visualize Intent Distribution
# ========================
plt.figure(figsize=(10, 6))
sns.countplot(y=train_df["label"], order=train_df["label"].value_counts().index, palette="viridis")
plt.title("Intent Distribution in Training Data")
plt.xlabel("Count")
plt.ylabel("Intent")
plt.show()

# ========================
# 4. Encode Labels
# ========================
labels = sorted(train_df["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

train_df["label"] = train_df["label"].map(label2id)
test_df["label"] = test_df["label"].map(label2id)

# ========================
# 5. HuggingFace Dataset
# ========================
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# ========================
# 6. Tokenizer & Model
# ========================
model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ========================
# 7. Metrics Function
# ========================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ========================
# 8. Training Arguments
# ========================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",   # evaluates at end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# ========================
# 9. Trainer
# ========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ========================
# 10. Train
# ========================
trainer.train()

# ========================
# 11. Evaluate
# ========================
results = trainer.evaluate()
print("Final Evaluation:", results)

# ========================
# 12. Confusion Matrix
# ========================
preds = trainer.predict(tokenized_datasets["test"])
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

cm = confusion_matrix(y_true, y_pred, labels=list(id2label.keys()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id2label.values()))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - IndicBERT Fine-tuned")
plt.show()

# ========================
# 13. Training Curves
# ========================
logs = trainer.state.log_history

train_loss = [x["loss"] for x in logs if "loss" in x]
eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
epochs = range(1, len(eval_loss) + 1)

plt.plot(epochs, train_loss[:len(eval_loss)], label="Training Loss")
plt.plot(epochs, eval_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()
