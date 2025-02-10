import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load and prepare data
df = pd.read_parquet("silver_IPTC_class.parquet")
df = df.dropna(subset=["class_code"])

# Ensure class_code is a list for each entry
df["class_code"] = df["class_code"].apply(lambda x: [x] if isinstance(x, int) else x)

# Convert class_code lists to binary labels
mlb = MultiLabelBinarizer()
labels_matrix = mlb.fit_transform(df["class_code"])
df["labels"] = [
    labels.astype(float).tolist() for labels in labels_matrix
]  # Convert to float

# Create id mappings
id2label = {i: str(label) for i, label in enumerate(mlb.classes_)}
label2id = {str(label): i for i, label in enumerate(mlb.classes_)}
number_of_labels = len(mlb.classes_)

# Split data
train_df = df[df["is_train"]]
test_df = df[~df["is_train"]]

# Create datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

# Initialize tokenizer
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    # Tokenize the texts
    tokenized = tokenizer(
        examples["content"], padding=True, truncation=True, max_length=512
    )

    # Add labels
    tokenized["labels"] = examples["labels"]
    return tokenized


# Tokenize and preprocess the dataset
tokenized_dataset = dataset_dict.map(
    preprocess_function, batched=True, remove_columns=dataset_dict["train"].column_names
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Metrics calculation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int)

    # Calculate metrics
    accuracy = (predictions == labels).mean()

    # Calculate other metrics per class
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for i in range(labels.shape[1]):
        true_positives = ((predictions[:, i] == 1) & (labels[:, i] == 1)).sum()
        false_positives = ((predictions[:, i] == 1) & (labels[:, i] == 0)).sum()
        false_negatives = ((predictions[:, i] == 0) & (labels[:, i] == 1)).sum()

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "f1": np.mean(f1_scores),
        "precision": np.mean(precision_scores),
        "recall": np.mean(recall_scores),
    }


# Enable TensorFloat32 for faster computation
torch.set_float32_matmul_precision("high")
# Initialize model
model = ModernBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=number_of_labels,
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification",
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_IPTC",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train model
print("\nStarting training...")
trainer.train()
print("\nTraining completed!")

# Save model
trainer.save_model("./IPTC_type_saved_model")

# Final evaluation
results = trainer.evaluate()
print("\nFinal evaluation results:")
print(results)
