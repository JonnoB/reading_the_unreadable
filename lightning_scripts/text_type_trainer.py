import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    ModernBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load and prepare data
df = pd.read_parquet("silver_text_type.parquet")
df = df.dropna(subset=["class_code2"])
df["labels"] = df["class_code2"].astype(int)

number_of_labels = df["labels"].nunique()
train_df = df[df["is_train"]]
test_df = df[~df["is_train"]]

# Create datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

# Initialize tokenizer
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples["content"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors=None,
    )


# Prepare datasets
tokenized_datasets = dataset_dict.map(
    tokenize_function,
    batched=True,
    remove_columns=[
        col for col in dataset_dict["train"].column_names if col not in ["labels"]
    ],
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels"]
)

# Enable TensorFloat32 for faster computation
torch.set_float32_matmul_precision("high")

# Initialize model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ModernBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=number_of_labels,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    fp16=True,
    fp16_opt_level="O1",
    half_precision_backend="auto",
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    logging_dir="./logs",  # Directory for storing logs
    logging_strategy="epoch",
    report_to=["tensorboard"],
)


# Evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Train model
print("\nStarting training...")
trainer.train()
print("\nTraining completed!")

# Save model
output_dir = "./text_type_saved_model"
trainer.save_model(output_dir)

# Final evaluation
results = trainer.evaluate(tokenized_datasets["test"])
print("\nFinal evaluation results:")
print(results)
