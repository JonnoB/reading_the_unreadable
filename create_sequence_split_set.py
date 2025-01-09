import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""This script creates the training and test set for the sequence to sequnce splitter""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from datasets import Dataset
    import random
    import os
    import glob
    from typing import List, Tuple
    import re

    def load_text_files(folder_path: str) -> List[str]:
        """Load all text files from the specified folder."""
        files = glob.glob(os.path.join(folder_path, "*.txt"))
        texts = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        return texts

    def split_into_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines and filter out empty paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        return paragraphs

    def get_adjacent_paragraphs(text: str) -> Tuple[str, str]:
        """Get two adjacent paragraphs from a text."""
        paragraphs = split_into_paragraphs(text)
        if len(paragraphs) < 2:
            return None
        
        # Randomly select an index for the first paragraph
        start_idx = random.randint(0, len(paragraphs) - 2)
        return paragraphs[start_idx], paragraphs[start_idx + 1]

    def create_dataset(folder_path: str, num_samples: int = 5000):
        """Create a dataset with both pure and combined samples."""
        # Load all texts
        texts = load_text_files(folder_path)
        
        # Initialize lists for dataset
        all_samples = []
        
        # Create pure samples (adjacent paragraphs from same text)
        pure_samples = []
        while len(pure_samples) < num_samples:
            text = random.choice(texts)
            result = get_adjacent_paragraphs(text)
            if result:
                para1, para2 = result
                pure_samples.append({
                    "text1": para1,
                    "text2": para2,
                    "is_same_text": 1
                })
        
        # Create combined samples (paragraphs from different texts)
        combined_samples = []
        while len(combined_samples) < num_samples:
            text1, text2 = random.sample(texts, 2)
            paras1 = split_into_paragraphs(text1)
            paras2 = split_into_paragraphs(text2)
            
            if paras1 and paras2:  # If both texts have paragraphs
                para1 = random.choice(paras1)
                para2 = random.choice(paras2)
                combined_samples.append({
                    "text1": para1,
                    "text2": para2,
                    "is_same_text": 0
                })
        
        # Combine all samples
        all_samples = pure_samples + combined_samples
        
        # Shuffle the samples
        random.shuffle(all_samples)
        
        # Create Hugging Face dataset
        dataset = Dataset.from_list(all_samples)
        
        return dataset

    return (
        Dataset,
        List,
        Tuple,
        create_dataset,
        get_adjacent_paragraphs,
        glob,
        load_text_files,
        os,
        random,
        re,
        split_into_paragraphs,
    )


@app.cell
def _(create_dataset):
    # Usage
    folder_path = 'data/synthetic_articles_text'


    dataset = create_dataset(folder_path)

    # Split into train/validation/test
    splits = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    test_valid = splits['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)

    final_dataset = {
        'train': splits['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    }

    # Save dataset to Hugging Face format
    final_dataset['train'].save_to_disk("dataset/train")
    final_dataset['validation'].save_to_disk("dataset/validation")
    final_dataset['test'].save_to_disk("dataset/test")

    # Print some statistics
    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(final_dataset['train'])}")
    print(f"Validation samples: {len(final_dataset['validation'])}")
    print(f"Test samples: {len(final_dataset['test'])}")
    return dataset, final_dataset, folder_path, splits, test_valid


@app.cell
def _(final_dataset):
    final_dataset['train'][0]
    return


@app.cell
def _(final_dataset):
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification, 
        TrainingArguments, 
        Trainer
    )
    from datasets import load_from_disk
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def tokenize_function(examples):
        return tokenizer(
            examples["text1"],
            examples["text2"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }



    # Load tokenizer and tokenize data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_datasets = final_dataset.map(tokenize_function, batched=True)

    # Convert to pytorch format
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=2
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate on test set
    results = trainer.evaluate(tokenized_datasets["test"])
    print(results)
    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        accuracy_score,
        compute_metrics,
        load_from_disk,
        model,
        np,
        precision_recall_fscore_support,
        results,
        tokenize_function,
        tokenized_datasets,
        tokenizer,
        trainer,
        training_args,
    )


if __name__ == "__main__":
    app.run()
