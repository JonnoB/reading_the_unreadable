from transformers import AutoTokenizer, ModernBertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
import gc
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Model Inference Script")

    # Required paths
    parser.add_argument(
        "--output_dir", required=True, help="Directory for output files"
    )
    parser.add_argument("--model_path", required=True, help="Path to the saved model")

    # Optional parameters
    parser.add_argument(
        "--input_dir",
        type=str,
        default="processed_files/post_processed",
        help="Directory containing input files",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for model inference"
    )
    parser.add_argument(
        "--processing_batch_size",
        type=int,
        default=100000,
        help="Batch size for processing dataframes",
    )
    parser.add_argument(
        "--multi_class",
        type=str,
        default=False,
        help="Whether the model is multi-class",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=512,
        help="Maximum sequence length for the model",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader"
    )

    return parser.parse_args()


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


@torch.no_grad()
def get_predictions(model, dataloader, device, multi_class=False):
    probabilities = []
    predictions = []  # We'll calculate predictions batch by batch
    model.eval()

    for batch in tqdm(dataloader, desc="Processing batches", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        if multi_class:
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).int()
        else:
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        probabilities.extend(probs.cpu().numpy())
        predictions.extend(preds.cpu().numpy())

    return predictions, probabilities


def process_in_batches(
    df,
    tokenizer,
    model,
    device,
    batch_size,
    processing_batch_size,
    multi_class,
    num_workers,
):
    all_predictions = []
    all_probabilities = []

    for i in tqdm(range(0, len(df), processing_batch_size), desc="Processing batches"):
        batch_df = df.iloc[i : i + processing_batch_size]

        encodings = tokenizer(
            batch_df["content"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors=None,
        )

        dataset = CustomDataset(encodings)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

        batch_predictions, batch_probabilities = get_predictions(
            model, dataloader, device, multi_class=multi_class
        )
        all_predictions.extend(batch_predictions)
        all_probabilities.extend(batch_probabilities)

        del encodings, dataset, dataloader
        torch.cuda.empty_cache()
        gc.collect()

    return all_predictions, all_probabilities


def main():
    args = parse_args()

    args.multi_class = args.multi_class.lower() in ["true", "1", "t", "y", "yes"]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision("high")

    # Initialize model and tokenizer
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=args.model_max_length,
        padding_side="right",
        truncation_side="right",
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ModernBertForSequenceClassification.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of files to process
    input_files = list(Path(args.input_dir).glob("*.parquet"))
    files_to_process = []
    files_skipped = []

    for input_file in input_files:
        output_file = Path(args.output_dir) / f"classified_{input_file.name}"
        if output_file.exists():
            files_skipped.append(input_file.name)
        else:
            files_to_process.append(input_file)

    # Print summary
    print(f"\nFound {len(input_files)} total files")
    print(f"Skipping {len(files_skipped)} already processed files")
    print(f"Processing {len(files_to_process)} new files")

    if files_skipped:
        print("\nSkipped files:")
        for file in files_skipped:
            print(f"- {file}")

    # Process files
    for input_file in tqdm(files_to_process, desc="Processing files"):
        print(f"\nProcessing {input_file}...")

        try:
            df = pd.read_parquet(
                input_file, columns=["content", "box_page_id", "page_id"]
            )
            df = df.astype({"content": "string"})

            predictions, probabilities = process_in_batches(
                df=df,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=args.batch_size,
                processing_batch_size=args.processing_batch_size,
                multi_class=args.multi_class,
                num_workers=args.num_workers,
            )

            if args.multi_class:
                # For multi-class, create binary columns for each class
                num_classes = len(probabilities[0])
                for i in range(num_classes):
                    df[f"class_{i}"] = [pred[i] for pred in predictions]
                    df[f"class_{i}_probability"] = [prob[i] for prob in probabilities]
            else:
                # For single-class, create one prediction column and probability columns
                df["predicted_class"] = predictions
                num_classes = len(probabilities[0])
                for i in range(num_classes):
                    df[f"class_{i}_probability"] = [prob[i] for prob in probabilities]

            output_file = Path(args.output_dir) / f"classified_{input_file.name}"

            df["bbox_uid"] =  df["page_id"] + "_" + df["box_page_id"]
            df = df.drop(columns=["content", "box_page_id", "page_id"])
            df.to_parquet(output_file)
            print(f"Saved predictions to {output_file}")

        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            continue

        del df
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

    print("\nProcessing completed!")

    # Final summary
    print("\nFinal Summary:")
    print(f"Successfully processed: {len(files_to_process)} files")
    print(f"Previously processed: {len(files_skipped)} files")
    if files_skipped:
        print("\nSkipped files:")
        for file in files_skipped:
            print(f"- {file}")


if __name__ == "__main__":
    main()
