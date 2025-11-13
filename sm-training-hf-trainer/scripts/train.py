import os
import argparse
import json
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


def parse_args():
    parser = argparse.ArgumentParser()

    # Model training hyperparameters
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--num_train_epochs", type=int, default=1)

    """
    # examples of other optional params, depends on model
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    """

    # SageMaker-specific dirs
    parser.add_argument(
        "--output_data_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )

    # Input channels (SageMaker sets these env vars)
    parser.add_argument(
        "--train_channel",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
    )
    parser.add_argument(
        "--test_channel",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST"),
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # --- 1. Resolve CSV paths from channels ---
    train_csv = os.path.join(args.train_channel, "train.csv")
    test_csv = os.path.join(args.test_channel, "test.csv")

    data_files = {
        "train": train_csv,
        "test": test_csv,
    }

    # --- 2. Load CSVs as a HuggingFace DatasetDict ---
    # Expects columns: "text" and "label"
    raw_datasets = load_dataset("csv", data_files=data_files)

    # --- 3. Load tokenizer & model ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,  # IMDb: positive / negative
    )

    # --- 4. Tokenization function ---
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Apply to whole DatasetDict
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],  # keep label + tokenized features
    )

    # Set torch format for Trainer
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    # split datasets
    tokenized_train = tokenized_datasets["train"]
    tokenized_eval = tokenized_datasets["test"]

    # --- 5. Metrics ---
    metric = evaluate.load("accuracy")

    # eval function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # --- 6. TrainingArguments ---
    training_args = TrainingArguments(output_dir=args.model_dir,
                                      num_train_epochs=args.num_train_epochs)

    # --- 7. Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval, #using test as eval
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # --- 8. Train ---
    train_result = trainer.train()

    # --- 9. Save model to SM_MODEL_DIR ---
    trainer.save_model(args.model_dir)

    # --- 10. Save metrics to SM_OUTPUT_DATA_DIR ---
    os.makedirs(args.output_data_dir, exist_ok=True)

    metrics = train_result.metrics
    eval_metrics = trainer.evaluate()
    metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    with open(os.path.join(args.output_data_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
