"""Fine-tune GPT-2 to tag legal contract clauses (Milestone 3).

    python -m scripts.train_classifier                 # trains and saves to models/clause-gpt2
    python -m scripts.train_classifier --epochs 10 --output models/my-model

The first run downloads the base ``gpt2`` weights (~500 MB) from Hugging Face, so it needs
internet. Training 108 short sentences takes a few minutes on a normal laptop CPU.
The saved folder is then picked up automatically by the game (CLASSIFIER_DIR in .env).
"""

from __future__ import annotations

import argparse
import random

from app.corpus import LABELED_CLAUSES, LABELS


def stratified_split(items, val_fraction: float, seed: int):
    """Split (label, text) pairs so every label appears in both train and validation."""
    rng = random.Random(seed)
    by_label: dict[str, list] = {}
    for label, text in items:
        by_label.setdefault(label, []).append((label, text))
    train, val = [], []
    for group in by_label.values():
        rng.shuffle(group)
        cut = max(1, round(len(group) * val_fraction))
        val.extend(group[:cut])
        train.extend(group[cut:])
    rng.shuffle(train)
    return train, val


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument("--output", default="models/clause-gpt2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    train, val = stratified_split(LABELED_CLAUSES, args.val_fraction, args.seed)
    print(f"Training on {len(train)} sentences, validating on {len(val)}.")

    def encode(batch):
        enc = tokenizer([t for _, t in batch], return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc["labels"] = torch.tensor([label2id[l] for l, _ in batch])
        return enc

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def evaluate() -> float:
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(val), args.batch_size):
                batch = encode(val[i : i + args.batch_size])
                labels = batch.pop("labels")
                correct += (model(**batch).logits.argmax(-1) == labels).sum().item()
        return correct / len(val)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        total_loss = 0.0
        for i in range(0, len(train), args.batch_size):
            batch = encode(train[i : i + args.batch_size])
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        steps = (len(train) + args.batch_size - 1) // args.batch_size
        print(f"epoch {epoch}/{args.epochs}  loss {total_loss / steps:.4f}  val accuracy {evaluate():.0%}")

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved fine-tuned model to {args.output}")


if __name__ == "__main__":
    main()
