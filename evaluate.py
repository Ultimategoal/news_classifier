import torch

import torch
import os
os.makedirs("results", exist_ok=True)

import csv

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from news_classifier.data.dataset import (
    load_ag_news,
    tokenize_dataset,
)

from news_classifier.data.dataloader import (
    create_dataloader,
)

from news_classifier.model import (
    NewsClassifier,
    VOCAB_SIZE,
    EMBED_DIM,
    NUM_HEADS,
    FFN_DIM,
    NUM_CLASSES,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 16
NUM_LAYERS = 2
MAX_SEQ_LEN = 128


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = load_ag_news()
tokenized_dataset = tokenize_dataset(dataset)

test_dataloader = create_dataloader(
    tokenized_dataset["test"],
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = NewsClassifier(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    max_seq_len=MAX_SEQ_LEN,
)

model = model.to(device)

# --------------------------------------------------
# Parameter Count
# --------------------------------------------------

total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(
    f"Total parameters: {total_params:,}"
)

print(
    f"Trainable parameters: {trainable_params:,}"
)

# --------------------------------------------------
# Parameter Distribution
# --------------------------------------------------

print("\nParameter distribution:")

for name, parameter in model.named_parameters():
    print(
        f"{name:60s} "
        f"{parameter.numel():>10,}"
    )

# --------------------------------------------------
# Save Parameter Distribution
# --------------------------------------------------

with open(
    "results/parameter_distribution.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write("Parameter Distribution\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        f"Total parameters: {total_params:,}\n"
    )

    f.write(
        f"Trainable parameters: {trainable_params:,}\n\n"
    )

    f.write("Parameters by tensor\n")
    f.write("-" * 80 + "\n")

    for name, parameter in model.named_parameters():

        count = parameter.numel()

        f.write(
            f"{name:60s} "
            f"{count:>10,}\n"
        )

    f.write("\n")
    f.write("Parameters by top-level module\n")
    f.write("-" * 80 + "\n")

    for name, count in module_params.items():

        percentage = (
            count / total_params * 100
        )

        f.write(
            f"{name:20s} "
            f"{count:>10,} "
            f"({percentage:6.2f}%)\n"
        )

print(
    "Parameter distribution saved: "
    "results/parameter_distribution.txt"
)



# --------------------------------------------------
# Load trained weights
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        "news_classifier.pt",
        map_location=device,
    )
)

print("Model loaded.")

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

model.eval()

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch in test_dataloader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        predictions = logits.argmax(dim=-1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        # CPU로 가져와서 저장
        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )


accuracy = correct / total

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
)

print("Confusion Matrix:")
print(cm)


# --------------------------------------------------
# Save Confusion Matrix
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

with open(
    "results/confusion_matrix.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Actual / Predicted",
        "World",
        "Sports",
        "Business",
        "Sci/Tech",
    ])

    for label, row in enumerate(cm):
        writer.writerow([
            label,
            *row,
        ])

print("Confusion matrix saved: results/confusion_matrix.csv")


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

class_names = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
]

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
)

print("Classification Report:")
print(report)

with open(
    "results/classification_report.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(report)

print(
    "Classification report saved: "
    "results/classification_report.txt"
)
