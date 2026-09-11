import torch
import torch.nn as nn

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
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2



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

train_dataloader = create_dataloader(
    tokenized_dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
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
# Loss & Optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)



# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(NUM_EPOCHS):
    model.train()

    total_loss = 0.0

    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 500 == 0:
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} "
                f"| Step {step + 1} "
                f"| Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_dataloader)

    print(
        f"Epoch {epoch + 1} finished "
        f"| Average Loss: {average_loss:.4f}"
    )

# --------------------------------------------------
# Save model checkpoint
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "news_classifier.pt",
)

print("Model saved: news_classifier.pt")
