import torch
from transformers import AutoTokenizer

from news_classifier.model import NewsClassifier


MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "news_classifier.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = NewsClassifier(
    vocab_size=30522,
    embed_dim=256,
    num_heads=4,
    ffn_dim=1024,
    num_layers=2,
    num_classes=4,
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

model.to(DEVICE)
model.eval()


# --------------------------------------------------
# Label mapping
# --------------------------------------------------

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict(text):

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():

        logits = model(
            input_ids,
            attention_mask,
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=-1,
        ).item()

    return {
        "label": LABELS[predicted_class],
        "confidence": probabilities[
            0,
            predicted_class
        ].item(),
    }


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    text = (
        "Arsenal transfer news: Martin Zubimendi may push to leave Premier League champions - Paper Talk"
    )

    result = predict(text)

    print("Text:", text)
    print("Prediction:", result["label"])
    print(
        f"Confidence: "
        f"{result['confidence']:.4f}"
    )
