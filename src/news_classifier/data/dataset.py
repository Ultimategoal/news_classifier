from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"


def load_ag_news():
    """Load the AG News dataset from Hugging Face."""
    return load_dataset("fancyzhx/ag_news")


def tokenize_dataset(dataset):
    """Tokenize AG News text using the DistilBERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["text"], # 문자열 리스트 -> batched=True이기 때문
            truncation=True, # max_length=128보다 길면 뒤에 내용 자른다.
            padding="max_length", # 길이가 짧으면 max_length만큼 pad로 채운다.
            max_length=128,
        ),
        batched=True,
    )

    return tokenized_dataset
