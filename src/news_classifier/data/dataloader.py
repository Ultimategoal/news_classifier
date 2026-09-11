import torch
from torch.utils.data import DataLoader


def create_dataloader(dataset, batch_size=16, shuffle=False):
    """Create a PyTorch DataLoader from a tokenized Hugging Face dataset."""

    dataset = dataset.remove_columns(["text"])
    dataset = dataset.rename_column("label", "labels")

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        # attention_mask 실제 토큰인지 pad 토큰인지 구분 1과 0으로
    )

    # 여기 밑에 text도 들어가지만 모델이 원래 text를 사용하지 않기 때문에 제거
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, # 훈련용 데이터는 True로 주고 예측용은 False로 보통 줌
    )

