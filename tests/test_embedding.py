import torch
import torch.nn as nn

from news_classifier.data.dataset import load_ag_news, tokenize_dataset
from news_classifier.data.dataloader import create_dataloader
from news_classifier.model import (
    TokenEmbedding,
    RotaryPositionalEmbedding,
    QKVProjection,
    MultiHeadReshape,
    MultiHeadConcat,
    AttentionOutputProjection,
    MultiHeadSelfAttention,
    TransformerLayerNorm,
    ResidualConnection,
    FeedForward,
    TransformerEncoderLayer,
    TransformerEncoder,
    MeanPooling,
    ClassificationHead,
    NewsClassifier,
    VOCAB_SIZE,
    NUM_CLASSES,
    FFN_DIM,
    EMBED_DIM,
    HEAD_DIM,
    NUM_HEADS,
)

dataset = load_ag_news()
tokenized_dataset = tokenize_dataset(dataset)

train_dataloader = create_dataloader(
    tokenized_dataset["train"],
    batch_size=16,
)

embedding_model = TokenEmbedding()

print(embedding_model)

batch = next(iter(train_dataloader))

print("input_ids:", batch["input_ids"].shape)

embeddings = embedding_model(batch["input_ids"])

print("embeddings:", embeddings.shape)

# --------------------------------------------------
# QKV Projection test
# --------------------------------------------------

qkv_projection = QKVProjection(EMBED_DIM)

q, k, v = qkv_projection(embeddings)

print("Q:", q.shape)
print("K:", k.shape)
print("V:", v.shape)

# --------------------------------------------------
# Multi-Head reshape test
# --------------------------------------------------

multi_head_reshape = MultiHeadReshape(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
)

q_heads = multi_head_reshape(q)
k_heads = multi_head_reshape(k)
v_heads = multi_head_reshape(v)

print("Q heads:", q_heads.shape)
print("K heads:", k_heads.shape)
print("V heads:", v_heads.shape)


# --------------------------------------------------
# RoPE on Q and K
# --------------------------------------------------

rope = RotaryPositionalEmbedding(
    head_dim=HEAD_DIM,
    max_seq_len=128,
)

q_rotated = rope(q_heads)
k_rotated = rope(k_heads)

print("Q before RoPE:", q_heads.shape)
print("Q after RoPE:", q_rotated.shape)

print("K before RoPE:", k_heads.shape)
print("K after RoPE:", k_rotated.shape)

# --------------------------------------------------
# Attention Score: QK^T
# --------------------------------------------------

attention_scores = torch.matmul(
    q_rotated,
    k_rotated.transpose(-2, -1), # shape보면 128, 64인데 이 부분만 64, 128로 바꿔야 하기 때문에 (-2, -1) 들어간듯
)

print("Attention scores:", attention_scores.shape)
scaled_scores = attention_scores / (HEAD_DIM ** 0.5)
print("scaled scores:", scaled_scores.shape)
# --------------------------------------------------
# Softmax
# --------------------------------------------------

attention_weights = torch.softmax(
    scaled_scores,
    dim=-1, # [16, 4, 128, 128] dim=1이 맨 마지막 128에 대해서 하겠다는 의미다.
)

print("Attention weights:", attention_weights.shape)

print(
    "Attention weight sum:",
    attention_weights[0, 0, 0].sum().item(),
)

# --------------------------------------------------
# Attention output
# --------------------------------------------------

attention_output = torch.matmul(
    attention_weights,
    v_heads,
)

print("Attention output:", attention_output.shape)


# --------------------------------------------------
# Concatenate attention heads
# --------------------------------------------------

multi_head_concat = MultiHeadConcat(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
)

concatenated_output = multi_head_concat(
    attention_output
)

print("Concatenated output:", concatenated_output.shape)

# --------------------------------------------------
# Output projection
# --------------------------------------------------

output_projection = AttentionOutputProjection(
    embed_dim=EMBED_DIM,
)

attention_output_projected = output_projection(
    concatenated_output
)

print(
    "Projected attention output:",
    attention_output_projected.shape,
)

# --------------------------------------------------
# Multi-Head Self-Attention test
# --------------------------------------------------

self_attention = MultiHeadSelfAttention(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    max_seq_len=128,
)

attention_output = self_attention(embeddings)

print(
    "Self-attention output:",
    attention_output.shape,
)
# --------------------------------------------------
# LayerNorm test
# --------------------------------------------------

layer_norm = TransformerLayerNorm(
    embed_dim=EMBED_DIM,
)

normalized_output = layer_norm(
    attention_output_projected
)

print(
    "LayerNorm output:",
    normalized_output.shape,
)

# --------------------------------------------------
# Residual connection test
# --------------------------------------------------

residual = ResidualConnection()

residual_output = residual(
    embeddings,
    attention_output_projected,
)

print(
    "Residual output:",
    residual_output.shape,
)

# --------------------------------------------------
# Feed-Forward Network test
# --------------------------------------------------

feed_forward = FeedForward(
    embed_dim=EMBED_DIM,
    ffn_dim=FFN_DIM,
)

ffn_output = feed_forward(
    residual_output
)

print(
    "FFN output:",
    ffn_output.shape,
)

# --------------------------------------------------
# Second residual connection
# --------------------------------------------------

final_output = residual(
    residual_output,
    ffn_output,
)

print(
    "Transformer block output:",
    final_output.shape,
)

# --------------------------------------------------
# Transformer Encoder Layer test
# --------------------------------------------------

encoder_layer = TransformerEncoderLayer(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM,
    max_seq_len=128,
)

encoder_output = encoder_layer(
    embeddings,
    attention_mask=batch["attention_mask"],
)

print(
    "Encoder layer output:",
    encoder_output.shape,
)

# --------------------------------------------------
# Transformer Encoder test
# --------------------------------------------------

encoder = TransformerEncoder(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM,
    num_layers=2,
    max_seq_len=128,
)

encoder_output = encoder(
    embeddings,
    attention_mask=batch["attention_mask"],
)

print(
    "Encoder output:",
    encoder_output.shape,
)

# --------------------------------------------------
# Mean Pooling test
# --------------------------------------------------

mean_pooling = MeanPooling()

pooled_output = mean_pooling(
    encoder_output,
    batch["attention_mask"],
)

print(
    "Pooled output:",
    pooled_output.shape,
)

# --------------------------------------------------
# Classification Head test
# --------------------------------------------------

classifier = ClassificationHead(
    embed_dim=EMBED_DIM,
    num_classes=NUM_CLASSES,
)

logits = classifier(
    pooled_output
)

print(
    "Logits:",
    logits.shape,
)

print(
    "Predicted labels:",
    logits.argmax(dim=-1)
)

# --------------------------------------------------
# Full NewsClassifier test
# --------------------------------------------------

model = NewsClassifier(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM,
    num_layers=2,
    num_classes=NUM_CLASSES,
    max_seq_len=128,
)

logits = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
)

print(
    "NewsClassifier logits:",
    logits.shape,
)

# --------------------------------------------------
# Loss + Backpropagation test
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

loss = criterion(
    logits,
    batch["labels"],
)

print("Loss:", loss.item())

loss.backward()

print("Backward pass: OK")
