import torch
import torch.nn as nn

# 정리
# vocab_size = 30,522
# embed_dim  = 256
# heads      = 4
# head_dim   = 64
# ffn_dim    = 1,024
# layers     = 2
# classes    = 4
# seq_len    = 128
# batch      = 16


VOCAB_SIZE = 30522
EMBED_DIM = 256
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
MAX_SEQ_LEN = 128


class TokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBED_DIM,
        )

    def forward(self, input_ids):
        return self.embedding(input_ids)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=128, base=10000):
        super().__init__()

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2).float()
                / head_dim
            )
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        x shape:
        [batch_size, num_heads, seq_len, head_dim]
        """

        seq_len = x.size(2)

        positions = torch.arange(
            seq_len,
            device=x.device,
            dtype=self.inv_freq.dtype,
        )

        freqs = torch.outer(positions, self.inv_freq)

        cos = freqs.cos()
        sin = freqs.sin()

        # [seq_len, 32]
        # ↓
        # [1, 1, seq_len, 32]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split even and odd dimensions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply rotation
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Interleave even and odd dimensions again
        x_rotated = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(-2)

        return x_rotated

class QKVProjection(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x shape:
        [batch_size, seq_len, embed_dim]
        """

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        return q, k, v


class MultiHeadReshape(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, x):
        """
        x shape:
        [batch_size, seq_len, embed_dim]

        output shape:
        [batch_size, num_heads, seq_len, head_dim]
        """

        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        x = x.transpose(1, 2)

        return x

class MultiHeadConcat(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, x):
        """
        x shape:
        [batch_size, num_heads, seq_len, head_dim]

        output shape:
        [batch_size, seq_len, embed_dim]
        """

        batch_size, num_heads, seq_len, head_dim = x.shape

        x = x.transpose(1, 2)

        x = x.contiguous().view(
            batch_size,
            seq_len,
            self.embed_dim,
        )

        return x

class AttentionOutputProjection(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

    def forward(self, x):
        """
        x shape:
        [batch_size, seq_len, embed_dim]
        """

        return self.projection(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        max_seq_len=128,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
        )

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim,
        )

    # def forward(self, x):
    def forward(self, x, attention_mask=None):
        """
        x:
        [batch_size, seq_len, embed_dim]
        """

        batch_size, seq_len, _ = x.shape

        # 1. Q, K, V projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Split into multiple heads
        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # 3. Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)

        # 4. Attention scores
        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        # attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

            attention_scores = attention_scores.masked_fill(
                attention_mask == 0,
                torch.finfo(attention_scores.dtype).min,
            )


        # 5. Scale
        attention_scores = attention_scores / (
            self.head_dim ** 0.5
        )

        # 6. Softmax
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        # 7. Weighted sum of V
        attention_output = torch.matmul(
            attention_weights,
            v,
        )

        # 8. Concatenate heads
        attention_output = attention_output.transpose(1, 2)

        attention_output = attention_output.contiguous().view(
            batch_size,
            seq_len,
            self.embed_dim,
        )

        # 9. Output projection
        output = self.out_proj(attention_output)

        return output

class TransformerLayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x shape:
        [batch_size, seq_len, embed_dim]
        """

        return self.norm(x)

class ResidualConnection(nn.Module):
    def forward(self, x, sublayer_output):
        return x + sublayer_output

FFN_DIM = 1024 # FFN hidden dimension
# 트랜스포머에서 보통 4 X d_model로 잡음


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()

        self.linear1 = nn.Linear(
            embed_dim,
            ffn_dim,
        )

        self.activation = nn.GELU()

        self.linear2 = nn.Linear(
            ffn_dim,
            embed_dim,
        )

    def forward(self, x):
        """
        x shape:
        [batch_size, seq_len, embed_dim]
        """

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_dim,
        max_seq_len=128,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.self_attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
        )

    # def forward(self, x):
    def forward(self, x, attention_mask=None):
        # Pre-LN Self-Attention + Residual
        x = x + self.self_attention(
            self.norm1(x),
            attention_mask=attention_mask,
        )

        # Pre-LN FFN + Residual
        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_dim,
        num_layers,
        max_seq_len=128,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(
                x,
                attention_mask=attention_mask,
            )

        return x

class MeanPooling(nn.Module):
    def forward(self, x, attention_mask):
        """
        x:
        [batch_size, seq_len, embed_dim]

        attention_mask:
        [batch_size, seq_len]
        """

        mask = attention_mask.unsqueeze(-1).float()

        # Padding token의 representation을 0으로 만든다.
        masked_x = x * mask

        # 실제 token들의 합
        summed = masked_x.sum(dim=1)

        # 실제 token 개수
        count = mask.sum(dim=1).clamp(min=1e-9)

        # 평균
        pooled = summed / count

        return pooled


NUM_CLASSES = 4


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()

        self.classifier = nn.Linear(
            embed_dim,
            num_classes,
        )

    def forward(self, x):
        """
        x shape:
        [batch_size, embed_dim]
        """

        return self.classifier(x)


class NewsClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        ffn_dim,
        num_layers,
        num_classes,
        max_seq_len=128,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )

        self.pooling = MeanPooling()

        self.classifier = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
        )

    def forward(self, input_ids, attention_mask):
        """
        input_ids:
        [batch_size, seq_len]

        attention_mask:
        [batch_size, seq_len]
        """

        # 1. Token Embedding
        x = self.embedding(input_ids)

        # 2. Transformer Encoder
        x = self.encoder(
            x,
            attention_mask=attention_mask,
        )

        # 3. Mean Pooling
        x = self.pooling(
            x,
            attention_mask,
        )

        # 4. Classification
        logits = self.classifier(x)

        return logits
