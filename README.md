# news_classifier

# 사용된 데이터셋: AG-NEWS Dataset

# 토크나이저: DistilBERT tokenizer (vocab_size = 30,522)

# 임베딩 디멘션: 256차원으로 설정 (학습 환경에서 사용 가능한 GPU Vram이 6기가로 보수적으로 설정)

# DistilBERT의 토크나이저 vocabulary를 사용하지만, Transformer encoder 자체는 직접 구현한 custom architecture

# Pretrained model인 DistilBERT를 fine-tuning하지 않음

# 아키텍처
256-dimensional embeddings
4 attention heads
2 Transformer encoder layers
FFN 256 → 1024(Gelu) → 256
RoPE positional encoding
Mask-aware mean pooling
4-class classifier

# 모델
9.39M Trainable parameters

# 학습
Batch size = 16
Epochs = 2
AdamW
LR = 1e-4

# 성능
Train Loss: 0.4614 → 0.2699
Test Accuracy: 90.28%
Macro F1: 90.25%


# 파라미터 분포
Parameter distribution:
embedding.weight                                              7,813,632
encoder.layers.0.norm1.weight                                       256
encoder.layers.0.norm1.bias                                         256
encoder.layers.0.self_attention.q_proj.weight                    65,536
encoder.layers.0.self_attention.q_proj.bias                         256
encoder.layers.0.self_attention.k_proj.weight                    65,536
encoder.layers.0.self_attention.k_proj.bias                         256
encoder.layers.0.self_attention.v_proj.weight                    65,536
encoder.layers.0.self_attention.v_proj.bias                         256
encoder.layers.0.self_attention.out_proj.weight                  65,536
encoder.layers.0.self_attention.out_proj.bias                       256
encoder.layers.0.norm2.weight                                       256
encoder.layers.0.norm2.bias                                         256
encoder.layers.0.feed_forward.linear1.weight                    262,144
encoder.layers.0.feed_forward.linear1.bias                        1,024
encoder.layers.0.feed_forward.linear2.weight                    262,144
encoder.layers.0.feed_forward.linear2.bias                          256
encoder.layers.1.norm1.weight                                       256
encoder.layers.1.norm1.bias                                         256
encoder.layers.1.self_attention.q_proj.weight                    65,536
encoder.layers.1.self_attention.q_proj.bias                         256
encoder.layers.1.self_attention.k_proj.weight                    65,536
encoder.layers.1.self_attention.k_proj.bias                         256
encoder.layers.1.self_attention.v_proj.weight                    65,536
encoder.layers.1.self_attention.v_proj.bias                         256
encoder.layers.1.self_attention.out_proj.weight                  65,536
encoder.layers.1.self_attention.out_proj.bias                       256
encoder.layers.1.norm2.weight                                       256
encoder.layers.1.norm2.bias                                         256
encoder.layers.1.feed_forward.linear1.weight                    262,144
encoder.layers.1.feed_forward.linear1.bias                        1,024
encoder.layers.1.feed_forward.linear2.weight                    262,144
encoder.layers.1.feed_forward.linear2.bias                          256
classifier.classifier.weight                                      1,024
classifier.classifier.bias                                            4

# 9.39M 파라티머 중 약 83%가 vocabulary embedding에 할당
# 실제 Transformer encoder는 약 1.58M 파라티머 차지

# Metric
<img width="418" height="201" alt="image" src="https://github.com/user-attachments/assets/f734b1fa-8701-4670-a00e-77afc645a100" />
