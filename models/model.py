import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel

class ACFDModule(nn.Module):
    """
    Adaptive Conditional Feature Diffusion (ACFD) for APT data synthesis.
    Implements the conditional denoising network as described in Figure 2.
    """
    def __init__(self, input_dim, cond_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x_t, t_emb, condition):
        # Concatenate noisy input with time embedding and class guidance
        inputs = torch.cat([x_t, t_emb, condition], dim=-1)
        return self.network(inputs)

class LongformerAPTDetector(nn.Module):
    """
    Optimized Longformer model with 0.84M parameters.
    Utilizes Sliding Window Attention (W=10) for temporal dependencies.
    """
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Configuration matching Table 4 of the manuscript
        self.config = LongformerConfig(
            hidden_size=embed_dim,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=512,
            attention_window=[10, 10], 
            max_position_embeddings=512
        )
        self.transformer = LongformerModel(self.config)
        self.classifier = nn.Linear(embed_dim, 2) 

    def forward(self, x):
        # Input shape: (Batch, Seq_Len, Features)
        x = self.embedding(x)
        outputs = self.transformer(inputs_embeds=x).last_hidden_state
        # Global Average Pooling for sequence classification
        return self.classifier(outputs.mean(dim=1))
