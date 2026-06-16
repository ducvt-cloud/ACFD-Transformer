"""
Model definitions for ACFD-Transformer.

  * ACFDDenoiser / ACFD : the conditional denoising diffusion module
    (Section 3.2, Eqs. 2-4; Table 3). A real DDPM with T=1000 steps, a linear
    beta schedule, a 3-layer MLP denoiser, conditioned on the attack sub-type
    via a learned class embedding plus a sinusoidal time embedding.
  * LongformerAPTDetector : the classification backbone (Section 3.3-3.4;
    Table 3). A HuggingFace Longformer encoder followed by global average
    pooling and a 2-layer MLP head.
"""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import LongformerConfig, LongformerModel


class SinusoidalTimeEmb(nn.Module):
    """Standard sinusoidal embedding of the diffusion timestep t."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class ACFDDenoiser(nn.Module):
    """eps_theta(x_t, t, c): 3-layer MLP (128 units, SiLU), conditioned on
    timestep t and attack sub-type c (Eq. 4)."""
    def __init__(self, x_dim, n_cond, t_dim=64, c_dim=64, hidden=128):
        super().__init__()
        self.t_emb = SinusoidalTimeEmb(t_dim)
        self.c_emb = nn.Embedding(n_cond, c_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + t_dim + c_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x, t, c):
        return self.net(torch.cat([x, self.t_emb(t), self.c_emb(c)], dim=-1))


class ACFD:
    """Adaptive Conditional Feature Diffusion: a conditional DDPM
    (T steps, linear beta schedule). Pre-trained on real minority-class
    windows, then used to sample synthetic minority windows for TRAINING only."""
    def __init__(self, x_dim, n_cond, cfg, device):
        self.cfg = cfg
        self.device = device
        self.T = cfg.t_steps
        self.betas = torch.linspace(cfg.beta_start, cfg.beta_end, self.T, device=device)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)
        self.model = ACFDDenoiser(x_dim, n_cond, hidden=cfg.acfd_hidden).to(device)

    def pretrain(self, X, c, verbose=True):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.acfd_lr)
        dl = DataLoader(TensorDataset(X, c), batch_size=256, shuffle=True)
        self.model.train()
        for ep in range(self.cfg.acfd_epochs):
            tot = 0.0
            for xb, cb in dl:
                xb, cb = xb.to(self.device), cb.to(self.device)
                t = torch.randint(0, self.T, (len(xb),), device=self.device)
                eps = torch.randn_like(xb)
                ab = self.abar[t][:, None]
                xt = ab.sqrt() * xb + (1 - ab).sqrt() * eps
                loss = nn.functional.mse_loss(self.model(xt, t, cb), eps)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            if verbose and (ep + 1) % 20 == 0:
                print(f"      [ACFD] epoch {ep + 1}/{self.cfg.acfd_epochs} "
                      f"noise-MSE={tot / len(dl):.4f}")

    @torch.no_grad()
    def sample(self, n, cond_pool, x_dim, bs=2048):
        """Full reverse diffusion. Conditions are drawn from the real
        sub-type distribution of the minority class."""
        self.model.eval()
        out = []
        for s in range(0, n, bs):
            m = min(bs, n - s)
            x = torch.randn(m, x_dim, device=self.device)
            c = cond_pool[torch.randint(0, len(cond_pool), (m,))].to(self.device)
            for t in reversed(range(self.T)):
                tt = torch.full((m,), t, dtype=torch.long, device=self.device)
                eps = self.model(x, tt, c)
                a, ab, b = self.alphas[t], self.abar[t], self.betas[t]
                x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
                if t > 0:
                    x = x + b.sqrt() * torch.randn_like(x)
            out.append(x.clamp(0, 1).cpu())   # features are Min-Max scaled to [0,1]
        return torch.cat(out)


class LongformerAPTDetector(nn.Module):
    """HuggingFace Longformer (2 layers, 8 heads, d=128). vocab_size is set to a
    minimal value because we feed `inputs_embeds` directly; this avoids the large
    unused word-embedding table. Classification head: global average pooling +
    2-layer MLP (Fig. 1)."""
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.embedding = nn.Linear(input_dim, cfg.embed_dim)
        lf_cfg = LongformerConfig(
            vocab_size=4,
            hidden_size=cfg.embed_dim,
            num_hidden_layers=cfg.n_layers,
            num_attention_heads=cfg.n_heads,
            intermediate_size=cfg.ffn_dim,
            attention_window=[cfg.window] * cfg.n_layers,   # spans the full W
            max_position_embeddings=cfg.window + 8,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, sep_token_id=2,
        )
        self.transformer = LongformerModel(lf_cfg, add_pooling_layer=False)
        self.head = nn.Sequential(
            nn.Linear(cfg.embed_dim, 64), nn.ReLU(),
            nn.Dropout(cfg.dropout), nn.Linear(64, 2),
        )

    def forward(self, x):
        h = self.transformer(inputs_embeds=self.embedding(x)).last_hidden_state
        return self.head(h.mean(dim=1))   # GAP + MLP head
