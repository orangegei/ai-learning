from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Continuous-time sinusoidal embedding for t in [0, 1].
    """
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=t.device)
    )
    angles = t * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


@dataclass
class Pi0CustomConfig:
    """
    Minimal, self-written pi0-style model config.
    """

    vocab_size: int
    max_text_len: int = 48
    image_size: int = 128
    n_cams: int = 2
    state_dim: int = 32
    action_dim: int = 7
    action_horizon: int = 16
    hidden_dim: int = 256
    cond_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    image_keys: list[str] = field(
        default_factory=lambda: ["observation.images.cam0", "observation.images.cam1"]
    )


class VisionEncoder(nn.Module):
    """
    Shared camera encoder:
    [B, N, 3, H, W] -> [B, hidden]
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = images.shape
        x = images.view(b * n, c, h, w)
        x = self.backbone(x).squeeze(-1).squeeze(-1)
        x = self.proj(x)
        x = x.view(b, n, -1).mean(dim=1)
        return x


class TextEncoder(nn.Module):
    """
    Token embedding + TransformerEncoder + masked mean pooling.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_text_len: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_text_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers // 2))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        x = self.token_emb(input_ids) + self.pos_emb[:, :seq_len]
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        mask = attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (x * mask).sum(dim=1) / denom
        return pooled


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ActionExpert(nn.Module):
    """
    Flow-matching action denoiser:
    input: noisy action chunk + cond + t
    output: velocity field of same shape as action chunk
    """

    def __init__(self, cfg: Pi0CustomConfig):
        super().__init__()
        self.cfg = cfg
        self.action_in = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.time_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.cond_proj = nn.Linear(cfg.cond_dim, cfg.hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.action_horizon, cfg.hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    def forward(self, noisy_actions: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, h, _ = noisy_actions.shape
        x = self.action_in(noisy_actions)
        time_emb = sinusoidal_time_embedding(t, self.cfg.hidden_dim)
        time_bias = self.time_proj(time_emb).unsqueeze(1).expand(b, h, -1)
        cond_bias = self.cond_proj(cond).unsqueeze(1).expand(b, h, -1)
        x = x + self.pos_emb[:, :h] + time_bias + cond_bias
        x = self.encoder(x)
        velocity = self.out_proj(x)
        return velocity


class Pi0CustomModel(nn.Module):
    """
    Self-written pi0-style model (not using official LeRobot policy implementation).

    Pipeline:
    obs(images, text, state) -> conditioning embedding -> flow denoiser -> action chunk
    """

    def __init__(self, cfg: Pi0CustomConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(out_dim=cfg.hidden_dim)
        self.text_encoder = TextEncoder(
            vocab_size=cfg.vocab_size,
            hidden_dim=cfg.hidden_dim,
            max_text_len=cfg.max_text_len,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        self.state_encoder = StateEncoder(cfg.state_dim, cfg.hidden_dim)
        self.cond_fuse = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3, cfg.cond_dim),
            nn.GELU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
            nn.LayerNorm(cfg.cond_dim),
        )
        self.action_expert = ActionExpert(cfg)

    def encode_observation(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trace: bool = False,
    ) -> torch.Tensor:
        """
        images: [B, N, 3, H, W]
        state: [B, state_dim]
        input_ids / attention_mask: [B, L]
        """
        vision_feat = self.vision_encoder(images)
        text_feat = self.text_encoder(input_ids, attention_mask)
        state_feat = self.state_encoder(state)
        cond = self.cond_fuse(torch.cat([vision_feat, text_feat, state_feat], dim=-1))
        if trace:
            print(f"[trace] vision_feat: {tuple(vision_feat.shape)}")
            print(f"[trace] text_feat: {tuple(text_feat.shape)}")
            print(f"[trace] state_feat: {tuple(state_feat.shape)}")
            print(f"[trace] cond: {tuple(cond.shape)}")
        return cond

    @torch.no_grad()
    def sample_action_chunk(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        denoise_steps: int = 10,
        trace: bool = False,
    ) -> torch.Tensor:
        """
        Euler integrate velocity field from t=0 to t=1.
        """
        cond = self.encode_observation(
            images=images,
            state=state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            trace=trace,
        )

        b = images.shape[0]
        h = self.cfg.action_horizon
        a = self.cfg.action_dim
        x = torch.randn(b, h, a, device=images.device, dtype=images.dtype)
        dt = 1.0 / float(max(1, denoise_steps))

        for i in range(max(1, denoise_steps)):
            t = torch.full((b,), i * dt, device=images.device, dtype=images.dtype)
            v = self.action_expert(x, cond, t)
            x = x + dt * v
            if trace and i in (0, denoise_steps - 1):
                print(f"[trace] denoise_step={i} x={tuple(x.shape)} v={tuple(v.shape)}")
        return x

    @torch.no_grad()
    def select_action(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        denoise_steps: int = 10,
        trace: bool = False,
    ) -> torch.Tensor:
        """
        Returns the first action in the generated action chunk: [B, action_dim].
        """
        chunk = self.sample_action_chunk(
            images=images,
            state=state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            denoise_steps=denoise_steps,
            trace=trace,
        )
        return chunk[:, 0]
