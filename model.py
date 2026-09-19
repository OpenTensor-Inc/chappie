"""
Chappie-aleph-v1: A Multi-Layer Self-Improving Architecture
============================================================

Complete model definition with BUILT-IN parallelization support:
    - Tensor Parallelism (TP) via parallelize_module
    - FSDP2 (fully_shard)
    - Activation Checkpointing (AC)
    - torch.compile integration

All layers included:
    - NeuralODECore (Layer 1)
    - DiffusionHead + GANHead (Layer 2)
    - Analyzer (Layer 3)
    - CostEngine (Layer 4)
    - RSS (Layer 5)

Author: OpenTensor Research
Codename: MegaMan
Version: 0.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# TorchTitan parallelization primitives
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

# TorchTitan internal
from torchtitan.distributed import ParallelDims
from torchtitan.config.job_config import JobConfig


# ============================================================
# Configuration
# ============================================================

@dataclass
class ChappieConfig:
    # --- Core (Layer 1) ---
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_branches: int = 4
    k_active: int = 2
    ode_steps: int = 4
    ode_dt: float = 0.25
    max_seq_len: int = 1024

    # --- Diffusion Head (Layer 2) ---
    diffusion_dim: int = 512
    diffusion_timesteps: int = 100

    # --- GAN Head (Layer 2) ---
    gan_hidden: int = 512

    # --- Analyzer (Layer 3) ---
    analyzer_state_dim: int = 16
    analyzer_control_dim: int = 12
    analyzer_d_model: int = 128
    analyzer_layers: int = 4
    analyzer_heads: int = 4

    # --- Cost Engine (Layer 4) ---
    cost_input_dim: int = 24
    cost_hidden: int = 128

    # --- RSS (Layer 5) ---
    rss_d_model: int = 64
    rss_layers: int = 2
    rss_confidence_threshold: float = 0.7

    # --- Training ---
    dropout: float = 0.1
    tie_embeddings: bool = True

    # --- Parallelism ---
    tp_degree: int = 1
    fsdp_degree: int = 1
    enable_ac: bool = True
    ac_mode: str = "full"  # "full", "selective", "none"
    enable_compile: bool = True


# ============================================================
# Layer 1: Neural ODE Core with Multi-Branch + MoE
# ============================================================

class ODEBranch(nn.Module):
    """A single branch of the Neural ODE."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(h) + u)


class MoERouter(nn.Module):
    """Top-k router for Mixture-of-Experts over branches."""

    def __init__(self, d_model: int, n_branches: int, k_active: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_branches, bias=False)
        self.k = k_active
        self.n_branches = n_branches

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(u)
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        topk_gates = F.softmax(topk_vals, dim=-1)
        gates = torch.zeros_like(logits).scatter(-1, topk_idx, topk_gates)
        return gates, topk_idx


class NeuralODELayer(nn.Module):
    """A single Neural ODE layer with Multi-Branch + MoE routing."""

    def __init__(self, cfg: ChappieConfig):
        super().__init__()
        self.cfg = cfg
        self.branches = nn.ModuleList([
            ODEBranch(cfg.d_model, cfg.dropout)
            for _ in range(cfg.n_branches)
        ])
        self.router = MoERouter(cfg.d_model, cfg.n_branches, cfg.k_active)
        self.input_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def _branch_forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.zeros_like(h)
        for b, branch in enumerate(self.branches):
            g = gates[..., b:b + 1]
            out = out + g * branch(h, u)
        return out

    def forward(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        gates, _ = self.router(u)
        u_proj = self.input_proj(u)

        dt = self.cfg.ode_dt
        for _ in range(self.cfg.ode_steps):
            k1 = self._branch_forward(h, u_proj, gates)
            k2 = self._branch_forward(h + 0.5 * dt * k1, u_proj, gates)
            k3 = self._branch_forward(h + 0.5 * dt * k2, u_proj, gates)
            k4 = self._branch_forward(h + dt * k3, u_proj, gates)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return h


# ============================================================
# Layer 2: Diffusion Head and GAN Head
# ============================================================

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionHead(nn.Module):
    def __init__(self, cfg: ChappieConfig):
        super().__init__()
        self.cfg = cfg
        self.time_embed = SinusoidalPositionEmbedding(cfg.diffusion_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.diffusion_dim, cfg.diffusion_dim),
            nn.GELU(),
            nn.Linear(cfg.diffusion_dim, cfg.diffusion_dim),
            nn.GELU(),
            nn.Linear(cfg.diffusion_dim, cfg.d_model),
        )
        betas = torch.linspace(1e-4, 0.02, cfg.diffusion_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        ac = self.alphas_cumprod[t].view(-1, 1, 1)
        return ac.sqrt() * x0 + (1 - ac).sqrt() * noise

    def forward(
        self,
        h_ode: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x0 is None or t is None:
            raise ValueError("DiffusionHead requires x0 and t during training")
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        t_emb = self.time_embed(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, x_t.size(1), -1)
        inp = torch.cat([x_t, t_emb], dim=-1)
        pred = self.denoiser(inp)
        loss = F.mse_loss(pred, noise)
        return {"loss": loss, "pred_noise": pred}


class GANHead(nn.Module):
    def __init__(self, cfg: ChappieConfig):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.gan_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.gan_hidden, cfg.gan_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.gan_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        return self.discriminator(pooled).squeeze(-1)


# ============================================================
# Layer 3: Analyzer
# ============================================================

class Analyzer(nn.Module):
    def __init__(self, cfg: ChappieConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.analyzer_state_dim, cfg.analyzer_d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.analyzer_d_model,
            nhead=cfg.analyzer_heads,
            dim_feedforward=4 * cfg.analyzer_d_model,
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.analyzer_layers
        )
        self.output_proj = nn.Linear(cfg.analyzer_d_model, cfg.analyzer_control_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(state)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return torch.sigmoid(self.output_proj(x))


# ============================================================
# Layer 4: Cost Engine
# ============================================================

class CostEngine(nn.Module):
    def __init__(self, cfg: ChappieConfig):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(cfg.cost_input_dim, cfg.cost_hidden),
            nn.GELU(),
            nn.Linear(cfg.cost_hidden, cfg.cost_hidden),
            nn.GELU(),
        )
        self.head_latency = nn.Linear(cfg.cost_hidden, 1)
        self.head_memory = nn.Linear(cfg.cost_hidden, 1)
        self.head_energy = nn.Linear(cfg.cost_hidden, 1)
        self.head_efficiency = nn.Linear(cfg.cost_hidden, 1)

    def forward(self, arch_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.trunk(arch_features)
        return {
            "latency": self.head_latency(x).squeeze(-1),
            "memory": self.head_memory(x).squeeze(-1),
            "energy": self.head_energy(x).squeeze(-1),
            "efficiency": self.head_efficiency(x).squeeze(-1),
        }


# ============================================================
# Layer 5: Recursive Self-Study (RSS)
# ============================================================

class SSMBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.A_log = nn.Parameter(torch.zeros(d_model))
        self.D = nn.Parameter(torch.ones(d_model))
        self.x_proj = nn.Linear(d_model, d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            u = self.x_proj(x[:, t])
            dt = F.softplus(self.dt_proj(x[:, t]))
            h = h + dt * (A * h + u)
            outputs.append(h + self.D * x[:, t])
        y = torch.stack(outputs, dim=1)
        return self.norm(y)


class RSS(nn.Module):
    def __init__(self, cfg: ChappieConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.analyzer_state_dim + cfg.analyzer_control_dim
        self.input_proj = nn.Linear(input_dim, cfg.rss_d_model)
        self.ssm_layers = nn.ModuleList([
            SSMBlock(cfg.rss_d_model) for _ in range(cfg.rss_layers)
        ])
        self.head_correctness = nn.Linear(cfg.rss_d_model, 1)
        self.head_confidence = nn.Linear(cfg.rss_d_model, 1)
        self.head_correction = nn.Linear(cfg.rss_d_model, cfg.analyzer_control_dim)
        self.head_reasoning = nn.Linear(cfg.rss_d_model, 64)

    def forward(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        control_seq = control.unsqueeze(1).expand(-1, state.size(1), -1)
        x = torch.cat([state, control_seq], dim=-1)
        x = self.input_proj(x)
        for layer in self.ssm_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return {
            "correctness": torch.sigmoid(self.head_correctness(x)).squeeze(-1),
            "confidence": torch.sigmoid(self.head_confidence(x)).squeeze(-1),
            "correction": self.head_correction(x),
            "reasoning": self.head_reasoning(x),
        }


# ============================================================
# Main Model: ChappieModel
# ============================================================

class ChappieModel(nn.Module):
    def __init__(self, cfg: ChappieConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.ode_layers = nn.ModuleList([
            NeuralODELayer(cfg) for _ in range(cfg.n_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)

        self.diffusion_head = DiffusionHead(cfg)
        self.gan_head = GANHead(cfg)
        self.analyzer = Analyzer(cfg)
        self.cost_engine = CostEngine(cfg)
        self.rss = RSS(cfg)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(input_ids) + self.pos_embed(pos)

        for layer in self.ode_layers:
            h = h + layer(h, h)
        h = self.final_norm(h)

        out: Dict[str, torch.Tensor] = {"hidden": h}

        if labels is not None:
            logits = self.lm_head(h)
            out["logits"] = logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            out["lm_loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if labels is not None and self.training:
            t = torch.randint(
                0, self.cfg.diffusion_timesteps, (B,), device=device
            )
            x0 = self.token_embed(labels).detach()
            diff_out = self.diffusion_head(h, x0=x0, t=t)
            out["diffusion_loss"] = diff_out["loss"]

        if self.training:
            out["gan_score"] = self.gan_head(h)

        if training_state is not None:
            control = self.analyzer(training_state)
            out["control"] = control
            rss_out = self.rss(training_state, control)
            out["rss"] = rss_out

        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            T = input_ids.size(1)
            if T >= self.cfg.max_seq_len:
                input_ids = input_ids[:, -self.cfg.max_seq_len:]
                T = input_ids.size(1)
            out = self.forward(input_ids)
            logits = self.lm_head(out["hidden"])[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ============================================================
# PARALLELIZATION LOGIC (Inside the same file!)
# ============================================================

def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool = False,
) -> None:
    """Apply Tensor Parallelism to the Chappie model."""
    # Embedding: shard on vocab dimension
    parallelize_module(
        model.token_embed,
        tp_mesh,
        ColwiseParallel(output_layouts=Replicate()),
    )
    parallelize_module(
        model.pos_embed,
        tp_mesh,
        ColwiseParallel(output_layouts=Replicate()),
    )

    # Neural ODE layers: shard input_proj and branches
    for layer in model.ode_layers:
        parallelize_module(
            layer.input_proj,
            tp_mesh,
            ColwiseParallel(output_layouts=Shard(1)),
        )
        for branch in layer.branches:
            parallelize_module(
                branch.net[0],
                tp_mesh,
                ColwiseParallel(output_layouts=Shard(1)),
            )
            parallelize_module(
                branch.net[3],
                tp_mesh,
                RowwiseParallel(input_layouts=Shard(1), output_layouts=Shard(2)),
            )

    # LM head: rowwise parallel
    parallelize_module(
        model.lm_head,
        tp_mesh,
        ColwiseParallel(output_layouts=Replicate()),
    )

    # Analyzer: full TransformerEncoder is parallelizable via plan
    for block in model.analyzer.transformer.layers:
        parallelize_module(
            block,
            tp_mesh,
            {
                "self_attn": PrepareModuleInput(
                    input_layouts=(Shard(1), Shard(1), Shard(1)),
                    desired_input_layouts=(Replicate(), Replicate(), Replicate()),
                ),
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.out_proj": RowwiseParallel(),
                "linear1": ColwiseParallel(),
                "linear2": RowwiseParallel(),
            },
        )


def apply_ac(model: nn.Module, ac_mode: str = "full") -> None:
    """Apply Activation Checkpointing to ODE layers."""
    if ac_mode == "none":
        return
    for layer in model.ode_layers:
        if ac_mode == "full":
            layer = ptd_checkpoint_wrapper(layer)
        # For selective, we'd wrap individual branches — simplified here


def apply_compile(model: nn.Module) -> None:
    """Apply torch.compile to the ODE layers."""
    for layer in model.ode_layers:
        layer.compile()


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
) -> None:
    """Apply FSDP2 (fully_shard) to the Chappie model."""
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )

    # Shard each ODE layer
    for layer in model.ode_layers:
        fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)

    # Shard controller networks
    fully_shard(model.analyzer, mesh=dp_mesh, mp_policy=mp_policy)
    fully_shard(model.rss, mesh=dp_mesh, mp_policy=mp_policy)
    fully_shard(model.cost_engine, mesh=dp_mesh, mp_policy=mp_policy)

    # Shard the root model
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)


def parallelize_chappie(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> nn.Module:
    """
    Main parallelization entry point for Chappie-aleph-v1.
    Mirrors the TorchTitan parallelize_llama interface.

    Order matters: TP -> AC -> compile -> FSDP
    """
    cfg: ChappieConfig = model.cfg

    # 1. Tensor Parallelism
    if parallel_dims.tp_enabled:
        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
        )

    # 2. Activation Checkpointing
    if cfg.enable_ac:
        apply_ac(model, cfg.ac_mode)

    # 3. torch.compile
    if cfg.enable_compile and job_config.training.compile:
        apply_compile(model)

    # 4. FSDP / HSDP
    if parallel_dims.dp_shard_enabled:
        dp_mesh_dim_names = ("dp_shard",) if not parallel_dims.dp_replicate_enabled else ("dp_replicate", "dp_shard")
        apply_fsdp(model, world_mesh[dp_mesh_dim_names])

    return model


# ============================================================
# Registration for TorchTitan
# ============================================================

def register_chappie_spec() -> None:
    """Register Chappie model spec with TorchTitan."""
    from torchtitan.protocols.model_spec import ModelSpec, register_model_spec

    model_spec = ModelSpec(
        name="chappie",
        model_cls=ChappieModel,
        config_cls=ChappieConfig,
        parallelize_fn=parallelize_chappie,
    )
    register_model_spec(model_spec)