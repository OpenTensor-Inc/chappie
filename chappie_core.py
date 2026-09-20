"""
Chappie Core v2
===============
A0: Dense MLP baseline
A1: Selective SSM
A2: Selective SSM + Sparse MoE
A3: Chappie Core-0 = Selective SSM + Sparse MoE + Adaptive Depth + Memory

Design goal:
    maximize capability per ACTIVE computation, not merely parameter count.

All four models share:
    - token embedding
    - causal next-token objective
    - RMSNorm
    - gated MLP experts
    - identical output interface

A3 additionally exposes:
    - sequential recurrent memory
    - input-dependent compute depth
    - sparse expert routing
    - compute/load-balance regularizers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CoreConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    d_ff: int = 2048
    n_experts: int = 8
    top_k: int = 2
    memory_dim: int = 256
    max_seq_len: int = 4096
    max_depth: int = 4
    dropout: float = 0.0
    ssm_state_size: int = 16
    tie_embeddings: bool = True

    # Auxiliary objective weights.
    compute_loss_weight: float = 0.01
    balance_loss_weight: float = 0.01
    memory_loss_weight: float = 0.001

    # Runtime diagnostics.
    return_router_logits: bool = False


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return self.weight * x


class GatedMLP(nn.Module):
    """SwiGLU-style feed-forward block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class CausalEmbedding(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position = nn.Embedding(cfg.max_seq_len, cfg.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, t = input_ids.shape
        if t > self.position.num_embeddings:
            raise ValueError(
                f"Sequence length {t} exceeds max_seq_len "
                f"{self.position.num_embeddings}"
            )
        pos = torch.arange(t, device=input_ids.device)
        return self.token(input_ids) + self.position(pos)[None, :, :]


# ---------------------------------------------------------------------------
# A0: Dense MLP baseline
# ---------------------------------------------------------------------------

class A0Block(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.ffn = GatedMLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


# ---------------------------------------------------------------------------
# A1: Selective SSM
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    Minimal selective state-space layer.

    The recurrence is causal:
        h_t = exp(dt_t A) h_{t-1} + dt_t B_t u_t
        y_t = C_t h_t + D u_t

    The implementation intentionally uses a Python time loop for Core-0
    correctness and transparent ablations. A fused scan can replace this
    kernel later without changing the model interface.
    """

    def __init__(self, d_model: int, state_size: int):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.b_proj = nn.Linear(d_model, state_size, bias=False)
        self.c_proj = nn.Linear(d_model, state_size, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Stable negative continuous-time spectrum.
        self.a_log = nn.Parameter(torch.log(
            torch.arange(1, state_size + 1, dtype=torch.float32)
        ))
        self.d = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        uv = self.in_proj(x)
        u, gate = uv.chunk(2, dim=-1)
        u = F.silu(u)
        gate = torch.sigmoid(gate)

        dt = F.softplus(self.dt_proj(u)) + 1e-4
        b = self.b_proj(u)
        c = self.c_proj(u)

        # One state vector per feature group.
        # Project scalar feature dynamics through a learned state basis.
        a = -torch.exp(self.a_log).to(x.dtype)
        state = torch.zeros(
            bsz, dim, self.state_size,
            device=x.device, dtype=x.dtype
        )

        ys = []
        a = a.view(1, 1, self.state_size)

        for t in range(seq_len):
            dt_t = dt[:, t].unsqueeze(-1)
            u_t = u[:, t].unsqueeze(-1)
            b_t = b[:, t].unsqueeze(1)
            c_t = c[:, t].unsqueeze(1)

            decay = torch.exp(dt_t * a)
            state = decay * state + dt_t * b_t * u_t
            y_t = (state * c_t).sum(dim=-1) + self.d * u[:, t]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        return self.out_proj(y * gate)


class A1Block(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.ssm = SelectiveSSM(cfg.d_model, cfg.ssm_state_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# ---------------------------------------------------------------------------
# Sparse MoE
# ---------------------------------------------------------------------------

class SparseMoE(nn.Module):
    """
    Top-k sparse MoE.

    Only selected experts are evaluated for each token. The implementation
    groups tokens by expert to avoid evaluating every expert on every token.
    """

    def __init__(self, cfg: CoreConfig):
        super().__init__()
        if cfg.top_k > cfg.n_experts:
            raise ValueError("top_k cannot exceed n_experts")

        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([
            GatedMLP(cfg.d_model, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_experts)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, x.size(-1))

        logits = self.router(flat)
        top_values, top_indices = torch.topk(
            logits, self.top_k, dim=-1
        )
        gates = F.softmax(top_values, dim=-1)

        output = torch.zeros_like(flat)

        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(top_indices == expert_id)
            if token_idx.numel() == 0:
                continue

            expert_input = flat[token_idx]
            expert_output = expert(expert_input)
            expert_gate = gates[token_idx, slot_idx].unsqueeze(-1)
            output.index_add_(
                0, token_idx, expert_output * expert_gate
            )

        # Switch-style load balancing proxy.
        probs = F.softmax(logits, dim=-1)
        importance = probs.mean(dim=0)
        load = torch.bincount(
            top_indices.reshape(-1),
            minlength=self.n_experts
        ).to(probs.dtype)
        load = load / max(1, top_indices.numel())
        balance_loss = self.n_experts * torch.sum(importance * load)

        return output.reshape(original_shape), balance_loss, logits


class A2Block(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.ssm_norm = RMSNorm(cfg.d_model)
        self.ssm = SelectiveSSM(cfg.d_model, cfg.ssm_state_size)
        self.moe_norm = RMSNorm(cfg.d_model)
        self.moe = SparseMoE(cfg)

    def forward(self, x: torch.Tensor):
        x = x + self.ssm(self.ssm_norm(x))
        moe_out, balance_loss, router_logits = self.moe(
            self.moe_norm(x)
        )
        x = x + moe_out
        return x, balance_loss, router_logits


# ---------------------------------------------------------------------------
# Recurrent long-term memory
# ---------------------------------------------------------------------------

class RecurrentMemory(nn.Module):
    """
    Causal learned memory:
        m_t = (1-w_t)m_{t-1} + w_t M(h_t)

    The memory is updated sequentially, so information written at token t
    is available to token t+1 and later tokens.
    """

    def __init__(self, d_model: int, memory_dim: int):
        super().__init__()
        self.read = nn.Linear(memory_dim, d_model, bias=False)
        self.write = nn.Linear(d_model + d_model, memory_dim, bias=False)
        self.gate = nn.Linear(d_model + d_model, 1)
        self.candidate = nn.Linear(d_model + d_model, memory_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, d_model = x.shape

        if memory is None:
            memory = torch.zeros(
                bsz, self.read.in_features,
                device=x.device, dtype=x.dtype
            )

        outputs = []
        writes = []

        for t in range(seq_len):
            read = self.read(memory)
            h = x[:, t] + read

            joined = torch.cat([x[:, t], read], dim=-1)
            write_gate = torch.sigmoid(self.gate(joined))
            candidate = torch.tanh(self.candidate(joined))

            memory = (
                (1.0 - write_gate) * memory
                + write_gate * candidate
            )

            outputs.append(h)
            writes.append(write_gate.squeeze(-1))

        return (
            torch.stack(outputs, dim=1),
            memory,
            torch.stack(writes, dim=1),
        )


# ---------------------------------------------------------------------------
# Adaptive computation
# ---------------------------------------------------------------------------

class AdaptiveController(nn.Module):
    def __init__(self, d_model: int, max_depth: int):
        super().__init__()
        self.max_depth = max_depth
        self.depth_head = nn.Linear(d_model, max_depth)
        self.halt_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # One decision per token.
        depth_logits = self.depth_head(x)
        halt = torch.sigmoid(self.halt_head(x)).squeeze(-1)

        # At least one computation step.
        depth = torch.argmax(depth_logits, dim=-1) + 1
        return depth, halt


class AdaptiveDynamics(nn.Module):
    """
    Repeated residual dynamics whose number of active steps is selected
    independently for each token.

    Forward routing is hard. A soft compute proxy is returned for the
    auxiliary compute objective.
    """

    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.max_depth = cfg.max_depth
        self.controller = AdaptiveController(
            cfg.d_model, cfg.max_depth
        )
        self.steps = nn.ModuleList([
            nn.Sequential(
                RMSNorm(cfg.d_model),
                GatedMLP(cfg.d_model, cfg.d_ff, cfg.dropout),
            )
            for _ in range(cfg.max_depth)
        ])

    def forward(self, x: torch.Tensor):
        depth, halt = self.controller(x)
        state = x
        active = []

        for step, block in enumerate(self.steps, start=1):
            mask = (depth >= step).unsqueeze(-1).to(x.dtype)
            candidate = state + block(state)
            state = mask * candidate + (1.0 - mask) * state
            active.append(mask.squeeze(-1))

        active_steps = torch.stack(active, dim=-1).sum(dim=-1)
        # Continuous surrogate for optimization/monitoring.
        compute_proxy = active_steps.float().mean() / self.max_depth

        return state, compute_proxy, depth, halt


# ---------------------------------------------------------------------------
# Base language-model wrapper
# ---------------------------------------------------------------------------

class BaseLM(nn.Module):
    architecture_name = "base"

    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = CausalEmbedding(cfg)
        self.layers = nn.ModuleList()
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(
            cfg.d_model, cfg.vocab_size, bias=False
        )

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embedding.token.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _loss(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if labels is None:
            return None

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

    def _output(
        self,
        hidden: torch.Tensor,
        labels: Optional[torch.Tensor],
        **extra,
    ) -> Dict[str, torch.Tensor]:
        logits = self.lm_head(hidden)
        loss = self._loss(logits, labels)

        out = {
            "logits": logits,
            "hidden": hidden,
        }
        if loss is not None:
            out["lm_loss"] = loss
        out.update(extra)
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            context = input_ids[:, -self.cfg.max_seq_len:]
            logits = self(context)["logits"][:, -1, :]

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(
                        logits, min(top_k, logits.size(-1))
                    )
                    cutoff = values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(
                        logits < cutoff, float("-inf")
                    )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        if was_training:
            self.train()

        return input_ids


# ---------------------------------------------------------------------------
# A0
# ---------------------------------------------------------------------------

class A0DenseMLP(BaseLM):
    architecture_name = "A0-DenseMLP"

    def __init__(self, cfg: CoreConfig):
        super().__init__(cfg)
        self.layers = nn.ModuleList([
            A0Block(cfg) for _ in range(cfg.n_layers)
        ])

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self._output(
            x,
            labels,
            compute_steps=torch.tensor(
                float(self.cfg.n_layers),
                device=x.device
            ),
        )


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------

class A1SelectiveSSM(BaseLM):
    architecture_name = "A1-SelectiveSSM"

    def __init__(self, cfg: CoreConfig):
        super().__init__(cfg)
        self.layers = nn.ModuleList([
            A1Block(cfg) for _ in range(cfg.n_layers)
        ])

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self._output(
            x,
            labels,
            compute_steps=torch.tensor(
                float(self.cfg.n_layers),
                device=x.device
            ),
        )


# ---------------------------------------------------------------------------
# A2
# ---------------------------------------------------------------------------

class A2SSMMoE(BaseLM):
    architecture_name = "A2-SSM-MoE"

    def __init__(self, cfg: CoreConfig):
        super().__init__(cfg)
        self.layers = nn.ModuleList([
            A2Block(cfg) for _ in range(cfg.n_layers)
        ])

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        balance_losses = []
        router_logits = []

        for layer in self.layers:
            x, balance, router = layer(x)
            balance_losses.append(balance)
            if self.cfg.return_router_logits:
                router_logits.append(router)

        x = self.final_norm(x)
        balance_loss = torch.stack(balance_losses).mean()

        return self._output(
            x,
            labels,
            balance_loss=balance_loss,
            compute_steps=torch.tensor(
                float(self.cfg.n_layers),
                device=x.device
            ),
            router_logits=(
                torch.stack(router_logits)
                if router_logits else torch.empty(0, device=x.device)
            ),
        )


# ---------------------------------------------------------------------------
# A3: Chappie Core-0
# ---------------------------------------------------------------------------

class A3ChappieCore0(BaseLM):
    architecture_name = "A3-Chappie-Core-0"

    def __init__(self, cfg: CoreConfig):
        super().__init__(cfg)

        # Sequence model.
        self.ssm_layers = nn.ModuleList([
            A1Block(cfg) for _ in range(cfg.n_layers)
        ])

        # Shared sparse expert transformation.
        self.moe_layers = nn.ModuleList([
            SparseMoE(cfg) for _ in range(cfg.n_layers)
        ])

        # Adaptive repeated dynamics.
        self.dynamics = nn.ModuleList([
            AdaptiveDynamics(cfg) for _ in range(cfg.n_layers)
        ])

        # Long-term recurrent memory is intentionally a single global stream
        # at Core-0 so its effect can be isolated cleanly.
        self.memory = RecurrentMemory(
            cfg.d_model, cfg.memory_dim
        )
        self.memory_norm = RMSNorm(cfg.d_model)

        self.memory_readout = nn.Linear(
            cfg.d_model, cfg.d_model, bias=False
        )

    def forward(
        self,
        input_ids,
        labels=None,
        memory_state=None,
        **kwargs,
    ):
        x = self.embedding(input_ids)

        x, new_memory, write_rates = self.memory(
            x, memory_state
        )
        x = self.memory_norm(x)

        balance_losses = []
        compute_proxies = []
        depths = []
        halts = []
        router_logits = []

        for ssm, moe, dynamics in zip(
            self.ssm_layers,
            self.moe_layers,
            self.dynamics,
        ):
            x = ssm(x)

            moe_out, balance_loss, router = moe(x)
            x = x + moe_out
            balance_losses.append(balance_loss)

            x, compute_proxy, depth, halt = dynamics(x)
            compute_proxies.append(compute_proxy)
            depths.append(depth)
            halts.append(halt)

            if self.cfg.return_router_logits:
                router_logits.append(router)

        x = self.final_norm(x)

        balance_loss = torch.stack(balance_losses).mean()
        compute_loss = torch.stack(compute_proxies).mean()

        # Penalize excessive writes only mildly; memory must remain usable,
        # but unconditional writing should not become the default solution.
        memory_write_loss = write_rates.mean()

        return self._output(
            x,
            labels,
            balance_loss=balance_loss,
            compute_loss=compute_loss,
            memory_write_loss=memory_write_loss,
            memory_state=new_memory,
            memory_write_rates=write_rates,
            compute_depth=torch.stack(depths, dim=-1),
            halt_probabilities=torch.stack(halts, dim=-1),
            router_logits=(
                torch.stack(router_logits)
                if router_logits else torch.empty(0, device=x.device)
            ),
        )


# ---------------------------------------------------------------------------
# Unified builder and training loss
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "a0": A0DenseMLP,
    "a1": A1SelectiveSSM,
    "a2": A2SSMMoE,
    "a3": A3ChappieCore0,
}


def build_model(name: str, cfg: Optional[CoreConfig] = None) -> BaseLM:
    key = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "a0": "a0",
        "a0densemlp": "a0",
        "a1": "a1",
        "a1selectivessm": "a1",
        "a2": "a2",
        "a2ssmoe": "a2",
        "a3": "a3",
        "a3chappiecore0": "a3",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown architecture {name!r}. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    cfg = cfg or CoreConfig()
    return MODEL_REGISTRY[aliases[key]](cfg)


def training_loss(
    outputs: Dict[str, torch.Tensor],
    cfg: CoreConfig,
) -> torch.Tensor:
    if "lm_loss" not in outputs:
        raise ValueError("training_loss requires labels/lm_loss")

    loss = outputs["lm_loss"]

    if "balance_loss" in outputs:
        loss = loss + cfg.balance_loss_weight * outputs["balance_loss"]

    if "compute_loss" in outputs:
        loss = loss + cfg.compute_loss_weight * outputs["compute_loss"]

    if "memory_write_loss" in outputs:
        loss = loss + cfg.memory_loss_weight * outputs["memory_write_loss"]

    return loss


def model_statistics(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
    }


__all__ = [
    "CoreConfig",
    "BaseLM",
    "A0DenseMLP",
    "A1SelectiveSSM",
    "A2SSMMoE",
    "A3ChappieCore0",
    "build_model",
    "training_loss",
    "model_statistics",
]
