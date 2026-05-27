# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange


def voxelize_pc(pc: torch.Tensor, resolution: int):
    """
    Voxelize the point cloud in [-1, 1]^3 with fixed resolution.
    Each point is replaced by the center of the voxel that contains it.

    Args:
        pc: (N, 3) or (bs, N, 3) point cloud in range [-1, 1].
        resolution: number of voxels per axis (e.g. 32 -> 32^3 grid).

    Returns:
        centers: same shape as pc, voxel center coordinates in [-1, 1], float32.
        vox_idx: same shape as pc, integer voxel indices in [0, resolution - 1].
    """
    pc = pc.to(dtype=torch.float32)
    t = (pc + 1.0) * 0.5 * resolution
    vox_idx = t.long().clamp(0, resolution - 1)
    centers = -1.0 + (vox_idx + 0.5) * 2.0 / resolution
    return centers.to(dtype=torch.float32), vox_idx


@dataclass
class Transformer1DModelOutput:
    """Output of the 1D transformer; single field `sample`."""

    sample: torch.FloatTensor


def modulate(x, shift, scale):
    """Apply affine modulation: x * (1 + scale) + shift (broadcast over sequence)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Build 1D sin/cos position embeddings.

    Args:
        embed_dim: Output dimension per position; must be even.
        pos: Position indices, shape (M,) or flattened to (M,).

    Returns:
        Position embeddings of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class DropPath(nn.Module):
    """Stochastic depth per sample (drop entire residual branch)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            mask.div_(keep_prob)
        return x * mask


class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) using a "Divide and Conquer" strategy.
    
    This implementation splits the head dimension equally among the input dimensions (ndim).
    Each axis (1D, 2D, or 3D) is assigned an independent chunk of the head_dim, 
    ensuring that rotational signals are applied consistently across all spatial axes.
    """
    
    def __init__(self, dim: int, base: float = 10_000.0, ndim: int = 1) -> None:
        """
        Args:
            dim: Total dimension of the attention head (head_dim).
            base: Base for the geometric progression of rotation frequencies.
            ndim: Number of spatial dimensions (e.g., 1 for sequence, 2 for 2D grid).
        """
        super().__init__()
        assert dim % ndim == 0, f"head_dim ({dim}) must be divisible by ndim ({ndim})"
        
        self.dim = dim
        self.base = base
        self.ndim = ndim
        self.sub_dim = dim // ndim  # Dimension assigned to each axis
        assert (
            self.sub_dim % 2 == 0
        ), f"sub_dim ({self.sub_dim}) must be even to form rotation pairs"
        
        # Precompute the inverse frequency for each axis chunk.
        # Each sub_dim chunk will be treated as sub_dim // 2 rotation pairs.
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.sub_dim, 2).float() / self.sub_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_cos_sin(self, pos_axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin for a specific axis based on its position coordinates.
        
        Args:
            pos_axis: Position indices for one axis, shape [..., 1].
        Returns:
            cos, sin: Both shape [..., sub_dim // 2].
        """
        # Outer product: [..., 1] * [sub_dim // 2] -> [..., sub_dim // 2]
        inv_freq = self.inv_freq.to(device=pos_axis.device)
        args = pos_axis.float() * inv_freq.view(1, -1)
        return torch.cos(args), torch.sin(args)

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply the rotation matrix to a chunk of the features.
        
        Args:
            x: Input feature chunk, shape [..., sub_dim].
            cos: Cosine values, shape [..., sub_dim // 2].
            sin: Sine values, shape [..., sub_dim // 2].
        Returns:
            Rotated features, shape [..., sub_dim].
        """
        
        # Reshape to [..., sub_dim // 2, 2] to apply 2D rotation for each pair
        x_float = x.float().reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_float[..., 0], x_float[..., 1]
        
        # Apply the rotation:
        # [x1_new] = [cos, -sin] * [x1]
        # [x2_new]   [sin,  cos]   [x2]
        x_out = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)
        
        return x_out.type_as(x)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to the input tensor.
        
        Args:
            x: Input tensor [batch, seq_len, num_heads, head_dim].
            pos: Spatial coordinates [batch, seq_len, ndim].
            
        Returns:
            Rotated tensor of the same shape as x.
        """
        if pos.shape[-1] != self.ndim:
            raise ValueError(f"pos last dim must be {self.ndim}, got {pos.shape[-1]}")
        # 1. Split x into ndim chunks along the head_dim
        # Each chunk has shape [..., sub_dim]
        B, N, H, D = x.shape
        x_split = x.chunk(self.ndim, dim=-1)

        out_list = []
        for i in range(self.ndim):
            # 2. Extract coordinates for the current axis: [..., 1]
            pos_axis = pos[..., i:i+1]
            
            # 3. Compute rotation frequencies for this specific axis
            cos, sin = self._compute_cos_sin(pos_axis)
            cos = cos.unsqueeze(2).expand(-1, -1, H, -1)
            sin = sin.unsqueeze(2).expand(-1, -1, H, -1)

            # 4. Apply rotation to the corresponding chunk
            out_list.append(self._apply_rotary(x_split[i], cos, sin))
            
        # 5. Concatenate all rotated chunks back to original head_dim
        return torch.cat(out_list, dim=-1)


class Timesteps(nn.Module):
    """Sinusoidal timestep embedding (sin/cos over frequency bands)."""

    def __init__(
        self,
        num_channels: int,
        downscale_freq_shift: float = 0.0,
        scale: int = 1000,
        max_period: int = 10000,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

        half_dim = num_channels // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32
        )
        exponent = exponent / (half_dim - downscale_freq_shift)
        self.register_buffer("freq_embedding", torch.exp(exponent), persistent=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed 1D timestep tensor (shape [B]) to [B, num_channels]."""
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        if self.scale == 1000:
            assert timesteps.max() <= 1.0, "Timesteps should be in the range [0, 1]"
        emb = timesteps[:, None].float() * self.freq_embedding[None, :].to(
            timesteps.device
        )
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.num_channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps to a vector [B, out_size] for AdaLN (no extra sequence token)."""

    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        cond_proj_dim=None,
        out_size=None,
    ):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, hidden_size, bias=False)

        self.time_embed = Timesteps(hidden_size)

    def forward(self, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return timestep embedding [B, out_size]."""
        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)

        if condition is not None:
            t_freq = t_freq + self.cond_proj(condition)

        return self.mlp(t_freq)


class MLP(nn.Module):
    """Two-layer FFN: Linear -> GELU -> Linear with 4x expansion."""

    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    """Multi-head cross-attention (Q from x, K/V from y) with optional QK norm and RoPE on Q."""

    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        use_rope=False,
        rope_input_ndim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary_emb_q = RotaryPositionalEmbeddings(self.head_dim, ndim=rope_input_ndim)
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention: queries from x, keys/values from y.

        Args:
            x: Query sequence [batch, seq_len_q, qdim].
            y: Key/value sequence [batch, seq_len_k, kdim].

        Returns:
            Output [batch, seq_len_q, qdim].
        """
        b, s1, c = x.shape
        _, s2, c = y.shape

        q = rearrange(self.to_q(x), "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(self.to_k(y), "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(self.to_v(y), "b n (h d) -> b n h d", h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("rope_input") is not None:
                rope_in = joint_attention_kwargs["rope_input"]
                q = self.rotary_emb_q(q, rope_in)
            else:
                pos_1d = torch.arange(s1, dtype=torch.long, device=q.device).view(1, -1, 1).expand(b, -1, -1)
                q = self.rotary_emb_q(q, pos_1d)

        with torch.nn.attention.sdpa_kernel(
            backends=[
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            q, k, v = map(
                lambda t: rearrange(t, "b n h d -> b h n d", h=self.num_heads),
                (q, k, v),
            )
            context = (
                F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
                .transpose(1, 2)
                .reshape(b, s1, -1)
            )

        return self.out_proj(context)


class Attention(nn.Module):
    """Multi-head self-attention with optional QK norm and RoPE."""

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        use_rope=False,
        rope_input_ndim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "dim should be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary_emb_q = RotaryPositionalEmbeddings(self.head_dim, ndim=rope_input_ndim)
            self.rotary_emb_k = RotaryPositionalEmbeddings(self.head_dim, ndim=rope_input_ndim)

        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, 
        x: torch.Tensor, 
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention over sequence; input/output [batch, seq_len, dim]."""
        b, s, _ = x.shape

        q = rearrange(self.to_q(x), "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(self.to_k(x), "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(self.to_v(x), "b n (h d) -> b n h d", h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("rope_input") is not None:
                rope_in = joint_attention_kwargs["rope_input"]
                q = self.rotary_emb_q(q, rope_in)
                k = self.rotary_emb_k(k, rope_in)
            else:
                pos_1d = torch.arange(s, dtype=torch.long, device=q.device).view(1, -1, 1).expand(b, -1, -1)
                q = self.rotary_emb_q(q, pos_1d)
                k = self.rotary_emb_k(k, pos_1d)

        with torch.nn.attention.sdpa_kernel(
            backends=[
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            q, k, v = map(
                lambda t: rearrange(t, "b n h d -> b h n d", h=self.num_heads),
                (q, k, v),
            )
            context = (
                F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
                .transpose(1, 2)
                .reshape(b, s, -1)
            )

        return self.out_proj(context)


def _zero_init_adaln_modulation(linear: nn.Linear) -> None:
    nn.init.constant_(linear.weight, 0)
    nn.init.constant_(linear.bias, 0)


class MeshFlowDiTBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero (7-way: self-attn + MLP + gate_cross for cross-attn),
    cross-attention, optional U-Net skip merge (learnable gate on residual path, default init 0.5),
    and DropPath on gated residuals.
    """

    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        context_dim=1024,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        qkv_bias=True,
        skip_connection=True,
        use_ele_affine: bool = True,
        use_rope: bool = False,
        rope_input_ndim: Optional[int] = None,
        use_rope_in_cross_attention: bool = False,
        skip_zero_init_gate: bool = True,
        drop_path_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_ele_affine = use_ele_affine
        self.use_rope = use_rope
        self.use_rope_in_cross_attention = use_rope_in_cross_attention

        self.norm1 = norm_layer(
            hidden_size, elementwise_affine=use_ele_affine, eps=1e-6
        )
        self.attn1 = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=qk_norm_layer,
            use_rope=use_rope,
            rope_input_ndim=rope_input_ndim,
        )
        self.norm2 = norm_layer(
            hidden_size, elementwise_affine=use_ele_affine, eps=1e-6
        )
        self.attn2 = CrossAttention(
            hidden_size,
            context_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=qk_norm_layer,
            use_rope=use_rope and use_rope_in_cross_attention,
            rope_input_ndim=rope_input_ndim,
        )
        self.norm3 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c_emb_size, 7 * hidden_size, bias=True))
        _zero_init_adaln_modulation(self.adaLN_modulation[-1])

        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
            self.skip_zero_init_gate = skip_zero_init_gate
            if skip_zero_init_gate:
                # Skip residual carries shallow 1D structure; 0.5 avoids fully suppressing it at init.
                self.skip_gate = nn.Parameter(torch.full((1,), 0.5))
            else:
                self.skip_gate = None
        else:
            self.skip_linear = None
            self.skip_norm = None
            self.skip_gate = None

        self.mlp = MLP(width=hidden_size)
        self.drop_path_attn = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.drop_path_mlp = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def _forward(
        self,
        hidden_states: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        skip_value: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.skip_linear is not None:
            assert self.skip_norm is not None
            cat = torch.cat([skip_value, hidden_states], dim=-1)
            skip_out = self.skip_linear(cat)
            if self.skip_gate is not None:
                hidden_states = self.skip_norm(skip_out + self.skip_gate * hidden_states)
            else:
                hidden_states = self.skip_norm(skip_out)

        assert temb is not None
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            gate_cross,
        ) = self.adaLN_modulation(temb).chunk(7, dim=1)

        attn_in = modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        hidden_states = hidden_states + self.drop_path_attn(
            gate_msa.unsqueeze(1)
            * self.attn1(
                attn_in,
                joint_attention_kwargs=joint_attention_kwargs,
                attention_mask=attention_mask,
            )
        )
        if conditions is not None:
            hidden_states = hidden_states + gate_cross.unsqueeze(1) * self.attn2(
                self.norm2(hidden_states),
                conditions,
                joint_attention_kwargs=joint_attention_kwargs,
                attention_mask=cross_attention_mask,
            )
        mlp_in = modulate(self.norm3(hidden_states), shift_mlp, scale_mlp)
        hidden_states = hidden_states + self.drop_path_mlp(
            gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        )

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        skip_value: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if skip_value is None and attention_mask is None and cross_attention_mask is None:
            return self._forward(
                hidden_states,
                conditions,
                temb,
                skip_value=None,
                joint_attention_kwargs=joint_attention_kwargs,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
            )

        def _run(
            _hidden_states: torch.Tensor,
            _conditions: Optional[torch.Tensor],
            _temb: Optional[torch.Tensor],
            _skip_value: Optional[torch.Tensor],
            _attention_mask: Optional[torch.Tensor],
            _cross_attention_mask: Optional[torch.Tensor],
        ) -> torch.Tensor:
            return self._forward(
                _hidden_states,
                _conditions,
                _temb,
                skip_value=_skip_value,
                joint_attention_kwargs=joint_attention_kwargs,
                attention_mask=_attention_mask,
                cross_attention_mask=_cross_attention_mask,
            )

        return torch.utils.checkpoint.checkpoint(
            _run,
            hidden_states,
            conditions,
            temb,
            skip_value,
            attention_mask,
            cross_attention_mask,
            use_reentrant=False,
        )


class FinalLayer(nn.Module):
    """Linear head; apply ``MeshFlowDiT.final_norm`` before this module."""

    def __init__(self, final_hidden_size: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(final_hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MeshFlowDiT(ModelMixin):
    """
    MeshFlow DiT: AdaLN-Zero, optional sin-cos PE, RoPE, visual cross-attention,
    and a final norm + linear head.
    """

    def __init__(
        self,
        input_size: int = 1024,
        in_channels: int = 4,
        hidden_size: int = 1024,
        context_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_type: str = "layer",
        qk_norm_type: str = "rms",
        qk_norm: bool = False,
        qkv_bias: bool = True,
        use_pos_emb: bool = False,
        use_rope: bool = False,
        rope_input_ndim: Optional[int] = None,
        use_rope_in_cross_attention: bool = False,
        use_proj_cond_on_temb: bool = False,
        proj_cond_dim: int = 1,
        visual_condition_dim: Optional[int] = None,
        skip_zero_init_gate: bool = True,
        drop_path_rate: float = 0.0,
        use_skip_connection: bool = True,
        zero_init: bool = False,
        pretrained_model_name_or_path: Optional[str] = None,
        strict_load: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm if norm_type == "layer" else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == "rms" else nn.LayerNorm
        self.context_dim = context_dim
        self.use_pos_emb = use_pos_emb
        self.use_rope = use_rope
        self.use_rope_in_cross_attention = use_rope_in_cross_attention
        self.use_proj_cond_on_temb = use_proj_cond_on_temb
        self.proj_cond_dim = proj_cond_dim
        self.drop_path_rate = drop_path_rate

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(
            hidden_size,
            hidden_size * 4,
            cond_proj_dim=proj_cond_dim if use_proj_cond_on_temb else None,
        )
        if self.use_pos_emb:
            self.register_buffer("pos_embed", torch.zeros(1, input_size, hidden_size))
            pos = np.arange(self.input_size, dtype=np.float32)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] if depth > 0 else []
        self.blocks = nn.ModuleList(
            [
                MeshFlowDiTBlock(
                    hidden_size=hidden_size,
                    c_emb_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    context_dim=context_dim,
                    qk_norm=qk_norm,
                    norm_layer=self.norm,
                    qk_norm_layer=self.qk_norm,
                    skip_connection=use_skip_connection and layer > depth // 2,
                    qkv_bias=qkv_bias,
                    use_rope=use_rope,
                    rope_input_ndim=rope_input_ndim,
                    use_rope_in_cross_attention=use_rope_in_cross_attention,
                    skip_zero_init_gate=skip_zero_init_gate,
                    drop_path_rate=dpr[layer] if dpr else 0.0,
                )
                for layer in range(depth)
            ]
        )

        self.final_norm = self.norm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

        visual_dim = visual_condition_dim if visual_condition_dim is not None else context_dim
        if visual_dim != context_dim:
            self.proj_visual = nn.Sequential(nn.RMSNorm(visual_dim), nn.Linear(visual_dim, context_dim))
        else:
            self.proj_visual = None

        if zero_init:
            for block in self.blocks:
                for name in ("attn1", "attn2"):
                    attn = getattr(block, name)
                    nn.init.constant_(attn.out_proj.weight, 0)
                    nn.init.constant_(attn.out_proj.bias, 0)
                nn.init.constant_(block.mlp.fc2.weight, 0)
                nn.init.constant_(block.mlp.fc2.bias, 0)

        if pretrained_model_name_or_path:
            ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu")
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            filtered = {}
            for k, v in state.items():
                if k.startswith("denoiser_model.dit_model."):
                    filtered[k[len("denoiser_model.dit_model.") :]] = v
                elif k.startswith("denoiser_model."):
                    rel = k[len("denoiser_model.") :]
                    if not rel.startswith("dit_model."):
                        filtered[rel] = v
            if filtered:
                self.load_state_dict(filtered, strict=strict_load)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self,
        model_input,
        timestep,
        visual_condition: Optional[torch.Tensor] = None,
        attention_mask=None,
        cross_attention_mask=None,
        joint_attention_kwargs=None,
        return_dict=True,
        **kwargs,
    ):
        assert timestep.max() <= 1.0
        conditions = None
        if visual_condition is not None:
            conditions = self.proj_visual(visual_condition) if self.proj_visual is not None else visual_condition

        x = model_input
        t = timestep
        if self.use_proj_cond_on_temb:
            temb = self.t_embedder(t, condition=kwargs["proj_cond_on_temb"])
        else:
            temb = self.t_embedder(t, condition=None)

        hidden_states = self.x_embedder(x)
        if self.use_pos_emb:
            pos_embed = self.pos_embed.to(hidden_states.dtype)
            hidden_states = hidden_states + pos_embed

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(1)
            else:
                attention_mask = attention_mask.bool()

        if cross_attention_mask is not None:
            if cross_attention_mask.dim() == 2:
                cross_attention_mask = cross_attention_mask.bool().unsqueeze(1).unsqueeze(1)
            else:
                cross_attention_mask = cross_attention_mask.bool()

        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            hidden_states = block(
                hidden_states,
                conditions,
                temb,
                skip_value=skip_value,
                joint_attention_kwargs=joint_attention_kwargs,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
            )
            if layer < self.depth // 2:
                skip_value_list.append(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.final_layer(hidden_states)
        if not return_dict:
            return hidden_states
        return Transformer1DModelOutput(sample=hidden_states)


def build_meshflow_dit(d: dict) -> MeshFlowDiT:
    """Map denoiser_model YAML dict to MeshFlowDiT constructor kwargs."""
    context_dim = d.get("condition_dim", 1024)
    return MeshFlowDiT(
        input_size=d["input_size"],
        in_channels=d["input_channels"],
        hidden_size=d["width"],
        context_dim=context_dim,
        depth=d["layers"],
        num_heads=d["num_heads"],
        mlp_ratio=d.get("mlp_ratio", 4.0),
        use_pos_emb=d.get("use_pos_emb", False),
        use_rope=d.get("use_rope", True),
        rope_input_ndim=d.get("rope_input_ndim", 3),
        use_rope_in_cross_attention=d.get("use_rope_in_cross_attention", False),
        norm_type=d.get("norm_type", "rms"),
        qk_norm_type=d.get("qk_norm_type", "rms"),
        qk_norm=d.get("qk_norm", True),
        qkv_bias=d.get("qkv_bias", False),
        visual_condition_dim=d.get("visual_condition_dim", context_dim),
        use_proj_cond_on_temb=d.get("use_proj_cond_on_temb", False),
        proj_cond_dim=d.get("proj_cond_dim", 1),
        skip_zero_init_gate=d.get("skip_zero_init_gate", True),
        use_skip_connection=d.get("use_skip_connection", True),
        drop_path_rate=d.get("drop_path_rate", 0.0),
        zero_init=d.get("zero_init", False),
        pretrained_model_name_or_path=d.get("pretrained_model_name_or_path"),
        strict_load=d.get("strict_load", True),
    )
