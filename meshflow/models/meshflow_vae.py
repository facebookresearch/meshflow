# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from meshflow.utils.mesh import extract_mesh_from_verts_normals_edges


class FP32LayerNorm(nn.LayerNorm):
    """LayerNorm computed in float32 for numerical stability."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class RMSNorm(nn.Module):
    """RMSNorm on the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states


class GELUProj(nn.Module):
    """Linear projection followed by GELU."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden_states))


class FeedForward(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        if activation_fn != "gelu":
            raise ValueError(f"Only activation_fn='gelu' is supported, got {activation_fn!r}")
        inner_dim = int(dim * 4) if inner_dim is None else inner_dim
        layers: List[nn.Module] = [
            GELUProj(dim, inner_dim, bias=bias),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=bias),
        ]
        if final_dropout:
            layers.append(nn.Dropout(dropout))
        self.net = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Attention(nn.Module):
    """
    Multi-head attention with optional QK normalization.
    Uses parameter names to_q, to_k, to_v, norm_q, norm_k, and to_out.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        qk_norm: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        kv_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=bias)

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm in ("layer_norm", "fp32_layer_norm"):
            norm_cls = FP32LayerNorm if qk_norm == "fp32_layer_norm" else nn.LayerNorm
            self.norm_q = norm_cls(dim_head, eps=eps)
            self.norm_k = norm_cls(dim_head, eps=eps)
        else:
            raise ValueError(f"Unsupported qk_norm: {qk_norm!r}")

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len_q, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len_q, -1)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        use_self_attention: bool = True,
        use_cross_attention: bool = False,
        self_attention_norm_type: Optional[str] = "layer_norm",
        cross_attention_dim: Optional[int] = None,
        cross_attention_norm_type: Optional[str] = None,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        head_dim = dim // num_attention_heads
        attn_kwargs = dict(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            eps=norm_eps,
        )

        def _make_norm(norm_type: Optional[str]) -> nn.Module:
            if norm_type == "fp32_layer_norm":
                return FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
            return nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        if use_self_attention:
            self.norm1 = _make_norm(self_attention_norm_type)
            self.attn1 = Attention(**attn_kwargs)
        else:
            self.norm1 = None
            self.attn1 = None

        if use_cross_attention:
            cross_norm = cross_attention_norm_type or self_attention_norm_type
            self.norm2 = _make_norm(cross_norm)
            self.attn2 = Attention(
                cross_attention_dim=cross_attention_dim, **attn_kwargs
            )
        else:
            self.norm2 = None
            self.attn2 = None

        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.attn1 is not None:
            norm_hidden_states = self.norm1(hidden_states)
            hidden_states = (
                self.attn1(norm_hidden_states, attention_mask=attention_mask)
                + hidden_states
            )
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            hidden_states = (
                self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                )
                + hidden_states
            )

        norm_hidden_states = self.norm3(hidden_states)
        return self.ff(norm_hidden_states) + hidden_states


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False, feat_dim=1):
        self.feat_dim = feat_dim
        if isinstance(parameters, list):
            self.mean, self.logvar = parameters[0], parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None, dims=(1, 2)):
        if self.deterministic:
            return torch.tensor(0.0, device=self.mean.device)
        if other is None:
            return 0.5 * torch.mean(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims
            )
        return 0.5 * torch.mean(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=dims,
        )

    def mode(self):
        return self.mean


class FourierEmbedder(nn.Module):
    def __init__(
        self,
        num_freqs: int = 12,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = False,
    ):
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32
            )
        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        temp = 1 if include_input or num_freqs == 0 else 0
        self.out_dim = input_dim * (num_freqs * 2 + temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs == 0:
            return x
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class Encoder(nn.Module):
    def __init__(
        self,
        num_enc_latents: int,
        in_channels: int,
        num_latents: int = 1024,
        latents_dim: int = 64,
        dim: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 8,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.num_enc_latents = num_enc_latents
        self.in_channels = in_channels
        self.num_latents = num_latents
        self.latents_dim = latents_dim
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.qk_norm = qk_norm

        assert num_enc_latents % num_latents == 0
        self.downsample_ratio = num_enc_latents // num_latents

        self.proj_in = nn.Linear(self.in_channels, dim, bias=True)
        self.downsample_mlp = nn.Linear(
            self.dim * self.downsample_ratio,
            self.dim,
            bias=True,
        )

        self.cross_attn_block = BasicTransformerBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            use_self_attention=False,
            use_cross_attention=True,
            cross_attention_dim=dim,
            cross_attention_norm_type="fp32_layer_norm",
            qk_norm=qk_norm,
        )

        self.blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=False,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = FP32LayerNorm(dim)

    def _get_init_query(self, hidden_states, pc):
        bs = hidden_states.shape[0]
        pc = pc.reshape(bs, -1, 3 * self.downsample_ratio)
        q = self.downsample_mlp(
            hidden_states.reshape(bs, -1, self.dim * self.downsample_ratio)
        )
        return q, pc

    def _forward(
        self,
        x: torch.FloatTensor,
        pc: torch.FloatTensor,
    ):
        _, num_tokens, _ = x.shape
        assert self.num_enc_latents == num_tokens

        hidden_states = self.proj_in(x)
        query, pc = self._get_init_query(hidden_states, pc)

        hidden_states = self.cross_attn_block(
            query,
            encoder_hidden_states=hidden_states,
        )

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=None)
        hidden_states = self.norm_out(hidden_states)

        return hidden_states, pc

    def forward(
        self,
        x: torch.FloatTensor,
        pc: torch.FloatTensor,
    ):
        if self.training:
            return checkpoint(self._forward, x, pc, use_reentrant=False)
        return self._forward(x, pc)


class Decoder(nn.Module):
    def __init__(
        self,
        num_dec_latents: int,
        num_latents: int = 1024,
        adjacency_emb_dim: int = 32,
        dim: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 16,
        qk_norm: Optional[str] = "rms_norm",
        use_verts_head: bool = True,
        use_verts_normal_head: bool = True,
        use_mask_head: bool = True,
        use_adjacency_head: bool = True,
    ):
        super().__init__()
        self.num_dec_latents = num_dec_latents
        self.num_latents = num_latents
        self.adjacency_emb_dim = adjacency_emb_dim
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.use_verts_head = use_verts_head
        self.use_verts_normal_head = use_verts_normal_head
        self.use_mask_head = use_mask_head
        self.use_adjacency_head = use_adjacency_head

        assert num_dec_latents % num_latents == 0
        self.upsample_ratio = num_dec_latents // num_latents
        self.upsample_mlp = nn.Linear(
            self.dim // self.upsample_ratio,
            self.dim,
            bias=True,
        )

        self.cross_attn_block = BasicTransformerBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            use_self_attention=False,
            use_cross_attention=True,
            cross_attention_norm_type="fp32_layer_norm",
            qk_norm=qk_norm,
        )

        self.blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=False,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = FP32LayerNorm(dim)

        if use_mask_head:
            self.mask_head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.LeakyReLU(),
                nn.Linear(dim // 2, 1),
            )
        if use_verts_head:
            self.verts_head = nn.Sequential(
                nn.Linear(dim, dim // 2), nn.LeakyReLU(), nn.Linear(dim // 2, 3)
            )
        if use_verts_normal_head:
            self.verts_normal_head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.LeakyReLU(),
                nn.Linear(dim // 2, 3),
                nn.Tanh(),
            )
        if use_adjacency_head:
            self.adjacency_head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.LeakyReLU(),
                nn.Linear(dim // 2, adjacency_emb_dim),
            )

    def _get_init_query(self, hidden_states):
        bs = hidden_states.shape[0]
        return self.upsample_mlp(
            hidden_states.reshape(bs, -1, self.dim // self.upsample_ratio)
        )

    def _forward_features(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        for block in self.blocks:
            latents = block(latents, attention_mask=None)

        query = self._get_init_query(latents)
        latents = self.cross_attn_block(query, encoder_hidden_states=latents)
        out = self.norm_out(latents)

        ret: Dict[str, torch.Tensor] = {}
        if self.use_mask_head:
            ret["mask"] = self.mask_head(out)
        if self.use_verts_head:
            ret["verts"] = self.verts_head(out)
        if self.use_verts_normal_head:
            ret["verts_normal"] = self.verts_normal_head(out)
        if self.use_adjacency_head:
            ret["adjacency_emb"] = self.adjacency_head(out)
        return ret

    def forward(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.training:
            return checkpoint(self._forward_features, latents, use_reentrant=False)
        return self._forward_features(latents)


class MeshFlowVAE(nn.Module):
    def __init__(
        self,
        *,
        n_samples: int,
        num_latents: int = 4096,
        latents_dim: int = 64,
        width_encoder: int = 1024,
        width_decoder: int = 1024,
        num_attention_heads: int = 16,
        num_layers_encoder: int = 8,
        num_layers_decoder: int = 8,
        qk_norm: Optional[str] = "rms_norm",
        adjacency_emb_dim: int = 32,
        max_degree: int = 50,
        point_feats_type: List[str] = ["verts_normal"],
        embedder_kwargs: Optional[Dict[str, Any]] = None,
        use_verts_head: bool = True,
        use_verts_normal_head: bool = True,
        use_mask_head: bool = True,
        use_adjacency_head: bool = True,
        use_learnable_tau: bool = False,
        init_tau: float = 0.6,
        adjacency_emb_rescale: float = 1.0,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        pretrained_model_name_or_path: Optional[str] = None,
        strict_load: bool = True,
    ):
        super().__init__()
        self.latents_dim = latents_dim
        self.width_encoder = width_encoder
        self.point_feats_type = point_feats_type
        self.adjacency_emb_rescale = float(adjacency_emb_rescale)
        self.register_buffer(
            "latent_mean",
            torch.tensor(0.0 if mean is None else float(mean), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            torch.tensor(1.0 if std is None else float(std), dtype=torch.float32),
            persistent=False,
        )

        if latents_dim > 0:
            self.post_kl = nn.Linear(latents_dim, width_decoder)
            self.latent_shape = (num_latents, latents_dim)
        else:
            self.latent_shape = (num_latents, width_encoder)

        self.embedder = FourierEmbedder(**(embedder_kwargs or {}))
        if latents_dim > 0:
            self.pre_kl = nn.Linear(width_encoder, latents_dim * 2)

        self.feature_dim = {
            "verts": 3,
            "verts_normal": 3,
            "neighbor_points": max_degree * 3,
            "adjacency_matrix": n_samples,
            "degree": 1,
            "verts_mask": 1,
            "adjacency_emb": adjacency_emb_dim,
        }
        in_channels = self.embedder.out_dim
        for k in self.point_feats_type:
            key = k.split("@")[-1]
            if key not in self.feature_dim:
                raise KeyError(
                    f"Unknown point feature '{key}' in point_feats_type; "
                    f"known keys: {sorted(self.feature_dim)}"
                )
            d = self.feature_dim[key]
            if k.startswith("embed@"):
                temp = 1 if self.embedder.include_input or self.embedder.num_freqs == 0 else 0
                in_channels += d * (self.embedder.num_freqs * 2 + temp)
            else:
                in_channels += d
        self.in_channels = in_channels

        self.encoder = Encoder(
            num_enc_latents=n_samples,
            in_channels=self.in_channels,
            num_latents=num_latents,
            latents_dim=latents_dim,
            dim=width_encoder,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers_encoder,
            qk_norm=qk_norm,
        )
        self.decoder = Decoder(
            num_dec_latents=n_samples,
            num_latents=num_latents,
            adjacency_emb_dim=adjacency_emb_dim,
            dim=width_decoder,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers_decoder,
            qk_norm=qk_norm,
            use_verts_head=use_verts_head,
            use_verts_normal_head=use_verts_normal_head,
            use_mask_head=use_mask_head,
            use_adjacency_head=use_adjacency_head,
        )
        self.tau = nn.Parameter(torch.ones(1) * init_tau) if use_learnable_tau else init_tau

        if pretrained_model_name_or_path:
            self.load_pretrained_model(pretrained_model_name_or_path, strict=strict_load)

    def load_pretrained_model(self, path: str, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        filtered = {}
        for k, v in state.items():
            for prefix in ("mesh_model.vae.", "mesh_model."):
                if k.startswith(prefix):
                    filtered[k[len(prefix) :]] = v
                    break
        if filtered:
            self.load_state_dict(filtered, strict=strict)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latent_mean.to(z)) / self.latent_std.to(z)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latent_std.to(z) + self.latent_mean.to(z)

    def encode(self, x, pc, sample_posterior=True):
        shape_latents, encode_pc = self.encoder(x, pc)
        assert shape_latents.shape[-1] == self.width_encoder
        if self.latents_dim > 0:
            posterior = DiagonalGaussianDistribution(self.pre_kl(shape_latents), feat_dim=-1)
            z = posterior.sample() if sample_posterior else posterior.mode()
        else:
            z, posterior = shape_latents, None
        return {
            "shape_latents": shape_latents,
            "encode_pc": encode_pc,
            "z": z,
            "posterior": posterior,
        }

    def decode(self, latents):
        if self.latents_dim > 0:
            latents = self.post_kl(latents)
        return self.decoder(latents)

    def collect_input(self, batch: dict) -> torch.Tensor:
        parts = [self.embedder(batch["padded_verts"])]
        for k in self.point_feats_type:
            raw_key = k.split("@")[-1]
            v = batch[raw_key]
            if k.startswith("embed@"):
                v = self.embedder(v)
            if raw_key == "adjacency_emb" and self.adjacency_emb_rescale != 1.0:
                v = v * self.adjacency_emb_rescale
            parts.append(v)
        return torch.cat(parts, dim=-1)

    def forward(self, batch, sample_posterior=False, **kwargs):
        pc = batch["padded_verts"]
        enc = self.encode(self.collect_input(batch), pc, sample_posterior=sample_posterior)
        return {**enc, "pc": pc, "dec": self.decode(enc["z"])}

    @staticmethod
    def compute_spacetime_distances(adjacency_embeddings: torch.Tensor, k_s: int) -> torch.Tensor:
        x_s = adjacency_embeddings[..., :k_s]
        x_t = adjacency_embeddings[..., k_s:]
        return torch.cdist(x_s, x_s, p=2) ** 2 - torch.cdist(x_t, x_t, p=2) ** 2

    def extract_mesh(
        self,
        pred: Dict[str, torch.Tensor],
        fill_holes: bool = False,
        max_cycle_size: int = 20,
        return_full: bool = False,
    ) -> List:
        """Extract one mesh per batch item. pred tensors are [B, N, ...] or [N, ...]."""
        if pred["mask"].dim() == 2:
            pred = {k: v.unsqueeze(0) for k, v in pred.items()}

        out: List = []
        for i in range(pred["mask"].shape[0]):
            pred_i = {k: v[i] for k, v in pred.items()}
            mask = pred_i["mask"].squeeze(-1) > 0.0
            verts = pred_i["verts"][mask].float()
            vnorm = pred_i["verts_normal"][mask].float().clamp(-1, 1)
            adj = pred_i["adjacency_emb"][mask].float()
            if self.adjacency_emb_rescale != 1.0:
                adj = adj / self.adjacency_emb_rescale

            k_s = adj.shape[-1] // 2
            dist = self.compute_spacetime_distances(adj, k_s)
            triu = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
            tau = self.tau if isinstance(self.tau, (int, float)) else float(self.tau.item())
            edges = torch.nonzero((torch.sigmoid(dist - tau) > 0.5) & triu, as_tuple=False)

            mesh = extract_mesh_from_verts_normals_edges(
                verts, vnorm, edges, fill_holes=fill_holes, max_cycle_size=max_cycle_size
            )
            out.append((mesh, verts, vnorm, edges) if return_full else mesh)
        return out


def build_mesh_model(cfg: dict) -> MeshFlowVAE:
    """Build MeshFlowVAE from a mesh_model config dict."""
    return MeshFlowVAE(
        n_samples=cfg["n_samples"],
        num_latents=cfg.get("num_latents", 4096),
        latents_dim=cfg.get("latents_dim", 64),
        width_encoder=cfg.get("width_encoder", 1024),
        width_decoder=cfg.get("width_decoder", 1024),
        num_attention_heads=cfg.get("num_attention_heads", 16),
        num_layers_encoder=cfg.get("num_layers_encoder", 8),
        num_layers_decoder=cfg.get("num_layers_decoder", 8),
        qk_norm=cfg.get("qk_norm", "rms_norm"),
        adjacency_emb_dim=cfg.get("adjacency_emb_dim", 32),
        max_degree=cfg.get("max_degree", 50),
        point_feats_type=cfg.get("point_feats_type", ["verts_normal"]),
        embedder_kwargs=cfg.get("embedder_kwargs"),
        use_verts_head=cfg.get("use_verts_head", True),
        use_verts_normal_head=cfg.get("use_verts_normal_head", True),
        use_mask_head=cfg.get("use_mask_head", True),
        use_adjacency_head=cfg.get("use_adjacency_head", True),
        use_learnable_tau=cfg.get("use_learnable_tau", False),
        init_tau=cfg.get("init_tau", 0.6),
        adjacency_emb_rescale=cfg.get("adjacency_emb_rescale", 1.0),
        mean=cfg.get("mean"),
        std=cfg.get("std"),
        pretrained_model_name_or_path=cfg.get("pretrained_model_name_or_path"),
        strict_load=cfg.get("strict_load", True),
    )
