from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math


def _find_multiple(a, b):
    return math.ceil(a / b) * b

""" From: https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/common.py#L7 """
def trunc_normal_init_(tensor, std, lower=-2.0, upper=2.0):
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, scale_factor):
        super().__init__()
        hidden = round(hidden_size * scale_factor * 2 / 3)
        inter = _find_multiple(hidden, 256)

        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class RoPE(nn.Module): # For 1D sequences
    def __init__(self, d, base=10000):
        super().__init__()
        self.d = d
        self.base = base
        self.cos_cache = None
        self.sin_cache = None

    def _build_cache(self, x):
        cache_is_empty = self.cos_cache is None or self.sin_cache is None
        seq_len = x.size(-2)
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d))
        start_ix = 0 if cache_is_empty else self.cos_cache.size(0) 
        seq_idx = torch.arange(start_ix, seq_len, dtype=torch.float)
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta = torch.cat([idx_theta, idx_theta], dim=-1)
        if cache_is_empty:
            self.cos_cache = torch.cos(idx_theta)
            self.sin_cache = torch.sin(idx_theta)
        else:
            self.cos_cache = torch.cat([self.cos_cache, torch.cos(idx_theta)], dim=0)
            self.sin_cache = torch.cat([self.sin_cache, torch.sin(idx_theta)], dim=0)

    def _rotate_half(self, x):
        x1 = x[..., :x.size(-1) // 2]
        x2 = x[..., x.size(-1) // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, x, cos, sin):
        return (x * cos) + (self._rotate_half(x) * sin) 

    def forward(self, x):
        self._build_cache(x)
        cos = self.cos_cache[:x.size(-2), :].to(x.device)
        sin = self.sin_cache[:x.size(-2), :].to(x.device)
        return self._apply_rotary_pos_emb(x, cos, sin)


class RoPEAttention(nn.Module):
    def __init__(self, in_dim, n_heads=8, dim_head=64):
        super().__init__()
        self.in_dim = in_dim
        self.n_heads = n_heads
        self.dim_head = dim_head
        dim_model = n_heads * dim_head
        self.norm = nn.LayerNorm(in_dim)
        self.qkv = nn.Linear(in_dim, dim_model * 3, bias=False)
        self.out = nn.Linear(dim_model, in_dim)
        self.rope = RoPE(dim_head)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, "bs l (h d) -> bs h l d", h=self.n_heads), qkv)
        q, k = self.rope(q), self.rope(k)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = einops.rearrange(attn, "bs h l d -> bs l (h d)")
        return self.out(attn)


class TRMBlock(nn.Module):
    def __init__(self, hidden_size=512, n_heads=8, dim_head=64):
        super().__init__()
        self.attn = RoPEAttention(in_dim=hidden_size, n_heads=n_heads, dim_head=dim_head)
        self.mlp = SwiGLU(hidden_size=hidden_size, scale_factor=4)

    def forward(self, x):
        x = F.rms_norm(x + self.attn(x), (x.size(-1),))
        x = F.rms_norm(x + self.mlp(x), (x.size(-1),))
        return x


class TRMNet(nn.Module):
    def __init__(self, num_blocks, hidden_size=512, n_heads=8, dim_head=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            TRMBlock(hidden_size=hidden_size, n_heads=n_heads, dim_head=dim_head)
            for _ in range(num_blocks)
        ])

    def forward(self, z, input_embed):
        z = z + input_embed
        for block in self.blocks:
            z = block(z)
        return z

# # TODO: Check if this is even neccessery. If not just use pytorch nn.init.trunc_normal_
# class TRMEmbedding(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, init_std):
#         super().__init__()
#         self.embedding = nn.Embedding(num_embeddings, embedding_dim)
#         trunc_normal_init_(self.embedding.weight, std=init_std)
        
#     def forward(self, input):
#         return self.embedding(input)


class TRMInner(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_trm_blocks=4, num_heads=8,
                 L_cycles=6, H_cycles=3):
        super().__init__()

        self.hidden_size = hidden_size
        self.embed_scale = math.sqrt(hidden_size)
        self.embed_tokens = nn.Linear(vocab_size, hidden_size, bias=False)
        # trunc_normal_init_(self.embed_tokens.weight, std=1.0/self.embed_scale)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.q_head = nn.Linear(hidden_size, 2, bias=True)
        self.t_embed = MLP(1, hidden_size, hidden_size)
        self.net = TRMNet(
            num_blocks=num_trm_blocks, hidden_size=hidden_size,
            n_heads=num_heads, dim_head=hidden_size // num_heads
        )

        self.input_reg = nn.Parameter(trunc_normal_init_(torch.empty(hidden_size), std=0))
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.hidden_size), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.hidden_size), std=1), persistent=True)
        self.L_cycles = L_cycles
        self.H_cycles = H_cycles

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def input_embed(self, x, t):
        embed = self.embed_tokens(x) * self.embed_scale + self.t_embed(t)
        embed = torch.cat([
            self.input_reg.expand(embed.size(0), 1, -1), embed
        ], dim=-2)
        return embed

    def init_carry(self, batch_size, seq_len):
        z_H = self.H_init.expand(batch_size, seq_len+1, self.hidden_size)
        z_L = self.L_init.expand(batch_size, seq_len+1, self.hidden_size)
        return TRMCarry(z_L=z_L, z_H=z_H)

    def latent_recursion(self, z_L, z_H, input_embeddings):
        for _ in range(self.L_cycles):
            z_L = self.net(z_L, z_H + input_embeddings)
        z_H = self.net(z_H, z_L)
        return z_L, z_H

    def deep_recursion(self, z_L, z_H, input_embeddings):
        with torch.no_grad():
            for _ in range(self.H_cycles):
                z_L, z_H = self.latent_recursion(z_L, z_H, input_embeddings)
        z_L, z_H = self.latent_recursion(z_L, z_H, input_embeddings)
        return z_L, z_H

    def forward(self, x, t, carry=None):
        bs, seq_len, _ = x.size()
        input_embeddings = self.input_embed(x, t)
        if carry is None:
            carry = self.init_carry(bs, seq_len)
        z_L, z_H = carry.z_L, carry.z_H
        z_L, z_H = self.deep_recursion(z_L, z_H, input_embeddings)

        new_carry = TRMCarry(z_L=z_L.detach(), z_H=z_H.detach())
        y_hat = self.lm_head(z_H[:, 1:, :])
        q_hat = self.q_head(z_H[:, 0, :])
        return y_hat, q_hat, new_carry

@dataclass
class TRMCarry:
    z_L: torch.Tensor
    z_H: torch.Tensor
