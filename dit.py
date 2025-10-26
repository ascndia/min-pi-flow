# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
#from favor import Attention

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dim_head):
        super().__init__()
        self.n_heads = n_heads
        self.n_rep = 1
        assert dim % n_heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // n_heads)
        self.head_dim = dim_head

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)



# def modulate(x, shift, scale):
#     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# class TimestepEmbedder(nn.Module):
#     def __init__(self, hidden_size, frequency_embedding_size=256):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(frequency_embedding_size, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size),
#         )
#         self.frequency_embedding_size = frequency_embedding_size

#     @staticmethod
#     def timestep_embedding(t, dim, max_period=10000):
#         half = dim // 2
#         freqs = torch.exp(
#             -math.log(max_period) * torch.arange(start=0, end=half) / half
#         ).to(t.device)
#         args = t[:, None] * freqs[None]
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         if dim % 2:
#             embedding = torch.cat(
#                 [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
#             )
#         return embedding

#     def forward(self, t):
#         t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
#             dtype=next(self.parameters()).dtype
#         )
#         t_emb = self.mlp(t_freq)
#         return t_emb


# class LabelEmbedder(nn.Module):
#     def __init__(self, num_classes, hidden_size, dropout_prob):
#         super().__init__()
#         use_cfg_embedding = int(dropout_prob > 0)
#         self.embedding_table = nn.Embedding(
#             num_classes + use_cfg_embedding, hidden_size
#         )
#         self.num_classes = num_classes
#         self.dropout_prob = dropout_prob

#     def token_drop(self, labels, force_drop_ids=None):
#         if force_drop_ids is None:
#             drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
#             drop_ids = drop_ids.cuda()
#             drop_ids = drop_ids.to(labels.device)
#         else:
#             drop_ids = force_drop_ids == 1
#         labels = torch.where(drop_ids, self.num_classes, labels)
#         return labels

#     def forward(self, labels, train, force_drop_ids=None):
#         use_dropout = self.dropout_prob > 0
#         if (train and use_dropout) or (force_drop_ids is not None):
#             labels = self.token_drop(labels, force_drop_ids)
#         embeddings = self.embedding_table(labels)
#         return embeddings

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
#         super().__init__()
#         hidden_dim = int(2 * hidden_dim / 3)
#         if ffn_dim_multiplier:
#             hidden_dim = int(ffn_dim_multiplier * hidden_dim)
#         hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

#         self.w1 = nn.Linear(dim, hidden_dim, bias=False)
#         self.w2 = nn.Linear(hidden_dim, dim, bias=False)
#         self.w3 = nn.Linear(dim, hidden_dim, bias=False)

#     def _forward_silu_gating(self, x1, x3):
#         return F.silu(x1) * x3

#     def forward(self, x):
#         return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


# class TransformerBlock(nn.Module):
#     def __init__(
#         self,
#         layer_id,
#         dim,
#         n_heads,
#         multiple_of,
#         ffn_dim_multiplier,
#         norm_eps,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.head_dim = dim // n_heads
#         self.attention = Attention(dim, n_heads, dim_head=self.head_dim)
#         self.feed_forward = FeedForward(
#             dim=dim,
#             hidden_dim=4 * dim,
#             multiple_of=multiple_of,
#             ffn_dim_multiplier=ffn_dim_multiplier,
#         )
#         self.layer_id = layer_id
#         self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
#         self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(min(dim, 1024), 6 * dim, bias=True),
#         )

#     def forward(self, x, freqs_cis, adaln_input=None):
#         if adaln_input is not None:
#             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
#                 self.adaLN_modulation(adaln_input).chunk(6, dim=1)
#             )

#             x = x + gate_msa.unsqueeze(1) * self.attention(
#                 modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
#             )
#             x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
#                 modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
#             )
#         else:
#             x = x + self.attention(self.attention_norm(x), freqs_cis)
#             x = x + self.feed_forward(self.ffn_norm(x))

#         return x


# class FinalLayer(nn.Module):
#     def __init__(self, hidden_size, patch_size, out_channels):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.linear = nn.Linear(
#             hidden_size, patch_size * patch_size * out_channels, bias=True
#         )
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
#         )
#         # init zero
#         nn.init.constant_(self.linear.weight, 0)
#         nn.init.constant_(self.linear.bias, 0)

#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
#         x = modulate(self.norm_final(x), shift, scale)
#         x = self.linear(x)
#         return x

# class DiT_Llama(nn.Module):
#     def __init__(
#         self,
#         in_channels=3,
#         input_size=32,
#         patch_size=2,
#         dim=512,
#         n_layers=5,
#         n_heads=16,
#         multiple_of=256,
#         ffn_dim_multiplier=None,
#         norm_eps=1e-5,
#         class_dropout_prob=0.1,
#         num_classes=10,
#         K=None,
#     ):
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = in_channels
#         self.input_size = input_size
#         self.patch_size = patch_size
#         self.K = K

#         self.init_conv_seq = nn.Sequential(
#             nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
#             nn.SiLU(),
#             nn.GroupNorm(32, dim // 2),
#             nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
#             nn.SiLU(),
#             nn.GroupNorm(32, dim // 2),
#         )

#         self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
#         nn.init.constant_(self.x_embedder.bias, 0)
#         self.t_embedder = TimestepEmbedder(min(dim, 1024))
#         self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

#         self.layers = nn.ModuleList(
#             [
#                 TransformerBlock(
#                     layer_id,
#                     dim,
#                     n_heads,
#                     multiple_of,
#                     ffn_dim_multiplier,
#                     norm_eps,
#                 )
#                 for layer_id in range(n_layers)
#             ]
#         )
#         if self.K is not None:
#             self.final_layer_A = FinalLayer(dim, patch_size, self.K)
#             self.final_layer_u = FinalLayer(dim, patch_size, self.K * self.out_channels)
#             self.final_layer_s = FinalLayer(dim, patch_size, 1)
#         else:
#             self.final_layer = FinalLayer(dim, patch_size, self.out_channels)
#         self.freqs_cis = DiT_Llama.precompute_freqs_cis(dim // n_heads, 4096)

#     def unpatchify(self, x):
#         c = self.out_channels
#         p = self.patch_size
#         h = w = int(x.shape[1] ** 0.5)
#         x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
#         x = torch.einsum("nhwpqc->nchpwq", x)
#         imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
#         return imgs

#     def patchify(self, x):
#         B, C, H, W = x.size()
#         x = x.view(
#             B,
#             C,
#             H // self.patch_size,
#             self.patch_size,
#             W // self.patch_size,
#             self.patch_size,
#         )
#         x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
#         return x

#     def forward(self, x, t, y):
#         self.freqs_cis = self.freqs_cis.to(x.device)
#         input_t = t.clone().detach()
#         input_x = x.clone().detach()
#         shape_x = x.shape
#         B, C, H, W = shape_x

#         x = self.init_conv_seq(x)
#         x = self.patchify(x)
#         x = self.x_embedder(x)

#         t = self.t_embedder(t)  # (N, D)
#         y = self.y_embedder(y, self.training)  # (N, D)
#         adaln_input = t.to(x.dtype) + y.to(x.dtype)

#         for layer in self.layers:
#             x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

#         if self.K is None:
#             x = self.final_layer(x, adaln_input)
#             x = self.unpatchify(x)
#             return x
        
#         else:
#             A = self.final_layer_A(x, adaln_input)
#             u = self.final_layer_u(x, adaln_input)
#             s = self.final_layer_s(x, adaln_input)
#             A = self.unpatchify(A) # (N, K, H, W)
#             u = self.unpatchify(u) # (N, C * K, H, W)
#             s = self.unpatchify(s) # (N, 1, H, W)

#             A = A.reshape(shape_x[0], self.K, 1, *shape_x[2:]).softmax(dim=1) # (N, K, 1, H, W)
#             u = rearrange(u, 'n (c k) h w -> n k c h w', k=self.K, c=C, h=H, w=W) # (N, K, C, H, W)
#             s = rearrange(s, 'n 1 h w -> n 1 1 h w').mean(dim=[-2,-1], keepdim=True) # (N, 1, 1, 1, 1)
#             s = F.softplus(s) # (N, 1, 1, 1, 1)

#             return { 
#                 'A_s': A,                                # (N, K, 1, H, W)
#                 'mu_s': u,                               # (N, K, C, H, W)
#                 'sigma_s': s,                            # (N, 1, 1, 1, 1)
#                 's': input_t[:, None, None, None, None], # (N, 1, 1, 1, 1)
#                 'x_s': input_x[:, None, :, :, :],        # (N, 1, C, H, W)
#             }

#     def forward_with_cfg(self, x, t, y, cfg_scale):
#         half = x[: len(x) // 2]
#         combined = torch.cat([half, half], dim=0)
#         model_out = self.forward(combined, t, y)
#         eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
#         cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
#         half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
#         eps = torch.cat([half_eps, half_eps], dim=0)
#         return torch.cat([eps, rest], dim=1)

#     @staticmethod
#     def precompute_freqs_cis(dim, end, theta=10000.0):
#         freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#         t = torch.arange(end)
#         freqs = torch.outer(t, freqs).float()
#         freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
#         return freqs_cis

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import Transformer Engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe # For defining FP8 recipes later

# --- Helper Functions (Remain Unchanged) ---
def modulate(x, shift, scale):
    # Ensure scale and shift are broadcastable
    # Assume x: [B, S, D], shift/scale: [B, D]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# --- TimestepEmbedder ---
# Replace nn.Linear with te.Linear
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        # Use te.Linear for potential acceleration
        self.mlp = nn.Sequential(
            te.Linear(frequency_embedding_size, hidden_size), # TE Linear
            nn.SiLU(),
            te.Linear(hidden_size, hidden_size),             # TE Linear
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # Math functions are not accelerated by TE, run in baseline precision
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        # timestep_embedding runs in baseline precision
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype # Match baseline dtype (e.g., BF16)
        )
        # self.mlp contains te.Linear which will use FP8/NVFP4 inside fp8_autocast
        t_emb = self.mlp(t_freq)
        return t_emb

# --- LabelEmbedder (Remains Unchanged) ---
# nn.Embedding has no direct TE training equivalent. Runs in baseline precision.
class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        # Embedding table will remain in baseline precision (BF16/FP32)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        # Standard tensor ops, run in baseline precision
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
            # Ensure drop_ids is on the correct device BEFORE comparison
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        # Embedding lookup runs in baseline precision
        embeddings = self.embedding_table(labels)
        return embeddings


# --- FeedForward (No changes needed IF fused, see TransformerBlock) ---
# We will replace the LayerNorm + FeedForward combination in TransformerBlock
# with te.LayerNormMLP, so this standalone module isn't strictly needed
# in the TE-accelerated version *if* that fusion happens. If used standalone,
# you would replace nn.Linear with te.Linear here.
# For simplicity in this example, we assume fusion occurs in TransformerBlock.


# --- TransformerBlock ---
# Replace LayerNorms, Fuse FFN, Use TE Linear in AdaLN
class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
    ):
        super().__init__()
        self.dim = dim
        self.dim = dim
        self.head_dim = dim // n_heads
        # Attention Norm (Use TE LayerNorm)
        self.attention_norm = te.LayerNorm(dim, eps=norm_eps)
        self.attention = Attention(dim, n_heads, dim_head=self.head_dim)

        # Fused LayerNorm + MLP (Use TE LayerNormMLP)
        hidden_dim = int(4 * dim) # Standard calculation
        hidden_dim = int(2 * hidden_dim / 3) # Specific SwiGLU hidden dim calc
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.ffn = te.LayerNormMLP(
            hidden_size=dim,
            ffn_hidden_size=hidden_dim,
            eps=norm_eps,
            # TE supports fused SwiGLU ('swiglu' = SiLU * W3)
            # Make sure hidden_dim calculation matches TE's expectation for swiglu
            activation='swiglu',
            bias=True # Fused MLP usually includes biases
        )

        self.layer_id = layer_id

        # AdaLN Modulation (Use TE Linear)
        # Input dim for AdaLN modulation adjusted if needed, e.g., min(dim, 1024)
        adaln_input_dim = min(dim, 1024)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            # Use te.Linear for potential acceleration
            te.Linear(adaln_input_dim, 6 * dim, bias=True),
        )

    def forward(self, x, freqs_cis, adaln_input=None):
        if adaln_input is not None:
            # adaLN_modulation's te.Linear runs in low precision
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            # --- Attention Path ---
            # Modulate (BF16) -> TE LayerNorm (Low Precision) -> Attention (BF16) -> Add (Low Precision)
            norm_out = self.attention_norm(modulate(x, shift_msa, scale_msa)) # TE Norm
            attn_out = self.attention(norm_out, freqs_cis) # BF16 Attention
            h = x + gate_msa.unsqueeze(1) * attn_out

            # --- MLP Path ---
            # Modulate (BF16) -> TE LayerNormMLP (Low Precision) -> Add (Low Precision)
            # Note: TE LayerNormMLP includes the norm internally.
            # We modulate the input *before* passing it to the fused block.
            # IMPORTANT: Confirm if te.LayerNormMLP allows pre-modulated input
            # or if modulation needs adjustment for fused layer.
            # Assuming modulation happens *before* the fused block:
            modulated_h = modulate(h, shift_mlp, scale_mlp)
            mlp_out = self.ffn(modulated_h) # TE Fused Norm+MLP
            x = h + gate_mlp.unsqueeze(1) * mlp_out

        else: # Standard path without AdaLN
             # TE LayerNorm -> BF16 Attention -> Add
            h = x + self.attention(self.attention_norm(x), freqs_cis)
            # TE LayerNormMLP -> Add
            x = h + self.ffn(h) # Input 'h' goes directly to LayerNormMLP

        return x


# --- FinalLayer ---
# Replace LayerNorm and Linear
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # Use TE LayerNorm, note elementwise_affine=False is kept
        self.norm_final = te.LayerNorm(
             hidden_size, eps=1e-6
        )
        # Use TE Linear
        self.linear = te.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )

        # AdaLN Modulation (Use TE Linear)
        adaln_input_dim = min(hidden_size, 1024)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            te.Linear(adaln_input_dim, 2 * hidden_size, bias=True),
        )

        # Initialization remains the same for the TE Linear layer's parameters
        with torch.no_grad():
            nn.init.constant_(self.linear.weight, 0)
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        # adaLN_modulation's te.Linear runs in low precision
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)

        # TE LayerNorm (Low Precision) -> Modulate (BF16) -> TE Linear (Low Precision)
        x_norm = self.norm_final(x)
        x_mod = modulate(x_norm, shift, scale)
        x_out = self.linear(x_mod)
        return x_out


# --- DiT_Llama ---
# Replace x_embedder, use modified sub-modules
class DiT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=512,
        n_layers=5,
        n_heads=16,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=10,
        K=None, # For PiFlow output style
        # Assume Attention class is defined elsewhere
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.K = K # For PiFlow multiple output heads

        # Initial Convs and GroupNorm remain nn.Module (run in BF16)
        # No direct TE equivalent with acceleration benefits here.
        self.init_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
        )

        # Replace patch embedder's Linear layer
        self.x_embedder = te.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
        with torch.no_grad():
            nn.init.constant_(self.x_embedder.bias, 0)

        # Use the modified TimestepEmbedder
        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        # LabelEmbedder remains unchanged
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

        # Use the modified TransformerBlock
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                )
                for layer_id in range(n_layers)
            ]
        )

        # Use the modified FinalLayer
        if self.K is not None: # For PiFlow style output
            self.final_layer_A = FinalLayer(dim, patch_size, self.K)
            self.final_layer_u = FinalLayer(dim, patch_size, self.K * self.out_channels)
            self.final_layer_s = FinalLayer(dim, patch_size, 1) # Scaling factor head
        else: # Standard DiT output
            self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        # Precompute freqs_cis (remains unchanged, runs in baseline precision)
        self.freqs_cis = DiT_Llama.precompute_freqs_cis(dim // n_heads, 4096) # Adjust max seq len if needed

    # --- patchify, unpatchify, precompute_freqs_cis methods remain unchanged ---
    def unpatchify(self, x):
        # ... (implementation as before) ...
        c = self.out_channels if self.K is None else -1 # Determine C dynamically if K is used
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        # Adjust unpatchify logic if output channels change (e.g., PiFlow K)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1)) # Last dim is channels_out * K / K
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, w * p)) # Adjust shape if C is variable
        return imgs

    def patchify(self, x):
        # ... (implementation as before) ...
        B, C_in, H, W = x.size() # Use C_in from input
        p = self.patch_size
        x = x.view( B, C_in, H // p, p, W // p, p)
        # Calculate expected features for x_embedder
        expected_features = p * p * (self.x_embedder.in_features // (p*p)) # Infer Cin from x_embedder
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), expected_features)
        return x

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
      # ... (implementation as before) ...
      freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
      t = torch.arange(end) # Use arange(end)
      freqs = torch.outer(t, freqs).float()
      # freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # Requires PyTorch 1.8+
      # Alternative for older PyTorch:
      freqs_cos = torch.cos(freqs)
      freqs_sin = torch.sin(freqs)
      freqs_cis = torch.complex(freqs_cos, freqs_sin)
      return freqs_cis

    def forward(self, x, t, y):
        self.freqs_cis = self.freqs_cis.to(x.device) # Move freqs_cis to device

        # Store originals if using PiFlow output structure
        input_t = t.clone().detach() if self.K is not None else None
        input_x = x.clone().detach() if self.K is not None else None
        shape_x = x.shape # Store original shape

        # --- Input Path ---
        # Convs/GroupNorm (BF16) -> Patchify (BF16) -> x_embedder (TE Linear -> Low Precision)
        x = self.init_conv_seq(x)
        x = self.patchify(x)

        expected_dtype = next(self.x_embedder.parameters()).dtype
        x = x.to(expected_dtype)
        # --- End Verify ---
        x = self.x_embedder(x) # Error happens here

        # --- Embeddings ---
        # t_embedder (TE Linear -> Low Precision), y_embedder (nn.Embedding -> BF16)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        adaln_input = t_emb.to(x.dtype) + y_emb.to(x.dtype) # Combine embeddings (ensure matching dtype)

        # --- Transformer Layers ---
        # Each layer uses the modified TransformerBlock (TE Norms/MLPs, BF16 Attention)
        for layer in self.layers:
            # Pass correct slice of precomputed freqs_cis
            x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

        # --- Output Path ---
        # Uses modified FinalLayer (TE Norm/Linear)
        if self.K is None: # Standard DiT
            x = self.final_layer(x, adaln_input) # TE FinalLayer
            x = self.unpatchify(x) # BF16 unpatchify
            return x
        else: # PiFlow structure
            A = self.final_layer_A(x, adaln_input) # TE FinalLayer
            u = self.final_layer_u(x, adaln_input) # TE FinalLayer
            s = self.final_layer_s(x, adaln_input) # TE FinalLayer

            # Unpatchify and reshape (BF16 operations)
            A = self.unpatchify(A) # (N, K, H, W) -> Needs C=-1 in unpatchify
            u = self.unpatchify(u) # (N, C*K, H, W) -> Needs C=-1
            s = self.unpatchify(s) # (N, 1, H, W) -> Needs C=-1

            # Reshaping and activation (BF16 operations)
            B, _, H, W = shape_x # Use original shape
            C = self.out_channels
            A = A.reshape(B, self.K, 1, H, W).softmax(dim=1)
            u = u.reshape(B, C * self.K, H, W).reshape(B, self.K, C, H, W)
            s = s.reshape(B, 1, H, W).mean(dim=[-2,-1], keepdim=True).reshape(B, 1, 1, 1, 1)
            s = F.softplus(s)

            # Return dictionary (BF16/FP32 tensors)
            return {
                'A_s': A,
                'mu_s': u,
                'sigma_s': s,
                's': input_t[:, None, None, None, None].to(s.dtype), # Match precision
                'x_s': input_x[:, None, :, :, :].to(u.dtype), # Match precision
            }

    # forward_with_cfg remains conceptually the same, relies on forward()