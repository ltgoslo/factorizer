import math
from typing import Optional, Final

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

from apex_wrapper import LayerNorm


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, dropout, attention_dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(hidden_size, num_heads, dropout, attention_dropout)
            for _ in range(num_layers)
        )
        self.output_norm = LayerNorm(hidden_size, elementwise_affine=False)

        for i, layer in enumerate(self.layers):
            layer.feed_forward.linear_1.weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.feed_forward.linear_2.weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

    def forward(self, x, padding_mask):
        for layer in self.layers:
             x = layer(x, padding_mask)
        x = self.output_norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, dropout, attention_dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(hidden_size, num_heads, dropout, attention_dropout)
            for _ in range(num_layers)
        )
        self.output_norm = LayerNorm(hidden_size, elementwise_affine=False)
        for i, layer in enumerate(self.layers):
            layer.feed_forward.linear_1.weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.feed_forward.linear_2.weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

    def forward(self, x, context):
        attention_mask = torch.triu(
            torch.full((x.size(0), x.size(0) + 3), True, dtype=torch.bool, device=x.device),
            diagonal=1 + 3
        )
        for layer in self.layers:
            x = layer(x, context, attention_mask)
        x = self.output_norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, attention_dropout):
        super().__init__()
        self.self_layer_norm_pre = LayerNorm(hidden_size, elementwise_affine=False)
        self.self_attention = AttentionLayer(hidden_size, num_heads, dropout, attention_dropout)
        self.self_layer_norm_post = LayerNorm(hidden_size, elementwise_affine=True)

        self.self_ff_layer_norm_pre = LayerNorm(hidden_size, elementwise_affine=False)
        self.feed_forward = FeedForward(hidden_size, dropout)

    def forward(self, x, padding_mask):
        query = key = value = self.self_layer_norm_pre(x)
        x = x + self.self_layer_norm_post(self.self_attention(query, key, value, padding_mask, None))
        x = x + self.feed_forward(self.self_ff_layer_norm_pre(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, attention_dropout):
        super().__init__()
        self.self_layer_norm_pre = LayerNorm(hidden_size, elementwise_affine=False)
        self.self_attention = AttentionLayer(hidden_size, num_heads, dropout, attention_dropout)
        self.self_layer_norm_post = LayerNorm(hidden_size, elementwise_affine=True)

        self.self_ff_layer_norm_pre = LayerNorm(hidden_size, elementwise_affine=False)
        self.feed_forward = FeedForward(hidden_size, dropout)

    def forward(self, x, context, attention_mask):
        query = self.self_layer_norm_pre(x)
        key = value = torch.cat([context, query], dim=0)

        x = x + self.self_layer_norm_post(self.self_attention(query, key, value, None, attention_mask))
        x = x + self.feed_forward(self.self_ff_layer_norm_pre(x))

        return x


# Follows paper "GLU Variants Improve Transformer": https://arxiv.org/abs/2002.05202
class FeedForward(jit.ScriptModule):
    hidden_size: Final[int]
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(hidden_size, hidden_size * 4 // 3 * 2, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size * 4 // 3, elementwise_affine=False)
        self.linear_2 = nn.Linear(hidden_size * 4 // 3, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.linear_1.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.linear_2.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    @jit.script_method
    def forward(self, x):
        x = self.linear_1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.layer_norm(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, attention_dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=attention_dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.attention.in_proj_weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.attention.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.attention.in_proj_bias.data.zero_()
        self.attention.out_proj.bias.data.zero_()

    def forward(self, query, key, value, key_padding_mask: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]):
        x, _ = self.attention(query, key, value, key_padding_mask, attn_mask=attention_mask, need_weights=False)
        x = self.dropout(x)
        return x
