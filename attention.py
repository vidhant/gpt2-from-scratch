#!/usr/bin/env python
# coding: utf-8


# see the colab step2-attention.ipynb for descriptions and examples for 
# invoking the classes.
import torch


class SelfAttention_v1(torch.nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values

        return context_vec

class SelfAttention_v2(torch.nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values =  self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values

        return context_vec


class CausalAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_fraction, kqv_bias=False):
        super().__init__()
        self.d_in = d_in  # say 3
        self.d_out = d_out  # say 2

        self.W_q = torch.nn.Linear(d_in, d_out, kqv_bias)  # 2x3
        self.W_k = torch.nn.Linear(d_in, d_out, kqv_bias)  # 2x3
        self.W_v = torch.nn.Linear(d_in, d_out, kqv_bias)  # 2x3

        self.dropout = torch.nn.Dropout(dropout_fraction)

        upper_triangular_matrix = torch.triu(
            # 0 will mean allowed, 1 will mean mask out
            # diagnonal itself won't be masked, only the future to the right
            # [[0 1 1 1]
            #  [0 0 1 1]
            #  [0 0 0 1]
            #  [0 0 0 0]]
            torch.ones(context_length, context_length),
            diagonal=1,
        )

        # Store the mask inside the model so PyTorch knows it belongs to the model.
        # This makes sure the mask is saved with the model and automatically moved
        # to the GPU/CPU together with the model. (A normal tensor would NOT do this.)
        self.register_buffer("mask", upper_triangular_matrix)

    def forward(self, inputs):
        B, num_tokens, d_in = inputs.shape

        q = self.W_q(inputs)  # B x T x d_out
        k = self.W_k(inputs)
        v = self.W_v(inputs)

        # q: B x T x d_out (batch x number of tokens x embedding dimension of k, q, v)
        # k.transpose(1,2): B x d_out x T
        # Hence, attn_scores will be B x T x T
        attn_scores = q @ k.transpose(1, 2)
        # [:num_tokens, :num_tokens] crops the full context-length mask down to the actual sequence length
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # the trailing underscore ensures the operation is performed in-place, avoiding
        # unnecessary memoery copies

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ v
        return context_vector


class MultiHeadAttentionWrapper(torch.nn.Module):

    def __init__(
        self, d_in, d_out, context_length, dropout_rate, number_of_heads, kqv_bias=False
    ):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout_rate, kqv_bias)
                for _ in range(number_of_heads)
            ]
        )

    def forward(self, input):
        # The heads are processed sequentially
        return torch.cat([head(input) for head in self.heads], dim=-1)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
