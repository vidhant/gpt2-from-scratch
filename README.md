# llm-from-scratch

## Attention Module

1. Implemented **simple self-attention** on a sample input
    * Computed attention scores (Q@K.transpose)
    * Computed attention weights using (softmax(attn_score/k_d**-0.5))
    * Computed context vectors using attn_weights @ V
2. Next, I implemented simple **self-attention with trainable weights**
3. Next, I implemented **causal self-attention** so that only the preceeding and current tokens are given importance
4. Finally, I implemented **Multi-head attention** using both, stacking and weight splits.

### Experiments

1. Stacking multiple attention heads leads to a slower forward pass than weight-splitting them
2. Increasing the number of attention heads reduces loss, converges faster (perhaps with a ceiling?)
3. Weight-splitting MHA is better at memory utilization than stacking multiple heads


## How to run

`uv run jupyter lab`
