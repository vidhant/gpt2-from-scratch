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
4. Vanishing gradients if we don't scale attention scores by square root of keys matrix.

## Blog ideas

* What is the semantic meaning of having residual connections?
* How gradients flow across the model (including through the different transformer blocks) and how new information gets added at each layer.
* Can we *see* what's flowing through the model or what it is "thinking"?
* How does data flow through the transformer, in terms of the original input x?


### TODO

1. KV-Cache Optimization
2. Speculative Decoding
3. Model Quantization


## How to run

`uv run jupyter lab`
