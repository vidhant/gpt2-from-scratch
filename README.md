# GPT 2.0 from Scratch

This repository contains the implementation of a 124M parameter GPT architecture trained on 'The Verdict' by Edith Wharton.


* Author: Vidhant Maini
* Timeline: Late 2025 - Mid 2026

## Features
- **Hardware-Agnostic:** Support for CUDA, MPS (Apple Silicon), and CPU.
- **Efficient Data Loading:** Implemented a sliding window buffer with configurable stride (currently 256 for zero-overlap).
- **Advanced Metrics:** Integrated Perplexity tracking ($e^{Loss}$) for better interpretability of model confidence.

## Input Data

* To make it possible to train the model on my personal Macbook Pro, I used a toy dataset - 'The Verdict' by Edith Wharton containing 5120 tokens. However, the same code should work for larger models as well (only that it'll take more time!)
* The train-validation split was 90-10.

## Model Architecture

![](images/gpt-2-architecture.png)

https://excalidraw.com/#json=kEtqwgDj8AgcazuEOskfg,y6R5g3fUyITXRsBLGTsqIQ

## Key Params

* 124M parameters
* 12 transformer layers
* 12 multi head attention modules within each transformer
* GeLU activation for the Feed Forward Network
* Context window: 256 tokens
* Optimization: AdamW with a learning rate of 0.0004 and 0.1 weight decay.
* Embedding dimension: 768
* Stride: 256 (Zero-overlap for maximum data efficiency).
* Tokenizer: Tiktoken (`tiktoken.get_encoding("gpt2")`)

## Training Performance
The model was trained for 10 epochs using the AdamW optimizer.

It took 13 minutes on my 2019, 1.4 GHz Quad-Core Intel Core i5 Macbook Pro

![](images/model-training-6.png)

![](images/model-training-7.png)


## Sample Output
* **Prompt:** "Every effort moves you"
* **Generated:** "Every effort moves you?" "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"

Yes, the generated output doesn't quite match up to a modern LLM, but:
* the architecure follows the real GPT-2.0 closely
* was trained on a much smaller dataset to make it possible on my personal macbook pro
* is not instruction tuned

## Experiments

### 1. KV Caching

#### KV Caching makes inference faster

Compared the inference time, with and without KV-caching, for a small 124M parameter GPT-2 model.

![](images/experiments-1.png)

**How it works:**
* **Standard inference:** At every decode step $i$, the model re-processes the entire sequence of length $P + i$ through attention ($P$ = prompt length). Total cost across all $G$ steps: $\sum_{i=0}^{G-1}(P+i)^2$, i.e. $O(n^2)$.
* **KV-Cache inference:** The prompt is processed once (prefill) and its $K$/$V$ tensors are stored. Each decode step processes only the 1 new token, attending over the growing cache. Total cost: $P^2 + \sum_{i=0}^{G-1}(P+i)$, i.e. $O(n)$.

**What actually drives speedup?**

Counterintuitively, `total_tokens = prompt_len + gen_len` is *not* the right predictor. The primary driver is **gen_len ($G$)**, because savings only accumulate during the $G$ decode steps — the prompt is processed once in both approaches and largely cancels in the ratio. Verified empirically:

| Config (P, G) | Total tokens | Theoretical speedup |
|---|---|---|
| prompt=10, gen=50 | 60 | ~38x |
| prompt=10, gen=200 | 210 | ~173x |
| prompt=200, gen=50 | 250 | ~50x |
| prompt=50, gen=200 | 250 | ~159x |

Two configs with the same total tokens (250) differ by **3x** in speedup, purely because gen_len differs.

**Theory vs. practice anomaly:**

Empirically, `prompt=200, gen=50` showed *higher* speedup than `prompt=50, gen=200`, contradicting the theoretical prediction. Each KV-cache decode step carries a fixed per-step overhead (Python loop, PyTorch dispatch, memory allocation) independent of sequence length. With $G=200$ steps this overhead accumulates 4× more than with $G=50$. Since KV-cache's FLOP cost per step is small, the fixed overhead becomes a proportionally larger fraction — diluting the speedup when $G$ is large. This gap closes on GPUs with large models where compute dominates overhead.

**Downsides:**
* Standard inference is compute-bound. KV-caching shifts the bottleneck to **memory bandwidth** — the cache must be streamed from VRAM at every decode step. Flash Attention addresses helps with this, but since it's a CUDA kernel optimization, it provides no benefit on CPU.

| Feature | Standard Inference | KV-Cache Inference |
|---|---|---|
| Computation | Recomputes all previous tokens | Computes only the new token |
| Time per Token | Increases per step | Constant |
| Complexity | $O(n^2)$ | $O(n)$ |
| Primary Bottleneck | GPU Compute (FLOPs) | Memory Bandwidth (IO) |
---

#### KV Cache costs Memory

This is the core tradeoff: KV-caching exchanges memory (linear growth) for compute (avoiding quadratic recomputation).

##### Theoretical size for the 124M param model

Each layer stores K and V tensors of shape `(batch, n_heads, seq_len, head_dim)`. For a single new token:

```
bytes_per_token = 2 (K+V) × 12 (layers) × 12 (heads) × 64 (head_dim) × 4 (float32) = 73,728 bytes ≈ 72 KB
```

For full context (256 tokens) = `72 KB × 256 ≈ 18 MB`

![](images/experiments-2.png)

* The KV cache size is **fully predictable** — measured and theoretical lines overlap exactly.
* Grows linearly with sequence length, capped at `context_length` tokens.
* In production (e.g. 70B param model, float16, batch=32, 128K context), the KV cache can exceed hundreds of GB.

<!-- TODO: add a plot showing projected cache size for larger models (7B, 70B) to make the scaling concrete -->

<!-- TODO: Architecture Decisions section — explain WHY each choice was made:
  - Pre-norm (LayerNorm before attention) vs post-norm: pre-norm stabilises training at depth
  - GELU vs ReLU: GELU is smoother, empirically better for transformers
  - Weight tying (out_head shares weights with token embedding): reduces params, regularises, aligns embedding/unembedding spaces
  - No bias in QKV projections: reduces overfitting, common in modern LLMs
  - AdamW over Adam: decoupled weight decay avoids L2-on-adaptive-lr interaction
-->

<!-- TODO: Sampling Strategies section — temperature, top-k, top-p:
  - Temperature: scales logits before softmax — lower = more deterministic, higher = more random
  - Top-k: truncate to k highest-probability tokens before sampling
  - Top-p (nucleus): truncate to smallest set of tokens whose cumulative prob ≥ p
  - Show how output quality changes across (temp=0.1, top_p=0.9) vs (temp=1.0, top_k=50)
-->

## Attention Module

1. Implemented **simple self-attention** on a sample input
    * Computed attention scores (Q@K.transpose)
    * Computed attention weights using (softmax(attn_score/k_d**-0.5))
    * Computed context vectors using attn_weights @ V
2. Next, I implemented simple **self-attention with trainable weights**
3. Next, I implemented **causal self-attention** so that only the preceeding and current tokens are given importance
4. Finally, I implemented **Multi-head attention** using both, stacking and weight splits.


### 2. Speculative Decoding

TODO

### 3. Logit Lens

TODO

### 4. Visualizing Attention

TODO


### 5. PeFT

TODO

### 6. Other ideas

1. Stacking multiple attention heads leads to a slower forward pass than weight-splitting them
2. Increasing the number of attention heads reduces loss, converges faster (perhaps with a ceiling?)
3. Weight-splitting MHA is better at memory utilization than stacking multiple heads
4. Vanishing gradients if we don't scale attention scores by square root of keys matrix.
5. **Logit lens - Prove that "Reasoning" happens in the middle layers, while "Grammar" happens in the later layers.**
6. Changing learning rate and weight decay of AdamW

## Blog ideas

* What is the semantic meaning of having residual connections?
* How gradients flow across the model (including through the different transformer blocks) and how new information gets added at each layer.
* Can we *see* what's flowing through the model or what it is "thinking"?
* How does data flow through the transformer, in terms of the original input x?


### TODO

1. KV-Cache Optimization
2. Speculative Decoding
3. LoRA
4. Model Quantization
5. Flash Attention


## How to run

`uv run jupyter lab`

## Acknowledgements

Most of the core LLM implementation in this repo follows `Build a Large Language Model` by `Sebastian Raschka`.

> Raschka, Sebastian. Build A Large Language Model (From Scratch). Manning, 2024. ISBN: 978-1633437166.
