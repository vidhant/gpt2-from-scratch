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

Compared the inference time, with and without KV-caching, for a small 124M parameter GPT-2 model.

There was almost a 5x speed-up when generating 100 tokens from a 5-token prompt.

| Method    | Total Time | Tokens/Sec | Speedup |
|------------|-------------|-------------|----------|
| Standard   | 19.22s      | ~5.2 tps    | 1.0x     |
| KV-Cache   | 3.97s       | ~25.1 tps   | 4.8x     |


![](images/experiments-1.png)
Observations:
* In the non KV-cache method, the time to generate each token increases, whereas in the KV cache one, the time to generate each token remains the same

Downsides
* The bottleneck moves from inference time to how fast we can move the KV cache around on the GPU. Solution - Flash Attention, which I try next
---

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
3. Model Quantization
4. Flash Attention
5. LoRA


## How to run

`uv run jupyter lab`

## Acknowledgements

Most of the core LLM implementation in this repo follows `Build a Large Language Model` by `Sebastian Raschka`.

> Raschka, Sebastian. Build A Large Language Model (From Scratch). Manning, 2024. ISBN: 978-1633437166.
