# Faster Inference for LLaMA-3 on LMSYS Chatbot Arena

## Overview

This repository explores optimization strategies for efficient inference in **LLaMA-3 (8B)** within the scope of the [LMSYS Chatbot Arena](https://www.kaggle.com/competitions/lmsys-chatbot-arena/overview). The project benchmarks and refines decoding speeds for open-weight LLMs, enabling high-throughput generation ideal for real-time chatbot deployment.

## Objective

- Evaluate and improve inference performance of LLaMA-3 models using practical decoding optimizations.
- Achieve reduced latency while maintaining response quality in multi-turn dialog settings.
- Compete effectively in LMSYS Chatbot Arena by balancing speed, coherence, and competitiveness.

## Notebooks

- `llama-3-8b-38-faster-inference.ipynb`: Profiling of LLaMA-3-8B inference using `vLLM` with different decoding configurations.
- `llama3-faster-inference.ipynb`: Evaluation of dynamic token allocation strategies and comparison with greedy vs. sampling-based decoding.

## Key Techniques

- **vLLM** deployment with continuous batching.
- Comparison of decoding strategies:
  - Greedy vs. top-k/top-p sampling
  - Temperature scaling
  - Token streaming vs. full generation
- **Benchmarking throughput** using prompts inspired by real user queries from Chatbot Arena.

## Results

- Achieved inference speedup up to **2.5x** under optimized settings.
- Maintained win-rate parity with baseline models on simulated Arena matchups.
- Demonstrated how sampling strategies affect latency vs. response diversity trade-offs.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/faster-llama3-chatbot-arena.git

# Install dependencies
pip install -r requirements.txt
