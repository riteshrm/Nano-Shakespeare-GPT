# Nano Shakespeare GPT

Minimal PyTorch implementations of Bigram and GPT language models from scratch, with KV caching. Both models operate at the character level and are intended for learning and experimentation.

## Project Structure

- `gpt.py`: Implementation of the GPT model.
- `bigram.py`: Implementation of the Bigram language model.
- `benchmark.py`: KV-cache correctness, latency, throughput, and memory benchmark.
- `requirements.txt`: Python dependencies.
- `input.txt`: The training data, which is a collection of Shakespeare's works.

## Models

### Bigram Language Model

The `bigram.py` file contains a simple language model that predicts the next character based only on the previous character. This serves as a baseline to compare against the more complex GPT model.

### GPT Model

The `gpt.py` file contains the full implementation of a GPT-like model. The model architecture includes:

- Token and positional embeddings
- Multi-head self-attention blocks
- Layer normalization
- Feed-forward networks
- KV-cached autoregressive generation

## Usage

### Prerequisites

- Python 3.10 or newer
- An NVIDIA GPU is recommended for training

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

### Training

To train the models, you can run the Python scripts directly:

```bash
python bigram.py
python gpt.py
```

The scripts load `input.txt` and train the selected model. The GPT training script records the per-step training loss, periodic validation loss, and generated samples with Trackio.

Open the local Trackio dashboard with:

```bash
trackio show --project nanoGPT
```

### Generation

The `generate` method in both `BigramLanguageModel` and `GPTLanguageModel` can be used to generate new text. You can modify the code to generate text from a specific prompt.
- The `bigram.py` is a character-level model that, given a sequence of characters, aims to predict the next character in the sequence. It's trained on a large text corpus (`input.txt`) to learn the statistical relationships between characters.
- The `gpt.py` takes an input text, tokenizes it into a sequence of integers, and then feeds it into the model. The model then generates a sequence of new tokens, which are decoded back into text.

### Example after training

| Prompt | Generated text |
|---|---|
| `Hi there` | `Hi there, ladies Of men stony himself this wish; yet, if I do remember Give us a life to get thee to life an` |

## KV-Cache Benchmark

Run the benchmark with:

```bash
python benchmark.py
```

The following results were measured with a batch size of 1, an 8-token prompt, a context size of 128, and 256 generated tokens.

| Environment | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1050 |
| PyTorch | 2.6.0+cu118 |
| CUDA build | 11.8 |
| Precision | FP32 |

| Metric | Without KV Cache | With KV Cache | Difference |
|---|---:|---:|---:|
| Average latency | 5.683 s | 5.075 s | 10.7% lower |
| Throughput | 45.05 tokens/s | 50.44 tokens/s | 12.0% higher |
| Total peak memory | 54.45 MiB | 56.87 MiB | +2.42 MiB |
| Additional generation memory | 3.10 MiB | 5.52 MiB | +2.42 MiB |

This is approximately a **1.12x generation speedup** at the cost of additional GPU memory.

The generation length exceeds the 128-token context window. After the window fills, the learned absolute positional embeddings require the cache to be rebuilt from the cropped context, limiting the overall speedup. Benchmark results will vary across hardware and software environments.

## Dataset

The repository uses Andrej Karpathy's [Tiny Shakespeare dataset](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt), containing 40,000 lines from a selection of Shakespeare's works.

## Acknowledgements

This repository is based on Andrej Karpathy's [Neural Networks: Zero to Hero GPT lecture](https://github.com/karpathy/ng-video-lecture/tree/master) and uses the Tiny Shakespeare dataset from his [char-rnn repository](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).
