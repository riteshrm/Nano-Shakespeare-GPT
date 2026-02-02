# GPT from scratch

This repository contains the code for building and training a Generative Pre-trained Transformer (GPT) model from scratch using PyTorch. The project also includes a simpler Bigram language model as a baseline.

## Project Structure

- `gpt.py`: Implementation of the GPT model.
- `bigram.py`: Implementation of the Bigram language model.
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

## Usage

### Prerequisites

- Python 3.x
- PyTorch

### Training

To train the models, you can run the Python scripts directly:

```bash
python bigram.py
python gpt.py
```

The scripts will load the `input.txt` file, train the models, and periodically print out the training and validation losses, along with some generated text.

### Generation

The `generate` method in both `BigramLanguageModel` and `GPTLanguageModel` can be used to generate new text. You can modify the code to generate text from a specific prompt.
- The `bigram.py` is a character-level model that, given a sequence of characters, aims to predict the next character in the sequence. It's trained on a large text corpus (`input.txt`) to learn the statistical relationships between characters.
- The `gpt.py` takes an input text, tokenizes it into a sequence of integers, and then feeds it into the model. The model then generates a sequence of new tokens, which are decoded back into text.

## Dataset

The `input.txt` file contains a collection of Shakespeare's works. This is used as the training data for the language models.

## Acknowledgements

This repository is based on the [nanoGPT video lecture series by Andrej Karpathy](https://github.com/karpathy/ng-video-lecture/tree/master).
