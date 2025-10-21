# NeuralNetwork

A C++20 framework for experimenting with neural networks and transformer-like language models written 
almost entirely from scratch for educational purposes, using only [xtensor](https://xtensor.readthedocs.io), 
a linear algebra framework. The project provides a modular, sequential deep learning framework built with custom 
layer implementations, utilities for working with MNIST, and a tiny Shakespeare LLM experiment.

## Features

- **Modular neural network core** – `NeuralNetwork` supports fully-connected, convolutional, residual, normalization, 
activation, attention, embedding, and dropout layers defined in [`layers/`](layers).
- **Training helpers** – Stochastic Gradient Descent, Adam, and AdamW optimizers with configurable schedules via the `TrainInfo` struct.
- **Vision example** – [`mnist/mnist.cpp`](mnist/mnist.cpp) demonstrates loading the `mnist-original.mat` dataset, 
building CNN/MLP models, visualizing predictions, and persisting checkpoints.
- **Language model demo** – [`llm/test.cpp`](llm/test.cpp) and [`llm/core.cpp`](llm/core.cpp) implement a decoder-only 
transformer based on the GPT-2 architecture that trains on `llm/data/tiny_shakespeare.txt`, saves checkpoints, and can generate sample text.

## Repository layout

```text
layers/          # Layer implementations (Dense, Convolution, Attention, etc.)
llm/             # Transformer language-model utilities and training entry point
mnist/           # MNIST data loader, demos, and visualization helpers
utilities.*      # Shared tensor utilities, activation functions, conversions
neural_network.* # Core training loop, loss evaluation, persistence helpers
main.cpp         # Calls train_llm() by default
```


## Toy LLM Experiment

The toy LLM follows the GPT-2 architecture, implementing a decoder-only transformer with dropout, 
a pre-LayerNorm scheme, and residual adds.

When trained over the tiny-shakespeare dataset using a character-based tokenizer, it produces basic coherency and 
illustrates some understanding of basic spelling and grammar.

Here is an example of what it generated at inference time:

```
GLOUCESTER:
What, beshrew his heart to march at thy summer studies
With hands of beauty of that in music
Writing the earth the heart stir against thy hardness;
And therefore from which the ground a crest;
That is no reasons so far as the morning:
Which over your father was fair a week.
```