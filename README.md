
# Custom RNN, LSTM, and GRU Implementation in PyTorch

This repository provides implementations from scratch of three common Recurrent Neural Network (RNN) architectures: vanilla RNN, LSTM, and GRU, using PyTorch. These implementations aim to practice coding RNN, LSTM, and GRU from scratch to gain a deeper understanding of how they work.

## Features

- **RNN Implementations**:
  - Vanilla RNN with support for `tanh` or `ReLU` activations.
  - Long Short-Term Memory (LSTM) network.
  - Gated Recurrent Unit (GRU) network.
- **Supports Bidirectionality**: All models support bidirectional processing.
- **Multi-Layered Architectures**: Configurable number of stacked layers.
- **Compatibility with PyTorch**: Includes utilities to copy parameters from PyTorch’s built-in RNN modules to these implementations for comparison.
- **Batch-First Input**: Allows inputs to be provided in either batch-first (`[batch, seq_len, features]`) or sequence-first (`[seq_len, batch, features]`) formats.

## File Overview

### `rnn.py`
This file contains:
- The base class `RNN_Base`, which provides shared functionality for RNN, LSTM, and GRU implementations.
- The `RNN` class, implementing a basic recurrent neural network with configurable activation (`tanh` or `ReLU`).

### `gru.py`
This file contains the `GRU` class:
- Implements the Gated Recurrent Unit architecture.
- Includes a implementation of GRU cells.
- Mimics the behavior of `torch.nn.GRU`.

### `lstm.py`
This file contains the `LSTM` class:
- Implements the Long Short-Term Memory architecture.
- Includes a implementation of LSTM cells.
- Mimics the behavior of `torch.nn.LSTM`.

### `utils.py`
This file contains utility functions:
- `copy_parameters`: A function to copy parameters from PyTorch’s built-in RNN modules to the custom implementations for result comparison and debugging.

## Usage

### Requirements
- Python 3.8+
- PyTorch 1.9+

### Running Tests
Each file includes a main section that tests the custom implementation by comparing its output to PyTorch's built-in modules. To run the tests:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Run the test for a specific model. For example:
   ```bash
   python gru.py
   ```

3. Check the output:
   If the custom implementation matches PyTorch’s outputs within a small tolerance, the test will pass.


## Contribution
Feel free to open issues or submit pull requests for bug fixes, feature requests, or improvements to the documentation.


## Acknowledgments
This implementation was inspired by the PyTorch deep learning framework. It is intended for learning purposes. For more details, refer to the [PyTorch RNN Documentation](https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#RNN).
