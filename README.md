# nnlab: Neural Network Lab

nnlab is a feedforward neural network library written in Python with NumPy as its only numerical dependency. It implements the full training pipeline from first principles: dense layers with forward and backward passes, activation functions, loss functions, weight initializers, optimizers with learning rate schedules, dropout and L2 regularization, mini-batch training with early stopping, model checkpointing, metrics, dataset utilities, JSON experiment configuration, and a command line interface for training, evaluation and benchmarking.

The project is intended for learning how neural networks work internally and for small-scale experiments on tabular and image-like data. Every gradient is computed by hand in the layer classes, and a gradient checker is included so the backward passes can be verified numerically.

## Features

- Dense layers with Xavier, He and LeCun initialization
- Activations: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax, Linear
- Losses: mean squared error, binary cross-entropy, categorical cross-entropy
- Optimizers: SGD with momentum and Nesterov, RMSProp, Adam, AdamW
- Learning rate schedules: constant, step decay, exponential decay, cosine annealing
- Regularization: inverted dropout and L2 weight decay
- Mini-batch training with shuffling, validation split, early stopping and best-weight restoration
- Save and load of model weights and architecture in NumPy archives
- Metrics: accuracy, precision, recall, F1, confusion matrix, mean squared error
- Data utilities: standardization, min-max scaling, one-hot encoding, stratified splitting, batching
- Built-in datasets: XOR, spirals, two moons, and a procedurally generated digits dataset
- Optional loader for the MNIST IDX format when the files are present locally
- JSON configuration files and a command line interface

## Requirements

- Python 3.9 or newer
- NumPy 1.24 or newer
- pytest for running the test suite (development only)

## Project Layout

- src/nnlab contains the library and the command line entry point
- tests contains the unit and integration tests
- configs contains experiment configurations in JSON
- scripts contains portable helper scripts used by the Makefile
- Makefile provides the same targets on macOS, Linux and Windows

## Setup

On macOS and Linux:

    make install

On Windows, run the same command from a shell where GNU Make is available, for example Git Bash, MSYS2, or Chocolatey's make package:

    make install

The install target creates a virtual environment in .venv, upgrades pip, installs the runtime dependencies and installs the package in editable mode. Use make dev instead to also install the test dependencies.

Without make, the equivalent steps are:

    python -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m pip install -e .

On Windows use .venv\Scripts\python.exe in place of .venv/bin/python.

## Running

Run the test suite:

    make test

Verify the backpropagation implementation against numerical gradients:

    make gradcheck

Train on the XOR problem:

    make xor

Train on the synthetic digits dataset:

    make digits

Train with any configuration file and output directory:

    make train CONFIG=configs/spirals.json OUTPUT=runs/spirals

Run every configuration in the configs directory and write a summary report:

    make benchmark

Remove generated artifacts:

    make clean

Remove generated artifacts together with the virtual environment:

    make distclean

The same commands are available through the package entry point after installation:

    nnlab train --config configs/digits.json --output runs/digits
    nnlab evaluate --model runs/digits/model.npz --config configs/digits.json
    nnlab predict --model runs/digits/model.npz --input samples.csv
    nnlab gradcheck
    nnlab benchmark --config-dir configs --output runs/benchmark

## Configuration

Each experiment is described by a JSON file. A configuration contains the dataset, the network architecture, the loss, the optimizer, the schedule, the training options and the random seed. The files in configs/ are complete examples and can be copied as a starting point for new experiments.

## Output

Each training run writes the following files into its output directory:

- model.npz with the trained weights and architecture
- history.json with per-epoch training and validation metrics
- summary.json with the final metrics and the run configuration
- metrics.txt with a readable report including the confusion matrix for classification tasks

## Testing

The tests cover activations, losses, layers, gradients, optimizers, schedules, data utilities, serialization, the training loop and the command line interface. Run them with make test on any supported platform.

## License

MIT
