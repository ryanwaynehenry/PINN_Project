# Navier-Stokes Model Trainer and Tester
This Python application is designed for training and testing a Physics Informed Neural Network based on the Navier-Stokes equations using PyTorch Lightning. The application predicts fluid dynamics variables such as velocity and pressure given spatial and temporal inputs. The script supports two optimizers, facilitates the loading and saving of trained models, and utilizes TensorBoard for logging training metrics.

## Table of Contents
- Dependencies
- Installation
- Usage
- Examples
- Command-Line Arguments
- Model Details
- Data Handling
- Outputs

## Dependencies
Ensure you have the following dependencies installed:

- Python 3.7 or higher
- PyTorch
- PyTorch Lightning
- NumPy
- Matplotlib
- SciPy
- scikit-learn
- imageio
- TensorBoard

## Installation
Before running the script, all necessary Python packages must be installed. Use pip to install the required packages:
```bash
pip install torch torchvision pytorch-lightning numpy matplotlib scipy scikit-learn imageio tensorboard
```
## Usage
Run the script from the terminal using the following command format. The script supports multiple command-line arguments to control various operations such as loading a model, saving a model, training, and setting the optimizer.
```bash
python main.py --load_file --load_filename your_model_path.pth --save_file --save_filename your_save_model_path.pth --train --num_nodes 20 --optimization LBFGS
```
## Examples
```bash
python main.py --load_file --load_filename model_plus10_adam_thenLBFGS.pth  --num_nodes 30 --optimization LBFGS
```
Original Architecture
```bash
python main.py --load_file --load_filename model_physics.pth  --num_nodes 20 --optimization LBFGS
```

## Command-Line Arguments
Detailed descriptions of the supported command-line arguments:

- --load_file: Flag to load a pre-trained model. Presence of this flag triggers the action. Only True if called
- --save_file: Flag to save the trained model after training. Only True if called
- --load_filename: Specify the path to the pre-trained model file to load.
- --save_filename: Specify the path where the trained model should be saved.
- --train: Flag to initiate model training. Only True if called
- --num_nodes: Set the number of nodes per hidden layer in the model.
- --optimization: Choose the type of optimizer to use (e.g., 'adam', 'LBFGS').

## Model Details
The model consists of multiple fully connected layers with Tanh activation functions. It computes a forward pass based on input features (x, y, t), calculates losses using physical laws derived from the Navier-Stokes equations, and uses gradient-based optimization techniques to minimize these losses.

## Data Handling
The application expects u, v, pressure, x, y, and t data in MATLAB .mat format. It processes this data to prepare input features and labels, handles training and testing splits, and supports efficient data loading for batch processing during training and evaluation phases.

## Outputs
The model generates two types of outputs in addition to saving the model. A gif file showing the output vs the desire pressure maps, and gradient error values to the terminal. The .gif file is saved to `PressureMap.gif`