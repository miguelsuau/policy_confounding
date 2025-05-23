# Breaking Habits
Source code for the paper "Breaking Habits: On the Role of the Advantage Function in Learning Causal State Representations"

## Installation

### Requirements
- Python 3.8
- pip or pipenv

### Setup with Pipenv (Recommended)
```bash
# Install pipenv if you don't have it
pip install pipenv

# Install dependencies
pipenv install

# Activate the virtual environment
pipenv shell
```

### Manual Setup
```bash
pip install stable-baselines3[extra] sacred pymongo sshtunnel pyyaml
```

## Algorithms
The following reinforcement learning algorithms are implemented:
- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Network)
- REINFORCE

## Run Experiments

### Using Configuration Files
Experiments are configured using YAML files in the `configs` directory. Each environment has its own set of configurations.

```bash
# Run a single experiment
python learn.py --config=./configs/key_door/reinforce.yaml

# Run a default experiment
python learn.py
```

### Batch Experiments
To run multiple experiments with the same configuration:

```bash
# Run the experiment script
bash run_experiments.sh
```

You can modify `run_experiments.sh` to use different configuration files.

## Results and Visualization
Experiment results are tracked with MLflow and can be visualized using:

```bash
# Run the MLflow UI
mlflow ui
```

Additionally, the Jupyter notebook `plot_results_mlflow.ipynb` can be used to generate plots and analyze results.

## Project Structure
- `environments/`: Custom RL environments
- `configs/`: Configuration files for experiments
- `learn.py`: Main script for running experiments
- `custom_ppo.py`, `dqn.py`, `reinforce.py`: RL algorithm implementations
- `callback.py`: Callbacks for training and evaluation
- `plots/`: Directory for generated plots
- `mlruns/`: MLflow logging directory
