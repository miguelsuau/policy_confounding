# Breaking Habits
Source code for the paper "Breaking Habits: On the Role of the Advantage Function in Learning Causal State Representations"

## Installation

### Requirements
- Python 3.8

### Setup with conda (Recommended)
```bash
conda create -n breaking_habits python=3.8 
conda activate breaking_habits
pip install -r requirements.txt
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
python learn.py --config=./configs/keydoor/reinforce.yaml

# Run a default experiment
python learn.py
```

## Results and Visualization
Experiment results are tracked with MLflow and can be visualized using:

```bash
# Run the MLflow UI
mlflow ui
```