# DRL Chess Engine 🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![PettingZoo](https://img.shields.io/badge/PettingZoo-classic-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

A framework for training a chess-playing AI from scratch using Deep Reinforcement Learning with TensorFlow. This project provides a stable training environment implementing both Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents.

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Training](#running-the-training)
- [How It Works](#how-it-works)
- [Key Components](#key-components)
  - [The Environment](#the-environment)
  - [The Agents](#the-agents)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)

## Getting Started

Follow these steps to get the training environment up and running on your local machine.

### Prerequisites

- Python 3.8 or newer
- `pip` and `venv` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [[https://github.com/your-username/DRL_ChessEngine.git](https://github.com/your-username/DRL_ChessEngine.git)](https://github.com/Svar7769/DRL_ChessEngine.git)
    cd DRL_ChessEngine
    ```

2.  **Set up a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install tensorflow "pettingzoo[classic]" numpy tensorflow-probability
    ```

### Running the Training

The main entry point for the project is `run.py`.

1.  **Configure the training session** by editing the parameters at the top of the `run.py` file.

2.  **Execute the script from your terminal:**
    ```bash
    python run.py
    ```

You will see output in your console detailing the initialization and the progress of each training episode, including the agent's learning steps and final game rewards.

---

## How It Works

The project trains an AI agent (`player_0`) by having it play thousands of games of chess against a simple opponent (`player_1`) that selects its moves randomly from the set of legal options.

The core training logic is managed by the `RLTrainer` class, which follows this cycle for each game:
1.  **Initialize**: The chess environment is reset for a new game.
2.  **Observe**: The agent receives the current board state and a mask of all legal moves.
3.  **Act**: The agent uses its neural network to decide on the best action to take.
4.  **Store & Learn**: The `(state, action, reward, next_state)` transition is stored. The agent's neural network weights are updated based on the outcome of its actions. For DQN, this happens periodically from a replay buffer; for PPO, it occurs at the end of each game.
5.  **Repeat**: The process repeats, allowing the agent to gradually associate board states with high-reward outcomes, thereby learning chess strategy.

---

## Key Components

### The Environment
The project uses `pettingzoo.classic.chess_v6`, a standardized, turn-based chess environment. It provides:
- An 8x8x111 observation space, representing the board state in a format suitable for convolutional neural networks.
- A discrete action space of 4672 possible moves.
- An `action_mask` at every step to ensure the agent only considers legal moves.

### The Agents
You can choose between two different DRL agents, located in the `/agents` directory.

#### 1. DQN (`dqn_agent.py`)
A **Deep Q-Network** agent that learns the *value* of taking an action in a given state. This implementation is stabilized using:
- **Target Networks**: A separate, periodically updated network to provide stable targets during training, preventing exploding loss.
- **Gradient Clipping**: Prevents excessively large updates from destabilizing the network.

#### 2. PPO (`ppo_agent.py`)
A **Proximal Policy Optimization** agent that directly learns a *policy* for what action to take. It uses an Actor-Critic architecture, where:
- The **Actor** decides which action to take.
- The **Critic** evaluates how good the current state is.

---

## Configuration

All primary training parameters can be adjusted in `run.py`:

- `SELECTED_AGENT_TYPE`: `"DQN"` or `"PPO"`. DQN is recommended for initial tests as it is often simpler to stabilize.
- `NUM_EPISODES`: The total number of games to play. For meaningful learning, this should be in the thousands.
- `TRAIN_START_SIZE`: (DQN only) Number of experiences to collect before training begins.
- `BATCH_SIZE`: The number of experiences used in each learning step.

---


## Roadmap

Future enhancements for this project could include:
- [ ] **Model Checkpointing**: Save and load trained model weights.
- [ ] **TensorBoard Integration**: Log rewards and loss for visual analysis of training performance.
- [ ] **Advanced Opponents**: Implement self-play, where the agent trains against previous versions of itself.
- [ ] **Hyperparameter Optimization**: Use tools like KerasTuner or Optuna to find the best training parameters.
- [ ] **MCTS Integration**: Combine the neural network with a Monte Carlo Tree Search for an AlphaZero-style agent.

---
*This README was last updated on June 29, 2025.*

