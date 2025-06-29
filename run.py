import os
import sys

# Add the project root to the Python path
# This allows importing modules from 'agents' and 'trainer'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir) # Current directory is drl_chess_project
sys.path.insert(0, project_root)

from trainer.rl_trainer import RLTrainer

if __name__ == "__main__":
    # --- Configuration ---
    # Choose the agent type: "DQN" or "PPO"
    # Note: PPO is significantly harder to train and stabilize for complex environments
    # like chess, especially with this simplified implementation and on CPU.
    # Start with DQN for initial debugging.
    # SELECTED_AGENT_TYPE = "DQN" 
    SELECTED_AGENT_TYPE = "PPO" 

    
    # Training parameters
    NUM_EPISODES = 50000 # Small number for demonstration, increase for actual learning
    TRAIN_START_SIZE = 5000 # Number of experiences to collect before DQN starts training
                           # Reduced significantly for immediate feedback.
                           # For real learning, often thousands (e.g., 50000).
    BATCH_SIZE = 32        # Number of experiences per training step

    print(f"Starting DRL Chess Training with {SELECTED_AGENT_TYPE} Agent.")
    print(f"  Running for {NUM_EPISODES} episodes.")
    print(f"  {SELECTED_AGENT_TYPE} will start learning after {TRAIN_START_SIZE} experiences.")

    trainer = RLTrainer(
        agent_type=SELECTED_AGENT_TYPE,
        num_episodes=NUM_EPISODES,
        train_start_size=TRAIN_START_SIZE,
        batch_size=BATCH_SIZE,
        # render_mode="human"  # Change to "ansi" for non-visual output
    )

    trainer.train_agent()

    print("\n--- Program finished ---")
    print(f"Consider increasing NUM_EPISODES and TRAIN_START_SIZE for actual learning.")
    print(f"For PPO, training is much more complex and requires careful hyperparameter tuning.")
