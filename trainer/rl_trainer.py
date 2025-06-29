import gymnasium as gym
import pettingzoo.classic.chess_v6 as chess_v6
import numpy as np
import random
import time
import collections

# Import your agent classes (make sure these files exist in the 'agents' directory)
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent

class RLTrainer:
    """
    Orchestrates the training of an RL agent in the PettingZoo Chess environment.
    Supports different agent types (DQN, PPO) and correctly handles the turn-based
    nature of the AEC API for experience replay.
    """
    def __init__(self, agent_type="DQN", num_episodes=100, train_start_size=1000, batch_size=64, render_mode="ansi"):
        self.agent_type = agent_type
        self.num_episodes = num_episodes
        self.train_start_size = train_start_size
        self.batch_size = batch_size
        self.render_mode = render_mode

        # env and agent are initialized in _initialize_environment_and_agent
        self.env = None
        self.agent = None
        self.learning_agent_id = 'player_0' # We will train this agent
        self.opponent_id = 'player_1'       # This agent will play random moves

        print(f"\n--- RL Trainer Initialized ---")
        print(f"  Agent Type: {self.agent_type}")
        print(f"  Total Episodes: {self.num_episodes}")

    def _initialize_environment_and_agent(self):
        """Initializes the PettingZoo environment and the selected agent."""
        self.env = chess_v6.env(render_mode=self.render_mode)
        # self.env = chess_v6.env(render_mode=self.render_mode)

        self.env.reset() # Reset once to populate spaces
        
        observation_space = self.env.observation_space(self.learning_agent_id)
        action_space = self.env.action_space(self.learning_agent_id)

        # The observation space for chess is a dictionary. The network needs the 'observation' part.
        observation_shape = observation_space['observation'].shape
        action_size = action_space.n
        
        print(f"DEBUG: Input shape for network: {observation_shape}")
        print(f"DEBUG: Action space size: {action_size}")

        if self.agent_type == "DQN":
            self.agent = DQNAgent(observation_shape, action_size)
        elif self.agent_type == "PPO":
            self.agent = PPOAgent(observation_shape, action_size)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    def train_agent(self):
        """Executes the main training loop for the selected RL agent."""
        self._initialize_environment_and_agent()
        
        episode_rewards = []

        print("\n--- Starting Training Loop ---")
        for episode in range(self.num_episodes):
            print(f"\n===== Episode {episode + 1}/{self.num_episodes} =====")
            
            self.env.reset(seed=random.randint(0, 100000))
            
            # --- CORRECTED LOGIC: Buffer to hold incomplete transitions ---
            # For a 2-player game, we can only complete player_0's transition
            # (S, A, R, S') after player_1 has made a move.
            # Key: agent_id, Value: (state, action, [value, log_prob for PPO])
            transition_buffer = {}

            if self.agent_type == "PPO":
                self.agent.clear_rollouts()

            # Main episode loop using PettingZoo's AEC API
            for agent_id in self.env.agent_iter():
                observation_dict, reward, terminated, truncated, info = self.env.last()
                
                # --- COMPLETE THE PREVIOUS TRANSITION (if any) ---
                # If it's our agent's turn, the 'reward' we just got is the result
                # of our previous action. Now we can form a full transition.
                if agent_id == self.learning_agent_id and agent_id in transition_buffer:
                    prev_state, prev_action, ppo_extras = transition_buffer[agent_id]
                    next_state = observation_dict['observation'] # This is S'
                    done = terminated or truncated

                    if self.agent_type == "DQN":
                        self.agent.store_experience(prev_state, prev_action, reward, next_state, done)
                    elif self.agent_type == "PPO":
                        prev_value, prev_log_prob = ppo_extras
                        self.agent.store_transition(prev_state, prev_action, reward, prev_value, prev_log_prob, done)
                    
                    # Train DQN agent if its buffer is full enough
                    if self.agent_type == "DQN" and len(self.agent.replay_buffer) > self.train_start_size:
                        loss = self.agent.learn(self.batch_size)
                        if loss is not None:
                            print(f"  [{agent_id}] DQN Learning Step. Loss: {loss:.4f}")

                # --- HANDLE GAME TERMINATION ---
                if terminated or truncated:
                    self.env.step(None) # Signal the env that this agent is done
                    continue # Skip to the next agent in agent_iter, which will also be done

                # --- CHOOSE AND EXECUTE ACTION ---
                legal_actions_mask = observation_dict["action_mask"]
                
                # It's possible for a player to have no legal moves (stalemate)
                if np.sum(legal_actions_mask) == 0:
                    self.env.step(None) # No legal action, agent must pass
                    continue

                state = observation_dict['observation']
                action = None
                
                if agent_id == self.learning_agent_id:
                    ppo_extras = (None, None)
                    if self.agent_type == "DQN":
                        action = self.agent.choose_action(state, legal_actions_mask)
                    elif self.agent_type == "PPO":
                        action, value, log_prob = self.agent.choose_action(state, legal_actions_mask)
                        ppo_extras = (value, log_prob)
                    
                    # Store the new, incomplete transition
                    transition_buffer[agent_id] = (state, action, ppo_extras)
                else: # The opponent plays randomly
                    action = self.env.action_space(agent_id).sample(legal_actions_mask)

                self.env.step(action)
            
            # --- EPISODE END ---

            # Handle the final transition of the game for the learning agent
            # This part is still correct and necessary
            if self.learning_agent_id in transition_buffer:
                prev_state, prev_action, ppo_extras = transition_buffer[self.learning_agent_id]
                # The game is over, so there is no 'next_state' from the agent's perspective.
                # The final reward for this last action is what matters.
                # We can get it directly from the last reward value received by the opponent.
                # However, the env.rewards dict should be populated after the episode is fully done.
                # To make this robust, let's use the reward attribute from the environment object itself.
                
                # *** CORRECTED LOGIC FOR FINAL REWARDS ***
                # The `env.rewards` dictionary is populated as the last step of an episode's lifecycle.
                # The most reliable way to get the final score is after the `agent_iter` loop is fully exhausted.
                final_reward_p0 = self.env.rewards.get(self.learning_agent_id, 0)
                final_reward_p1 = self.env.rewards.get(self.opponent_id, 0)

                done = True

                if self.agent_type == "DQN":
                    # For DQN, the next state can be treated as None or a zero-vector
                    self.agent.store_experience(prev_state, prev_action, final_reward_p0, np.zeros_like(prev_state), done)
                elif self.agent_type == "PPO":
                    prev_value, prev_log_prob = ppo_extras
                    # The final reward is the one that was received for the last action.
                    self.agent.store_transition(prev_state, prev_action, final_reward_p0, prev_value, prev_log_prob, done)

            # Perform PPO update at the end of the episode
            if self.agent_type == "PPO" and len(self.agent.states) > 0:
                actor_loss, critic_loss = self.agent.update()
                print(f"  [PPO Trainer] Episode Update. Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

            # Decay epsilon for DQN agent
            if self.agent_type == "DQN":
                self.agent.decay_epsilon()
            
            # Log final rewards using a safe .get() method
            # This is the corrected and safe way to access the final rewards
            final_reward_p0 = self.env.rewards.get(self.learning_agent_id, 0)
            final_reward_p1 = self.env.rewards.get(self.opponent_id, 0)
            episode_rewards.append(final_reward_p0)
            print(f"Episode {episode + 1} finished. Player_0 Reward: {final_reward_p0}, Player_1 Reward: {final_reward_p1}")

        # --- TRAINING END ---
        self.env.close()
        print("\n--- DRL Chess Training Completed ---")
        return episode_rewards

# Example of how to run the trainer
if __name__ == '__main__':
    # Make sure you have placeholder agent files:
    # agents/dqn_agent.py and agents/ppo_agent.py
    # This example will not run without them.
    
    try:
        # To run this, you need actual implementations of DQNAgent and PPOAgent
        trainer = RLTrainer(agent_type="DQN", num_episodes=50, render_mode=None)
        rewards = trainer.train_agent()
        print("\nRewards per episode:", rewards)
    except ImportError as e:
        print("\nCould not run trainer example.")
        print(f"Error: {e}")
        print("Please make sure you have 'agents/dqn_agent.py' and 'agents/ppo_agent.py' files with the required classes.")