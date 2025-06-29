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
    def __init__(self, agent_type="DQN", num_episodes=100, train_start_size=1000, batch_size=64, render_mode="ansi"):
        self.agent_type = agent_type
        self.num_episodes = num_episodes
        self.train_start_size = train_start_size
        self.batch_size = batch_size
        self.render_mode = render_mode

        self.env = None
        self.agent = None
        self.learning_agent_id = 'player_0' 
        self.opponent_id = 'player_1'       

        print(f"\n--- RL Trainer Initialized ---")
        print(f"  Agent Type: {self.agent_type}")
        print(f"  Total Episodes: {self.num_episodes}")

    def _initialize_environment_and_agent(self):
        self.env = chess_v6.env(render_mode=self.render_mode)

        self.env.reset() 
        
        observation_space = self.env.observation_space(self.learning_agent_id)
        action_space = self.env.action_space(self.learning_agent_id)

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
        self._initialize_environment_and_agent()
        
        episode_rewards = []

        print("\n--- Starting Training Loop ---")
        for episode in range(self.num_episodes):
            print(f"\n===== Episode {episode + 1}/{self.num_episodes} =====")
            
            self.env.reset(seed=random.randint(0, 100000))
            
            transition_buffer = {}

            if self.agent_type == "PPO":
                self.agent.clear_rollouts()

            for agent_id in self.env.agent_iter():
                observation_dict, reward, terminated, truncated, info = self.env.last()
                
                if agent_id == self.learning_agent_id and agent_id in transition_buffer:
                    prev_state, prev_action, ppo_extras = transition_buffer[agent_id]
                    next_state = observation_dict['observation'] # This is S'
                    done = terminated or truncated

                    if self.agent_type == "DQN":
                        self.agent.store_experience(prev_state, prev_action, reward, next_state, done)
                    elif self.agent_type == "PPO":
                        prev_value, prev_log_prob = ppo_extras
                        self.agent.store_transition(prev_state, prev_action, reward, prev_value, prev_log_prob, done)
                    
                    if self.agent_type == "DQN" and len(self.agent.replay_buffer) > self.train_start_size:
                        loss = self.agent.learn(self.batch_size)
                        if loss is not None:
                            print(f"  [{agent_id}] DQN Learning Step. Loss: {loss:.4f}")

                if terminated or truncated:
                    self.env.step(None) 
                    continue 

                legal_actions_mask = observation_dict["action_mask"]
                
                if np.sum(legal_actions_mask) == 0:
                    self.env.step(None) 
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
                    
                    transition_buffer[agent_id] = (state, action, ppo_extras)
                else: 
                    action = self.env.action_space(agent_id).sample(legal_actions_mask)

                self.env.step(action)
            

            
            if self.learning_agent_id in transition_buffer:
                prev_state, prev_action, ppo_extras = transition_buffer[self.learning_agent_id]
            
                final_reward_p0 = self.env.rewards.get(self.learning_agent_id, 0)
                final_reward_p1 = self.env.rewards.get(self.opponent_id, 0)

                done = True

                if self.agent_type == "DQN":
                    self.agent.store_experience(prev_state, prev_action, final_reward_p0, np.zeros_like(prev_state), done)
                elif self.agent_type == "PPO":
                    prev_value, prev_log_prob = ppo_extras
                    self.agent.store_transition(prev_state, prev_action, final_reward_p0, prev_value, prev_log_prob, done)

            if self.agent_type == "PPO" and len(self.agent.states) > 0:
                actor_loss, critic_loss = self.agent.update()
                print(f"  [PPO Trainer] Episode Update. Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

            if self.agent_type == "DQN":
                self.agent.decay_epsilon()
            
            final_reward_p0 = self.env.rewards.get(self.learning_agent_id, 0)
            final_reward_p1 = self.env.rewards.get(self.opponent_id, 0)
            episode_rewards.append(final_reward_p0)
            print(f"Episode {episode + 1} finished. Player_0 Reward: {final_reward_p0}, Player_1 Reward: {final_reward_p1}")

        self.env.close()
        print("\n--- DRL Chess Training Completed ---")
        return episode_rewards

if __name__ == '__main__':
    
    try:
        trainer = RLTrainer(agent_type="DQN", num_episodes=50, render_mode=None)
        rewards = trainer.train_agent()
        print("\nRewards per episode:", rewards)
    except ImportError as e:
        print("\nCould not run trainer example.")
        print(f"Error: {e}")
        print("Please make sure you have 'agents/dqn_agent.py' and 'agents/ppo_agent.py' files with the required classes.")