import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
import tensorflow_probability as tfp 

tfd = tfp.distributions

class PPOAgent:
    """
    A simplified Proximal Policy Optimization (PPO) agent.
    Implements an Actor-Critic architecture.
    """
    def __init__(self, observation_shape, action_size,
                 actor_lr=0.0003, critic_lr=0.001,
                 gamma=0.99, lambda_gae=0.95, clip_ratio=0.2):
        
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma          
        self.lambda_gae = lambda_gae 
        self.clip_ratio = clip_ratio 
        
        self.actor = self._build_actor_network()
        self.critic = self._build_critic_network()

        self.actor_optimizer = optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=self.critic_lr)
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        self.states, self.actions, self.rewards, self.values, self.log_probs, self.terminateds = [],[],[],[],[],[]

        print("\n--- PPO Agent Initialized ---")
        print(f"  Observation Shape: {self.observation_shape}, Action Size: {self.action_size}")
        print(f"  Hyperparameters: Actor LR={self.actor_lr}, Critic LR={self.critic_lr}, Gamma={self.gamma}, Lambda={self.lambda_gae}, Clip Ratio={self.clip_ratio}")
        print("  Actor Network Summary:")
        self.actor.summary()
        print("  Critic Network Summary:")
        self.critic.summary()

    def _build_actor_network(self):
        input_layer = layers.Input(shape=self.observation_shape)
        conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
        bn1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(bn1)
        bn2 = layers.BatchNormalization()(conv2)
        conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(bn2)
        bn3 = layers.BatchNormalization()(conv3)
        flatten = layers.Flatten()(bn3)
        dense1 = layers.Dense(512, activation='relu')(flatten)
        dropout1 = layers.Dropout(0.2)(dense1)
        dense2 = layers.Dense(256, activation='relu')(dropout1)
        policy_logits = layers.Dense(self.action_size, activation='linear', name='policy_logits')(dense2)
        model = models.Model(inputs=input_layer, outputs=policy_logits, name="Actor_Network")
        return model

    def _build_critic_network(self):
        input_layer = layers.Input(shape=self.observation_shape)
        conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
        bn1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(bn1)
        bn2 = layers.BatchNormalization()(conv2)
        conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(bn2)
        bn3 = layers.BatchNormalization()(conv3)
        flatten = layers.Flatten()(bn3)
        dense1 = layers.Dense(512, activation='relu')(flatten)
        dropout1 = layers.Dropout(0.2)(dense1)
        dense2 = layers.Dense(256, activation='relu')(dropout1)
        state_value = layers.Dense(1, activation='linear', name='state_value')(dense2)
        model = models.Model(inputs=input_layer, outputs=state_value, name="Critic_Network")
        return model

    def choose_action(self, state, legal_actions_mask): # <-- CHANGED: Parameter name reflects it's a NumPy array
        """
        Selects an action based on the Actor network's policy, respecting legal moves.
        Also returns the state value predicted by the Critic and the log probability of the chosen action.
        
        Args:
            state (np.array): The board observation NumPy array.
            legal_actions_mask (np.array): Boolean mask indicating legal actions.
            
        Returns:
            tuple: (action, value, log_prob_action).
        """
        obs_tensor = tf.convert_to_tensor(state[np.newaxis, ...], dtype=tf.float32) # <-- CHANGED: Use `state` directly

        policy_logits = self.actor(obs_tensor)
        value = self.critic(obs_tensor).numpy()[0, 0]

        masked_logits = tf.where(legal_actions_mask[np.newaxis, :] == 1, policy_logits, -1e10)
        action_dist = tfd.Categorical(logits=masked_logits)
        
        action = action_dist.sample().numpy()[0]
        log_prob_action = action_dist.log_prob(action).numpy()[0]

        if not legal_actions_mask[action]:
            print(f"CRITICAL ERROR: PPO chose illegal action {action} despite masking! Falling back to random legal.")
            legal_indices = np.where(legal_actions_mask == 1)[0]
            if len(legal_indices) > 0:
                action = np.random.choice(legal_indices)
                log_prob_action = tfd.Categorical(logits=masked_logits).log_prob(action).numpy()[0]
            else:
                return None, None, None

        return action, value, log_prob_action

    def store_transition(self, state, action, reward, value, log_prob, terminated):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.terminateds.append(terminated)

    def update(self, batch_size=64, num_epochs=3):
        if len(self.states) == 0:
            return None, None

        # Prepare data from stored rollouts
        states_obs = np.array(self.states, dtype=np.float32) # <-- CHANGED: Directly use the list of arrays
        actions = np.array(self.actions, dtype=np.int32)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)
        
        returns = []
        R = 0
        for r, t in zip(reversed(self.rewards), reversed(self.terminateds)):
            R = r + self.gamma * R * (1 - t) 
            returns.insert(0, R) 
        returns = np.array(returns, dtype=np.float32)

        current_values = self.critic(states_obs).numpy().flatten()
        advantages = returns - current_values
        
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        states_tensor = tf.convert_to_tensor(states_obs, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=np.int32)
        old_log_probs_tensor = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        total_actor_loss = 0
        total_critic_loss = 0

        for _ in range(num_epochs):
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(self.states), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                batch_states = tf.gather(states_tensor, batch_indices)
                batch_actions = tf.gather(actions_tensor, batch_indices)
                batch_old_log_probs = tf.gather(old_log_probs_tensor, batch_indices)
                batch_advantages = tf.gather(advantages_tensor, batch_indices)
                batch_returns = tf.gather(returns_tensor, batch_indices)

                with tf.GradientTape() as tape:
                    predicted_values = self.critic(batch_states)
                    predicted_values = tf.squeeze(predicted_values, axis=-1)
                    critic_loss = self.mse_loss(batch_returns, predicted_values)
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                total_critic_loss += critic_loss.numpy()

                with tf.GradientTape() as tape:
                    current_policy_logits = self.actor(batch_states)
                    current_action_dist = tfd.Categorical(logits=current_policy_logits)
                    current_log_probs = current_action_dist.log_prob(batch_actions)
                    
                    ratio = tf.exp(current_log_probs - batch_old_log_probs)
                    
                    pg_loss1 = batch_advantages * ratio
                    pg_loss2 = batch_advantages * tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    actor_loss = -tf.reduce_mean(tf.minimum(pg_loss1, pg_loss2))

                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                total_actor_loss += actor_loss.numpy()
        
        self.clear_rollouts()
        return total_actor_loss, total_critic_loss

    def clear_rollouts(self):
        self.states, self.actions, self.rewards, self.values, self.log_probs, self.terminateds = [],[],[],[],[],[]