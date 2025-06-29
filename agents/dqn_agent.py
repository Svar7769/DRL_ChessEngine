import numpy as np
import random
import collections
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DQNAgent:
    """
    A Deep Q-Network (DQN) agent with a Target Network and Gradient Clipping
    for stable training.
    """
    def __init__(self, observation_shape, action_size, 
                 learning_rate=0.00025,       # <-- CHANGED: Lowered learning rate
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_decay=0.999,      # <-- CHANGED: Slower decay for more episodes
                 epsilon_min=0.01,
                 replay_buffer_size=20000,   # <-- CHANGED: Slightly larger buffer
                 target_update_freq=500):    # <-- NEW: Frequency to update the target network (in steps)

        self.observation_shape = observation_shape
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq # <-- NEW
        self.update_counter = 0                      # <-- NEW: Counter for target network updates

        self.replay_buffer = collections.deque(maxlen=replay_buffer_size)

        # Main "online" network for predicting Q-values and choosing actions
        self.q_network = self._build_q_network()
        # Separate "target" network for calculating stable target Q-values
        self.target_network = self._build_q_network() # <-- NEW
        # Initialize target network with the same weights as the online network
        self.target_network.set_weights(self.q_network.get_weights()) # <-- NEW

        # Optimizer with Gradient Clipping
        self.optimizer = optimizers.Adam(learning_rate=self.lr, clipvalue=1.0) # <-- CHANGED: Added clipvalue
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        print("\n--- DQN Agent Initialized (with Target Network & Gradient Clipping) ---")
        print(f"  Hyperparameters: LR={self.lr}, Gamma={self.gamma}, Target Update Freq: {self.target_update_freq}")
        self.q_network.summary()

    def _build_q_network(self):
        # (This function remains unchanged)
        model = models.Sequential([
            layers.Input(shape=self.observation_shape),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ], name="DQN_Q_Network")
        return model

    def store_experience(self, state, action, reward, next_state, terminated):
        # (This function remains unchanged)
        self.replay_buffer.append((state, action, reward, next_state, terminated))

    def choose_action(self, state, legal_actions_mask):
        # (This function remains unchanged)
        legal_indices = np.where(legal_actions_mask == 1)[0]
        if len(legal_indices) == 0: return None
        if random.random() < self.epsilon:
            return random.choice(legal_indices)
        else:
            obs_tensor = tf.convert_to_tensor(state[np.newaxis, ...], dtype=tf.float32)
            q_values = self.q_network(obs_tensor).numpy().flatten()
            q_values[legal_actions_mask == 0] = -1e10
            action = np.argmax(q_values)
            if not legal_actions_mask[action]:
                action = random.choice(legal_indices)
            return action

    def learn(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return None

        batch = random.sample(self.replay_buffer, batch_size)
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        terminateds = np.array([exp[4] for exp in batch])

        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        terminateds_tensor = tf.convert_to_tensor(terminateds, dtype=tf.float32)

        # --- THIS IS THE KEY CHANGE ---
        # Calculate target Q-values using the STABLE target_network
        next_q_values = self.target_network(next_states_tensor) # <-- CHANGED: Use target_network
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        target_q_values = rewards_tensor + self.gamma * max_next_q * (1 - terminateds_tensor)

        with tf.GradientTape() as tape:
            current_q_values_full = self.q_network(states_tensor)
            action_indices = tf.stack([tf.range(batch_size), actions_tensor], axis=1)
            current_q_values = tf.gather_nd(current_q_values_full, action_indices)
            loss = self.loss_fn(target_q_values, current_q_values)
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Increment counter and update target network if it's time
        self.update_counter += 1 # <-- NEW
        if self.update_counter % self.target_update_freq == 0: # <-- NEW
            self._update_target_network()

        return loss.numpy()
        
    def _update_target_network(self): # <-- NEW
        """Copies the weights from the main q_network to the target_network."""
        print("--- Updating target network ---")
        self.target_network.set_weights(self.q_network.get_weights())

    def decay_epsilon(self):
        # (This function remains unchanged)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)