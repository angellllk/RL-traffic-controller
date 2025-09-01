import os
import sys
import traci
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Te rog declară variabila de mediu 'SUMO_HOME'")

GUI_MODE = False 
SUMOCFG_FILE = "sim.sumocfg"
sumoBinary = "sumo-gui" if GUI_MODE else "sumo"
sumoCmd = [sumoBinary, "-c", SUMOCFG_FILE, "--start", "--no-warnings", "true"]

# Semaphore Data
TLS_ID = "cluster_286836527_286836866_286837118_286837205"

PHASE_EW_STRAIGHT_GREEN = 0
PHASE_EW_LEFT_GREEN     = 2
PHASE_NORD_GREEN        = 4
PHASE_SUD_GREEN         = 6
GREEN_PHASES = [PHASE_EW_STRAIGHT_GREEN, PHASE_EW_LEFT_GREEN, PHASE_NORD_GREEN, PHASE_SUD_GREEN]

YELLOW_PHASES = {
    PHASE_EW_STRAIGHT_GREEN: 1,
    PHASE_EW_LEFT_GREEN:     3,
    PHASE_NORD_GREEN:        5,
    PHASE_SUD_GREEN:         7
}

YELLOW_DURATION = 3
MIN_GREEN_DURATION = 20

INCOMING_LANES = [
    "302795167#0_0", "302795167#0_1", "302795167#0_2",
    "242364261_0", "242364261_1",
    "678011317_0", "678011317_1",
    "26195549#0_0", "26195549#0_1",
]

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Construieste o retea Dueling DQN."""
        input_layer = Input(shape=(self.state_size,))
        common_layer1 = Dense(64, activation='relu')(input_layer)
        common_layer2 = Dense(64, activation='relu')(common_layer1)
        
        value_stream = Dense(1, activation='linear')(common_layer2)
        advantage_stream = Dense(self.action_size, activation='linear')(common_layer2)
        
        def combine_streams(streams):
            v, a = streams
            return v + (a - tf.reduce_mean(a, axis=1, keepdims=True))

        output_layer = Lambda(combine_streams)([value_stream, advantage_stream])
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                best_action_next_state = np.argmax(self.model.predict(next_state, verbose=0)[0])
                q_value_next_state = self.target_model.predict(next_state, verbose=0)[0][best_action_next_state]
                target = reward + self.gamma * q_value_next_state
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)

# Helpers
def get_state():
    """Get state based on waiting time."""
    state = [traci.lane.getWaitingTime(lane_id) for lane_id in INCOMING_LANES]
    return np.reshape(state, [1, len(INCOMING_LANES)])

def calculate_reward(state_before, state_after):
    """Calculate reward based on the waiting time."""
    wait_time_before = np.sum(state_before)
    wait_time_after = np.sum(state_after)
    
    return wait_time_before - wait_time_after

# Function to run a training episode
def run_episode(agent):
    traci.start(sumoCmd)
    
    total_episode_reward = 0
    done = False
    
    current_green_phase = GREEN_PHASES[0]
    traci.trafficlight.setPhase(TLS_ID, current_green_phase)

    for _ in range(MIN_GREEN_DURATION):
        traci.simulationStep()
    
    state_before_action = get_state()

    while not done:
        action_index = agent.choose_action(state_before_action)
        chosen_green_phase = GREEN_PHASES[action_index]
        
        switched_phase = False
        if chosen_green_phase != current_green_phase:
            switched_phase = True
            yellow_phase_to_set = YELLOW_PHASES[current_green_phase]
            traci.trafficlight.setPhase(TLS_ID, yellow_phase_to_set)
            for _ in range(YELLOW_DURATION):
                if traci.simulation.getMinExpectedNumber() == 0: break
                traci.simulationStep()
        
        traci.trafficlight.setPhase(TLS_ID, chosen_green_phase)
        for _ in range(MIN_GREEN_DURATION):
            if traci.simulation.getMinExpectedNumber() == 0: break
            traci.simulationStep()
        
        state_after_action = get_state()
        done = traci.simulation.getMinExpectedNumber() == 0

        reward = calculate_reward(state_before_action, state_after_action)
        
        if switched_phase:
            reward -= 200

        agent.remember(state_before_action, action_index, reward, state_after_action, done)
        
        total_episode_reward += reward
        state_before_action = state_after_action
        current_green_phase = chosen_green_phase
        
    traci.close()
    return total_episode_reward

if __name__ == "__main__":
    UPDATE_TARGET_EVERY = 5
    num_episodes_total = 400
    batch_size = 32

    state_size = len(INCOMING_LANES)
    action_size = len(GREEN_PHASES)
    agent = DQNAgent(state_size, action_size)

    log_file_path = "training_log_final.csv"
    start_episode = 1
    
    if os.path.exists(log_file_path):
        print(f"Log file '{log_file_path}' found")
        try:
            log_data = pd.read_csv(log_file_path)
            if not log_data.empty:
                last_episode = log_data['Episod'].iloc[-1]
                start_episode = last_episode + 1
                print(f"Starting training from episode {start_episode}")

                last_saved_model_episode = (last_episode // 10) * 10
                if last_saved_model_episode > 0:
                    model_path = f"model_ep_{last_saved_model_episode}.weights.h5"
                    if os.path.exists(model_path):
                        print(f"Loading weights from model: {model_path}")
                        agent.load(model_path)
                        agent.epsilon = log_data['Epsilon'].iloc[-1]
        except Exception as e:
            print(f"Error reading log file: {e}. Over-writing it.")
            with open(log_file_path, "w") as log_file:
                 log_file.write("Episod,RecompensaTotala,Epsilon\n")
    else:
        print(f"Log file not found. Creating '{log_file_path}'.")
        with open(log_file_path, "w") as log_file:
            log_file.write("Episod,RecompensaTotala,Epsilon\n")

    for e in range(start_episode - 1, num_episodes_total):
        total_reward = run_episode(agent)
                
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{e+1},{total_reward},{agent.epsilon:.4f}\n")
        
        agent.replay(batch_size)
        
        if (e + 1) % UPDATE_TARGET_EVERY == 0:
            agent.update_target_model()
        
        if (e + 1) % 10 == 0:
            agent.save(f"model_ep_{e+1}.weights.h5")
            
    print("Training finished.")
