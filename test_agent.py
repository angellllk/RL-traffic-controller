import os
import sys
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

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
        self.learning_rate = 0.0005
        self.model = self._build_model()

    def _build_model(self):
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

    def choose_action(self, state):
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
            
    def load(self, name):
        self.model.load_weights(name)

def get_state():
    state = [traci.lane.getWaitingTime(lane_id) for lane_id in INCOMING_LANES]
    return np.reshape(state, [1, len(INCOMING_LANES)])

if __name__ == "__main__":
    state_size = len(INCOMING_LANES)
    action_size = len(GREEN_PHASES)
    agent = DQNAgent(state_size, action_size)

    MODEL_CAMPION = "model_ep_140.weights.h5"
    agent.load(MODEL_CAMPION)
    print(f"Se testeaza modelul: {MODEL_CAMPION}")

    traci.start(sumoCmd)
    
    current_state = get_state()
    done = False
    current_green_phase = GREEN_PHASES[0]
    traci.trafficlight.setPhase(TLS_ID, current_green_phase)

    while not done:
        action_index = agent.choose_action(current_state)
        chosen_green_phase = GREEN_PHASES[action_index]
        
        if chosen_green_phase != current_green_phase:
            yellow_phase_to_set = YELLOW_PHASES[current_green_phase]
            traci.trafficlight.setPhase(TLS_ID, yellow_phase_to_set)
            for _ in range(YELLOW_DURATION):
                if traci.simulation.getMinExpectedNumber() == 0: break
                traci.simulationStep()
        
        traci.trafficlight.setPhase(TLS_ID, chosen_green_phase)
        for _ in range(MIN_GREEN_DURATION):
            if traci.simulation.getMinExpectedNumber() == 0: break
            traci.simulationStep()
        
        current_state = get_state()
        done = traci.simulation.getMinExpectedNumber() == 0
        current_green_phase = chosen_green_phase
        
    traci.close()
    print("Testing AI agent finished. 'tripinfo.xml' was generatead.")