import os
import sys
import traci
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Te rog declară variabila de mediu 'SUMO_HOME'")

GUI_MODE = True
SUMOCFG_FILE = "sim.sumocfg"
sumoBinary = "sumo-gui" if GUI_MODE else "sumo"
sumoCmd = [sumoBinary, "-c", SUMOCFG_FILE, "--start"]

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
    "302795167#0_0",
    "302795167#0_1",
    "302795167#0_2",
    "242364261_0",
    "242364261_1",
    "678011317_0",
    "678011317_1",
    "26195549#0_0",
    "26195549#0_1",
]


LANE_GROUPS = {
    0: [0, 1, 5, 6],    
    1: [2, 6],          
    2: [3, 4],         
    3: [7, 8]           
}

def get_state():
    return [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in INCOMING_LANES]

def choose_action_heuristic(state_vector):
    max_congestion = -1
    best_action = -1

    for action_index, lane_indices in LANE_GROUPS.items():
        current_congestion = 0

        for lane_idx in lane_indices:
            current_congestion += state_vector[lane_idx]
        
        if current_congestion > max_congestion:
            max_congestion = current_congestion
            best_action = action_index
            
    return best_action

if __name__ == "__main__":
    traci.start(sumoCmd)
    current_state = get_state()
    current_green_phase_index = 0
    traci.trafficlight.setPhase(TLS_ID, GREEN_PHASES[current_green_phase_index])

    while traci.simulation.getMinExpectedNumber() > 0:
        action_index = choose_action_heuristic(current_state)
        chosen_green_phase = GREEN_PHASES[action_index]
        
        current_phase_in_sim = traci.trafficlight.getPhase(TLS_ID)
        current_green_phase = GREEN_PHASES[current_green_phase_index]

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
        current_green_phase_index = action_index

    traci.close()
    print("Testing heuristic agent finished. 'tripinfo.xml' was generatead.")