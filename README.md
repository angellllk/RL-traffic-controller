# Traffic Light Control Using Deep Reinforcement Learning
This repository contains the source code and resources developed for my dissertation titled **"Smart Traffic Signal Management in Bucharest using Reinforcement Learning"**. The project explores the use of an agent based on a Dueling Double Deep Q-Network (DDDQN) to control a traffic light cluster in a complex intersection in real-time, simulated using SUMO (Simulation of Urban MObility).

The main objective is to demonstrate that an intelligent agent can significantly reduce vehicle waiting times and increase throughput compared to traditional fixed-time controllers and heuristic (congestion-based) methods.

# Results
The comparative analysis of the three approaches (Fixed-Time, Heuristic, AI Agent) demonstrated the superiority of the Reinforcement Learning agent.

**Comparative Learning Curve**. This plot shows the evolution of the average reward during training for different versions of the agent, illustrating the convergence towards an effective policy.

**Average Waiting Time Comparison**. The AI agent achieved a significant reduction in the average waiting time per vehicle compared to both baseline control methods.

**Throughput Comparison**. By managing green phases more efficiently, the AI agent allowed a larger number of vehicles to traverse the intersection within the same simulated time period.
