import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

def parse_tripinfo(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found!")
        return []
    tree = ET.parse(file_path)
    root = tree.getroot()
    waiting_times = [float(trip.get('waitingTime')) for trip in root.findall('tripinfo')]
    return waiting_times

def plot_comparative_learning_curve():  
    log_files_to_compare = {
        "Attempt 1: Basic DQN (Reward = Delta)": "training_log_v1.csv",
        "Attempt 2: Quadratic Reward": "training_log_v2.csv",
        "Final Model: Dueling Double DQN": "training_log_final.csv"
    }

    plt.figure(figsize=(14, 8))
    
    for label, file_name in log_files_to_compare.items():
        if not os.path.exists(file_name):
            print(f"WARNING: File '{file_name}' not found! Skipping.")
            continue
        
        try:
            log_data = pd.read_csv(file_name, sep=',')
            rolling_avg = log_data['RecompensaTotala'].rolling(window=20, min_periods=1).mean()
            plt.plot(log_data['Episod'], rolling_avg, label=label, linewidth=2.5)
        except Exception as e:
            print(f"ERROR processing'{file_name}': {e}")

    plt.title('Comparative Evolution of Training Performance', fontsize=18)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Accumulated Reward (20-Episode Moving Average)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    plt.savefig('comparative_training_curve.png')


def plot_final_performance_charts():  
    wait_times_fix = parse_tripinfo('tripinfo_FIX.xml')
    wait_times_heuristic = parse_tripinfo('tripinfo_HEURISTIC.xml')
    wait_times_ai = parse_tripinfo('tripinfo_AI.xml')

    if not all([wait_times_fix, wait_times_heuristic, wait_times_ai]):
        print("One or more tripinfo files are missing.")
        return

    avg_wait_fix = np.mean(wait_times_fix)
    avg_wait_heuristic = np.mean(wait_times_heuristic)
    avg_wait_ai = np.mean(wait_times_ai)
    throughput_fix = len(wait_times_fix)
    throughput_heuristic = len(wait_times_heuristic)
    throughput_ai = len(wait_times_ai)

    print("\n" + "="*30)
    print("--- FINAL RESULTS ---")
    print("="*30)
    print(f"Fixed-Time Controller: Avg Waiting Time = {avg_wait_fix:.2f}s | Throughput = {throughput_fix} vehicles")
    print(f"Heuristic Controller:  Avg Waiting Time = {avg_wait_heuristic:.2f}s | Throughput = {throughput_heuristic} vehicles")
    print(f"AI Agent (DQN):        Avg Waiting Time = {avg_wait_ai:.2f}s | Throughput = {throughput_ai} vehicles")
    
    wait_improvement_ai = ((avg_wait_fix - avg_wait_ai) / avg_wait_fix) * 100
    print(f"\nAI Agent Improvement vs Fixed-Time: {wait_improvement_ai:.2f}% reduction in waiting time.")
    print("="*30)

    plt.figure(figsize=(10, 7))
    labels = ['Fixed-Time Controller', 'Heuristic Controller', 'AI Agent (DQN)']
    values = [avg_wait_fix, avg_wait_heuristic, avg_wait_ai]
    colors = ['#d9534f', '#f0ad4e', '#5cb85c']
    bars = plt.bar(labels, values, color=colors)
    plt.bar_label(bars, fmt='%.2f s')
    plt.title('Average Waiting Time Comparison', fontsize=16)
    plt.ylabel('Waiting Time (seconds)', fontsize=12)
    plt.tight_layout()
    plt.savefig('waiting_time_comparison.png')

    plt.figure(figsize=(10, 7))
    values_throughput = [throughput_fix, throughput_heuristic, throughput_ai]
    bars = plt.bar(labels, values_throughput, color=colors)
    plt.bar_label(bars)
    plt.title('Total Throughput Comparison', fontsize=16)
    plt.ylabel('Number of Processed Vehicles', fontsize=12)
    plt.tight_layout()
    plt.savefig('throughput_comparison.png')

if __name__ == "__main__":
    plot_comparative_learning_curve()
    plot_final_performance_charts()

