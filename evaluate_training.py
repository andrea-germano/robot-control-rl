import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURAZIONE ---
project_path = './udr_25_all_cl_40'
LOG_DIR = os.path.join(project_path, "training_logs")
TRAIN_RES_DIR = os.path.join(project_path, "training_results")

os.makedirs(TRAIN_RES_DIR, exist_ok=True)

# Formato: (nome_file_log_senza_estensione, etichetta_grafico, colore)
experiments = [
    ("ppo_baseline", "PPO Standard (No Memory)", "blue"),
    ("ppo_stack_10", "PPO FrameStack (10 frames)", "orange"),
    ("recurrent_ppo", "Recurrent PPO (LSTM)", "green"),
    ("rppo_curriculum", "Curriculum Learning RPPO", "red"),
]

def moving_average(values, window):
    """Calcola la media mobile per smussare le curve"""
    if len(values) < window:
        return values
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_training_results():
    print(f"Reading logs from: {LOG_DIR}")
    
    data = {}
    
    # 1. Caricamento Dati
    for filename, label, color in experiments:
        # VecMonitor salva solitamente come "nome.monitor.csv"
        file_path = os.path.join(LOG_DIR, f"{filename}.monitor.csv")
        
        if os.path.exists(file_path):
            try:
                # Skiprows=1 perché la prima riga è metadata di SB3
                df = pd.read_csv(file_path, skiprows=1)
                print(f"Loaded {label}: {len(df)} episodes found.")
                
                # Calcoliamo i timesteps cumulativi
                if 'l' in df.columns:
                    data[label] = {
                        'x': df['l'].cumsum(),   # Asse X: Timesteps totali
                        'reward': df['r'],       # Asse Y1: Reward
                        'length': df['l'],       # Asse Y2: Lunghezza episodio
                        'color': color
                    }
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"MISSING FILE: {file_path}")
            print(f"Make sure training is finished and filenames match.")

    if not data:
        print("No data found. Cannot generate plots.")
        return

    plt.figure(figsize=(12, 6))
    for label, d in data.items():
        # Smoothing window: più alto è, più "liscia" è la linea
        window = 100  
        if len(d['reward']) > window:
            y_smooth = moving_average(d['reward'], window)
            x_smooth = d['x'][len(d['x']) - len(y_smooth):] # Taglia X per allineare
            plt.plot(x_smooth, y_smooth, linewidth=2, label=label, color=d['color'])
        else:
            print(f"Not enough data points for smoothing {label}, plotting raw data.")
            plt.plot(d['x'], d['reward'], alpha=0.3, label=f"{label} (raw)", color=d['color'])

    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.title("Performance Comparison (Average Reward)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(TRAIN_RES_DIR, "reward_comparison.png"), dpi=150)
    plt.show()

    # 3. Generazione Grafico LUNGHEZZA EPISODIO
    plt.figure(figsize=(12, 6))
    for label, d in data.items():
        window = 100
        if len(d['length']) > window:
            y_smooth = moving_average(d['length'], window)
            x_smooth = d['x'][len(d['x']) - len(y_smooth):]
            plt.plot(x_smooth, y_smooth, linewidth=2, label=label, color=d['color'])
        else:
            print(f"Not enough data points for smoothing {label}, plotting raw data.")
            plt.plot(d['x'], d['length'], alpha=0.3, label=f"{label} (raw)", color=d['color'])

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Length (Steps)")
    plt.title("Stability Comparison (Episode Length)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(TRAIN_RES_DIR, "length_comparison.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_training_results()