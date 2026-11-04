import sys
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from sb3_contrib import RecurrentPPO

# --- CONFIGURAZIONE ---
project_path = './udr_25_all_cl_40'
sys.path.append(project_path)

from env.custom_hopper import *

# List of models to test: (Label, Filename Prefix, Is Recurrent, N Stack)
MODELS_TO_TEST = [
    ("PPO Standard", "ppo_baseline", False, 0),
    ("PPO FrameStack", "ppo_stack_10", False, 10),
    ("Recurrent PPO", "recurrent_ppo", True, 0),
    ("CL RPPO\nlinear", "rppo_curriculum_linear", True, 0),
    ("CL RPPO\nquadratic 45%", "rppo_curriculum_quadratic", True, 0),
    ("CL RPPO\nquadratic 60%", "rppo_curriculum_quadratic_wider", True, 0)
]

# Colors for plotting, consistent with evaluate training plots
MODEL_COLORS = {
    "PPO Standard": "blue",
    "PPO FrameStack": "orange",
    "Recurrent PPO": "green",
    "CL RPPO\nlinear": "red",
    "CL RPPO\nquadratic 45%": "black",
    "CL RPPO\nquadratic 60%": "violet"
}

# Cartelle
MODELS_DIR = os.path.join(project_path, "trained_models")
RESULTS_DIR = os.path.join(project_path, "test_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_EPISODES = 20  
# Range ottimizzato , testiamo da 0.6 (molto leggero) a 1.4 (molto pesante)
MASS_MULTIPLIERS = np.linspace(0.6, 1.4, 20)
ENV_ID = 'CustomHopper-target-v0' 

def make_eval_env(env_id, model_path_prefix, n_stack=0) -> gym.Env:
    """Crea l'ambiente di test replicando ESATTAMENTE la pipeline di training."""
    raw_env = gym.make(env_id, render_mode=None)
    env = DummyVecEnv([lambda: raw_env])
    norm_path = f"{model_path_prefix}_vecnormalize.pkl"
    if os.path.exists(norm_path):
        print(f"  -> Loading Normalization stats from: {os.path.basename(norm_path)}")
        env = VecNormalize.load(norm_path, env)
        # Nel test non vogliamo aggiornare le statistiche (training=False) e vogliamo vedere i reward VERI (norm_reward=False)
        env.training = False
        env.norm_reward = False
    else:
        print(f"  -> WARNING: Normalization file NOT found at {norm_path}. Model might fail.")

    # 3. FrameStack (Se usato)
    if n_stack > 0:
        print(f"  -> Applying FrameStack: {n_stack}")
        env = VecFrameStack(env, n_stack=n_stack)
        
    return env

def evaluate_model_on_mass(model, env, mass_multiplier, n_episodes, is_recurrent=False):
    """
    Valuta un modello forzando una specifica massa.
    Restituisce liste di float puri (non numpy array).
    """
    rewards = []
    lengths = []

    # Accediamo all'ambiente originale per modificare le masse
    current_env = env
    while hasattr(current_env, 'venv') or hasattr(current_env, 'env'):
        if hasattr(current_env, 'venv'):
            current_env = current_env.venv
        elif hasattr(current_env, 'env'):
            current_env = current_env.env

    if hasattr(current_env, 'envs'):
        real_env = current_env.envs[0].unwrapped
    else:
        real_env = current_env.unwrapped
    
    if not hasattr(real_env, 'original_masses'):
        print("Warning: Could not access 'original_masses'. Ensure CustomHopper is wrapped correctly.")
        return [], []
    
    original_masses = real_env.original_masses
    target_masses = np.array(original_masses, dtype=np.float64, copy=True)
    # Moltiplica solo la massa del torso (primo link in CustomHopper)
    target_masses[0] = target_masses[0] * mass_multiplier
    
    for _ in range(n_episodes):
        obs = env.reset()
        real_env.set_parameters(target_masses)

        done = False
        lstm_states = None
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            
            episode_reward += float(reward.item()) if isinstance(reward, np.ndarray) else float(reward)
            done = done[0] if isinstance(done, np.ndarray) else done
            episode_length += 1
                
        rewards.append(episode_reward)
        lengths.append(episode_length)
        
    return rewards, lengths

def run_sensitivity_analysis():
    print("--- START SENSITIVITY ANALYSIS (Mass vs Performance) ---")
    
    results = [] 
    
    
    for label, filename, is_recurrent, n_stack in MODELS_TO_TEST:
        model_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
            print(f"SKIPPED: Model {filename} not found in {MODELS_DIR}")
            continue
            
        print(f"Testing Model: {label}...")
        
        env = make_eval_env(ENV_ID, os.path.join(MODELS_DIR, filename), n_stack)
        
        if is_recurrent:
            model = RecurrentPPO.load(model_path, env=env)
        else:
            model = PPO.load(model_path, env=env)
            
        # Loop Masse
        for mult in MASS_MULTIPLIERS:
            ep_rewards, ep_lengths = evaluate_model_on_mass(
                model, env, mult, N_EPISODES, is_recurrent
            )
            
            # Salvataggio dati (assicurandoci siano tipi base)
            for r, l in zip(ep_rewards, ep_lengths):
                results.append({
                    "Model": label,
                    "Mass Multiplier": float(mult),
                    "Reward": float(r),
                    "Episode Length": int(l)
                })
        
        env.close()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_data.csv"), index=False)
    if not df.empty:
        df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_data.csv"), index=False)
        print("Data saved.")
    return df

def test_sim2real_transfer():
    print("\n--- FINAL TEST: SIM2REAL TRANSFER (Target Env) ---")
    
    results = []
    
    for label, filename, is_recurrent, n_stack in MODELS_TO_TEST:
        model_path = os.path.join(MODELS_DIR, filename)
        print("model_path", model_path)
        
        if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
            continue

        print(f"Sim2Real Testing: {label}")
        
        env = make_eval_env(ENV_ID, model_path, n_stack)
        
        if is_recurrent:
            model = RecurrentPPO.load(model_path, env=env)
        else:
            model = PPO.load(model_path, env=env)
        
        rewards = []
        for _ in range(50): # 50 episodi di test
            obs = env.reset()
            done = False
            lstm_states = None
            ep_reward = 0.0
            while not done:
                if is_recurrent:
                    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, done, _ = env.step(action)
                
                r = reward.item() if isinstance(reward, np.ndarray) else float(reward)
                d = done[0] if isinstance(done, np.ndarray) else done
                ep_reward += r
                done = d
            
            rewards.append(ep_reward)
            
        results.append({
            "Model": label,
            "Mean": np.mean(rewards),
            "Std": np.std(rewards),
            "Rewards": rewards
        })
        env.close()
    
    plot_sim2real_results(results)
    return results

def plot_graphs(df):
    if df.empty:
        print("No data to plot.")
        return

    sns.set_theme(style="whitegrid")
    
    # 1. Sensitivity
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Mass Torso Multiplier", y="Reward", hue="Model", marker="o",palette=MODEL_COLORS)
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label="Nominal Mass")
    plt.title("Robustness: Mass Variation vs Reward")
    plt.savefig(os.path.join(RESULTS_DIR, "plot_sensitivity_reward.png"), dpi=300)
    plt.show()

    # 2. BOXPLOT ESTREMI
    # extreme_df = df[ (df["Mass Multiplier"] <= 0.701) | (df["Mass Multiplier"] >= 1.299) ].copy()
    # if not extreme_df.empty:
    #     extreme_df["Condition"] = np.where(extreme_df["Mass Multiplier"] <= 0.701, "Light (<=0.7x)", "Heavy (>=1.3x)")
    #     plt.figure(figsize=(10, 6))
    #     sns.boxplot(data=extreme_df, x="Condition", y="Reward", hue="Model", palette="Set2")
    #     plt.title("Performance in Extreme Conditions (OOD)", fontsize=14)
    #     plt.savefig(os.path.join(RESULTS_DIR, "plot_extreme_cases_boxplot.png"), dpi=300)
    #     plt.show()

    # 3. SOPRAVVIVENZA
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Mass Multiplier", y="Episode Length", hue="Model", linestyle="--", palette=MODEL_COLORS)
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.title("Survival Analysis", fontsize=14)
    plt.ylabel("Avg Episode Length (Steps)")
    plt.savefig(os.path.join(RESULTS_DIR, "plot_survival.png"), dpi=300)
    plt.show()


def plot_sim2real_results(results):
    """Plot sim2real transfer results as bar chart with error bars."""
    if not results:
        print("No models tested. Cannot generate plots.")
        return
    
    models = [r["Model"] for r in results]
    means = [r["Mean"] for r in results]
    stds = [r["Std"] for r in results]

    bar_colors = [MODEL_COLORS.get(model, 'gray') for model in models]
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(models))
    ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, 
                   color=bar_colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.set_title("Sim2Real Transfer: Target Environment Performance", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean+5, f"{mean:.2f}", ha='left', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sim2real_transfer_comparison.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    df_results = run_sensitivity_analysis()
    plot_graphs(df_results)
    test_sim2real_transfer()
    print(f"Plots saved in: {RESULTS_DIR}")