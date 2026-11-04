import sys
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

project_path = './content'
sys.path.append(project_path)

from env.custom_hopper import *
from env.curriculum_hopper import *

MODELS_DIR = os.path.join(project_path, "trained_models")
LOG_DIR = os.path.join(project_path, "training_logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- CONFIGURAZIONE ---
TOTAL_TIMESTEPS = 2_000_000 
ENV_ID = 'CustomHopper-source-v0'
ENV_CL_ID = 'CurriculumHopper-source-v0'
NUM_CPU = 16 
LEARNING_RATE = 1e-4

# Importante: con 16 CPU, n_steps deve essere basso per avere update frequenti
# Buffer size = NUM_CPU * N_STEPS = 16 * 256 = 4096 (standard PPO)
PPO_KWARGS = {
    "learning_rate": LEARNING_RATE,
    "n_steps": 256, 
    "batch_size": 64,
    "verbose": 1,
    "seed": 42,
    "device": "cuda"
}

def train_ppo_baseline():
    print(f"\n--- 1. Training PPO Standard (Parallelized) ---")
    
    env = make_vec_env(ENV_ID, n_envs=NUM_CPU, seed=42, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env, os.path.join(LOG_DIR, "ppo_baseline"))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    model = PPO('MlpPolicy', env, **PPO_KWARGS)
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="ppo_baseline")
    model.save(os.path.join(MODELS_DIR, "ppo_baseline"))
    env.save(os.path.join(MODELS_DIR, "ppo_baseline_vecnormalize.pkl"))
    env.close()

def train_ppo_framestack(n_stack=10):
    print(f"\n--- 2. Training PPO + FrameStack (Parallelized, Stack={n_stack}) ---")
    
    env = make_vec_env(ENV_ID, n_envs=NUM_CPU, seed=42, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env, os.path.join(LOG_DIR, f"ppo_stack_{n_stack}"))
    # FrameStack prima o dopo Normalize? SB3 di solito consiglia: Env -> Monitor -> Normalize -> Stack
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    env = VecFrameStack(env, n_stack=n_stack) 
    
    model = PPO('MlpPolicy', env, **PPO_KWARGS)
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=f"ppo_stack_{n_stack}")
    model.save(os.path.join(MODELS_DIR, f"ppo_stack_{n_stack}"))
    # Salviamo il normalizzatore (nota: VecFrameStack wrappa VecNormalize, quindi accediamo a env.venv per salvare se necessario, 
    # ma env.save() su VecFrameStack dovrebbe gestire la gerarchia o dovresti salvare env.venv se è quello normalizzato)
    if hasattr(env, 'venv') and isinstance(env.venv, VecNormalize):
        env.venv.save(os.path.join(MODELS_DIR, f"ppo_stack_{n_stack}_vecnormalize.pkl"))
    else:
        # Fallback se la struttura è diversa, prova a salvare l'oggetto top-level
        print("Warning: Could not find VecNormalize wrapper directly, saving top-level env.")
        env.save(os.path.join(MODELS_DIR, f"ppo_stack_{n_stack}_vecnormalize.pkl"))
    env.close()

def train_recurrent_ppo():
    print(f"\n--- 3. Training Recurrent PPO (Parallelized LSTM) ---")
    
    env = make_vec_env(ENV_ID, n_envs=NUM_CPU, seed=42, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env, os.path.join(LOG_DIR, "recurrent_ppo"))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Parametri specifici per LSTM
    policy_kwargs = {
        "enable_critic_lstm": False, 
        "lstm_hidden_size": 256,
        "net_arch": dict(pi=[64, 64], vf=[64, 64]) # Un po' di feature extraction prima della LSTM aiuta
    }
    
    model = RecurrentPPO(
        'MlpLstmPolicy', 
        env, 
        policy_kwargs=policy_kwargs,
        **PPO_KWARGS
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="recurrent_ppo")
    model.save(os.path.join(MODELS_DIR, "recurrent_ppo"))
    env.save(os.path.join(MODELS_DIR, "recurrent_ppo_vecnormalize.pkl"))
    env.close()

def train_rppo_linear_cl():
    print(f"\n--- 5. Training Recurrent PPO (Parallelized LSTM) using Curriculum Learning (linear) ---")
    
    CL_KWARGS = {
        "total_timesteps": TOTAL_TIMESTEPS // NUM_CPU,
        "domain": "source"
    }
    
    env = make_vec_env(ENV_CL_ID, n_envs=NUM_CPU, seed=42, 
                       vec_env_cls=SubprocVecEnv, 
                       env_kwargs=CL_KWARGS,)
    
    env = VecMonitor(env, os.path.join(LOG_DIR, "rppo_curriculum_linear"))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    policy_kwargs = {
        "enable_critic_lstm": False, 
        "lstm_hidden_size": 256,
        "net_arch": dict(pi=[64, 64], vf=[64, 64])
    }
    
    model = RecurrentPPO(
        'MlpLstmPolicy', 
        env, 
        policy_kwargs=policy_kwargs,
        **PPO_KWARGS
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="rppo_curriculum_linear")
    model.save(os.path.join(MODELS_DIR, "rppo_curriculum_linear"))
    env.save(os.path.join(MODELS_DIR, "rppo_curriculum_linear_vecnormalize.pkl"))
    env.close()

def train_rppo_quadratic_cl():
    print(f"\n--- 6. Training Recurrent PPO (Parallelized LSTM) using Curriculum Learning (quadratic) ---")

    CL_KWARGS = {
        "total_timesteps": TOTAL_TIMESTEPS // NUM_CPU,
        "domain": "source",
        "quadratic": True
    }
    
    env = make_vec_env(ENV_CL_ID, n_envs=NUM_CPU, seed=42, 
                       vec_env_cls=SubprocVecEnv, 
                       env_kwargs=CL_KWARGS,)
    
    env = VecMonitor(env, os.path.join(LOG_DIR, "rppo_curriculum_quadratic"))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    policy_kwargs = {
        "enable_critic_lstm": False, 
        "lstm_hidden_size": 256,
        "net_arch": dict(pi=[64, 64], vf=[64, 64])
    }
    
    model = RecurrentPPO(
        'MlpLstmPolicy', 
        env, 
        policy_kwargs=policy_kwargs,
        **PPO_KWARGS
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="rppo_curriculum_quadratic")
    model.save(os.path.join(MODELS_DIR, "rppo_curriculum_quadratic"))
    env.save(os.path.join(MODELS_DIR, "rppo_curriculum_quadratic_vecnormalize.pkl"))
    env.close()

def train_rppo_quadratic_cl_wider():
    print(f"\n--- 7. Training Recurrent PPO (Parallelized LSTM) using Curriculum Learning (quadratic), wider range ---")

    CL_KWARGS = {
        "total_timesteps": TOTAL_TIMESTEPS // NUM_CPU,
        "domain": "source",
        "quadratic": True,
        "max_range": 0.6
    }
    
    env = make_vec_env(ENV_CL_ID, n_envs=NUM_CPU, seed=42, 
                       vec_env_cls=SubprocVecEnv, 
                       env_kwargs=CL_KWARGS,)
    
    env = VecMonitor(env, os.path.join(LOG_DIR, "rppo_curriculum_quadratic_wider"))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    policy_kwargs = {
        "enable_critic_lstm": False, 
        "lstm_hidden_size": 256,
        "net_arch": dict(pi=[64, 64], vf=[64, 64])
    }
    
    model = RecurrentPPO(
        'MlpLstmPolicy', 
        env, 
        policy_kwargs=policy_kwargs,
        **PPO_KWARGS
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="rppo_curriculum_quadratic_wider")
    model.save(os.path.join(MODELS_DIR, "rppo_curriculum_quadratic_wider"))
    env.save(os.path.join(MODELS_DIR, "rppo_curriculum_quadratic_wider_vecnormalize.pkl"))
    env.close()

if __name__ == "__main__":
    try:
        #train_ppo_baseline()
        #train_ppo_framestack(n_stack=10)
        #train_recurrent_ppo()
        #train_rppo_linear_cl()
        #train_rppo_quadratic_cl()
        #train_rppo_quadratic_cl_wider()
        print("Parrallelized training completed successfully.")
    except Exception as e:
        print(f"Error: {e}")