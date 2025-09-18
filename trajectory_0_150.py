import os
import gymnasium as gym
import torch
import numpy as np
import pickle
import gc
from stable_baselines3 import PPO
from tqdm import tqdm
import warnings
from collections import deque
import time

warnings.filterwarnings("ignore")

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_ID = "LunarLander-v3"

# GPU Optimization Settings
BATCH_SIZE = 512  # Larger batch size for GPU efficiency
PREFETCH_FACTOR = 4  # Number of batches to prefetch
GPU_MEMORY_FRACTION = 0.8  # Reserve some GPU memory
ASYNC_ENVS = True  # Use async environments if available


MODEL_PATHS = {
    "medium": "Trajectory_models/medium_policy_seed123.zip",
}

# --- CHANGE 2: Adjusted the return range for 'medium' trajectories ---
RETURN_RANGES = {
    "medium": (0.0, 150.0)  # Changed from (0.0, 249.0) to (0.0, 150.0)
}

# Dynamically set number of environments based on GPU memory
def get_optimal_num_envs():
    if DEVICE == "cuda":
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # Estimate based on GPU memory (rough heuristic)
            if gpu_memory > 16 * 1024**3:  # >16GB
                return 32
            elif gpu_memory > 8 * 1024**3:  # >8GB
                return 24
            else:
                return 16
        except:
            return 16
    else:
        return 8

NUM_ENVS = get_optimal_num_envs()

# --- GPU Memory Management ---
def setup_gpu_optimization():
    if DEVICE == "cuda":
        # Set memory fraction
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Set memory growth
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        
        print(f"🔧 GPU optimization enabled: {torch.cuda.get_device_name()}")
        print(f"   Memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

def cleanup_gpu_memory():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

# --- Helper Functions ---
def load_model(model_path: str, device: str):
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return None
    try:
        model = PPO.load(model_path, device=device)
        # Optimize model for inference
        if hasattr(model.policy, 'eval'):
            model.policy.eval()
        
        # Compile model for better GPU performance (PyTorch 2.0+)
        if hasattr(torch, 'compile') and device == "cuda":
            try:
                model.policy = torch.compile(model.policy, mode="reduce-overhead")
                print(f"✅ Model compiled for GPU optimization")
            except:
                print(f"⚠️ Model compilation failed, using standard mode")
        
        print(f"✅ Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        return None

# --- Optimized Trajectory Buffer ---
class GPUTrajectoryBuffer:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.trajectories = deque(maxlen=max_size)
        self.returns = deque(maxlen=max_size)
    
    def __len__(self):
        return len(self.trajectories)
    
    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)
        self.returns.append(trajectory['total_reward'])
    
    def get_trajectories_by_range(self, return_range, max_count=None):
        valid_trajectories = []
        for traj in self.trajectories:
            if return_range[0] <= traj['total_reward'] <= return_range[1]:
                valid_trajectories.append(traj)
                if max_count and len(valid_trajectories) >= max_count:
                    break
        return valid_trajectories
    
    def clear(self):
        self.trajectories.clear()
        self.returns.clear()

# --- Optimized Trajectory Generation Function ---
def generate_trajectory_data_gpu_optimized(
    model,
    env_id,
    num_trajectories,
    return_range,
    tag,
    num_envs=NUM_ENVS,
    base_seed=0,
    attempt_multiplier=20,  # Reduced since we're more efficient
    early_stop_thresh=10000
):
    MAX_STEPS_PER_ENV = 1000
    MAX_TOTAL_SIMULATION_STEPS = num_trajectories * MAX_STEPS_PER_ENV * attempt_multiplier
    
    # Use GPU-optimized buffer
    trajectory_buffer = GPUTrajectoryBuffer()
    total_steps_simulated = 0
    
    print(f"\n✨ Generating {num_trajectories} '{tag}' trajectories (return ∈ [{return_range[0]:.2f}, {return_range[1]:.2f}])")
    print(f"🔧 Using {num_envs} parallel environments with GPU optimization")
    
    # Create vectorized environment
    vec_env = gym.make_vec(env_id, num_envs=num_envs, vectorization_mode="sync")
    
    # Pre-allocate tensors for better GPU memory usage
    obs_tensor = None
    action_tensor = None
    
    current_episode_buffers = [
        {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': [], 'total_reward': 0.0, 'seed': 0}
        for _ in range(num_envs)
    ]
    
    initial_seeds = [base_seed + i for i in range(num_envs)]
    obs, info = vec_env.reset(seed=initial_seeds)
    
    # Convert to tensor for GPU operations
    if DEVICE == "cuda":
        obs_tensor = torch.FloatTensor(obs).to(DEVICE)
    
    for i in range(num_envs):
        current_episode_buffers[i]['seed'] = initial_seeds[i]
        current_episode_buffers[i]['states'].append(obs[i])
    
    no_success_streak = 0
    batch_count = 0
    
    with tqdm(total=num_trajectories, desc=f"[{tag}] Collecting", ncols=120, colour="green") as pbar:
        while len(trajectory_buffer) < num_trajectories and total_steps_simulated < MAX_TOTAL_SIMULATION_STEPS:
            
            # GPU-optimized prediction with batching
            with torch.no_grad():
                if DEVICE == "cuda":
                    # Use tensor operations for better GPU utilization
                    obs_tensor = torch.FloatTensor(obs).to(DEVICE, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        actions, _ = model.predict(obs_tensor.cpu().numpy(), deterministic=True)
                else:
                    actions, _ = model.predict(obs, deterministic=True)
            
            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
            dones = terminateds | truncateds
            
            total_steps_simulated += num_envs
            batch_count += 1
            
            # Process episodes in batch
            completed_episodes = []
            for i in range(num_envs):
                current_episode_buffers[i]['actions'].append(actions[i])
                current_episode_buffers[i]['rewards'].append(rewards[i])
                current_episode_buffers[i]['next_states'].append(next_obs[i])
                current_episode_buffers[i]['dones'].append(dones[i])
                current_episode_buffers[i]['total_reward'] += rewards[i]
                
                if dones[i]:
                    total_reward = current_episode_buffers[i]['total_reward']
                    episode_length = len(current_episode_buffers[i]['actions'])
                    
                    if episode_length > 0:
                        # --- CHANGE 1: Note that the seed is already being stored here ---
                        trajectory = {
                            'states': np.array(current_episode_buffers[i]['states'], dtype=np.float32),
                            'actions': np.array(current_episode_buffers[i]['actions'], dtype=np.int64),
                            'rewards': np.array(current_episode_buffers[i]['rewards'], dtype=np.float32),
                            'next_states': np.array(current_episode_buffers[i]['next_states'], dtype=np.float32),
                            'dones': np.array(current_episode_buffers[i]['dones'], dtype=bool),
                            'total_reward': total_reward,
                            'tag': tag,
                            'seed': current_episode_buffers[i]['seed'],
                            'length': episode_length
                        }
                        
                        if return_range[0] <= total_reward <= return_range[1]:
                            trajectory_buffer.add_trajectory(trajectory)
                            no_success_streak = 0
                            
                            # Update progress bar
                            if len(trajectory_buffer.returns) > 0:
                                avg_return = np.mean(list(trajectory_buffer.returns))
                                pbar.set_postfix({
                                    "Last": f"{total_reward:.1f}",
                                    "Avg": f"{avg_return:.1f}",
                                    "GPU": f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if DEVICE == "cuda" else "N/A"
                                })
                            pbar.update(1)
                        else:
                            no_success_streak += 1
                    
                    # Reset episode buffer
                    current_episode_buffers[i] = {
                        'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': [], 'total_reward': 0.0
                    }
                    new_seed_for_env_i = base_seed + total_steps_simulated + i
                    current_episode_buffers[i]['seed'] = new_seed_for_env_i
                    current_episode_buffers[i]['states'].append(next_obs[i])
                else:
                    current_episode_buffers[i]['states'].append(next_obs[i])
            
            # Periodic GPU memory cleanup
            if batch_count % 100 == 0 and DEVICE == "cuda":
                torch.cuda.empty_cache()
            
            if no_success_streak >= early_stop_thresh:
                print(f"⚠️ No valid trajectories in last {early_stop_thresh} attempts for {tag}. Stopping early.")
                break
            
            if len(trajectory_buffer) >= num_trajectories:
                break
            
            obs = next_obs
    
    vec_env.close()
    cleanup_gpu_memory()
    
    # Extract final trajectories
    final_trajectories = list(trajectory_buffer.trajectories)[:num_trajectories]
    
    if len(final_trajectories) < num_trajectories:
        print(f"⚠️ Only collected {len(final_trajectories)}/{num_trajectories} '{tag}' trajectories.")
    else:
        print(f"✅ Collected {len(final_trajectories)} '{tag}' trajectories.")
    
    if trajectory_buffer.returns:
        returns_list = list(trajectory_buffer.returns)
        print(f"📊 '{tag}' return stats: Mean = {np.mean(returns_list):.2f}, Min = {np.min(returns_list):.2f}, Max = {np.max(returns_list):.2f}")
    
    return final_trajectories

# --- User Input Function ---
def get_user_trajectory_counts():
    print("📥 Enter number of NEW trajectories to generate:")
    try:
        medium = int(input("Medium trajectories: "))
        bad = int(input("Bad trajectories: "))
    except ValueError:
        print("❌ Invalid input. Please enter integers.")
        exit(1)
    return {"medium": medium, "bad": bad}

# --- Main Execution ---
def main():
    print(f"🚀 Starting GPU-optimized trajectory generation (device: {DEVICE})")
    print("=" * 80)
    
    # Setup GPU optimization
    setup_gpu_optimization()
    
    # Compute final filename
    expert_target_count = 4000
    new_trajectory_counts = get_user_trajectory_counts()
    medium_count = new_trajectory_counts.get("medium", 0)
    bad_count = new_trajectory_counts.get("bad", 0)
    
    final_output_filename = os.path.join(
        "TRAJECTORIES",
        f"trajectories_35000.pkl"
    )
    
    # Test write to final output file
    try:
        os.makedirs("datasets_3", exist_ok=True)
        with open(final_output_filename, "wb") as f:
            pickle.dump([], f)
        print(f"🧪 Test write to '{final_output_filename}' succeeded.")
    except Exception as e:
        print(f"❌ Test write failed: {e}")
        return
    
    # Initialize trajectory collection
    all_trajectories = []
    expert_count_from_file = 0
    print(f"⚠️ Skipping expert trajectories as requested.")
    
    # Load models with GPU optimization
    models_to_load = {}
    if medium_count > 0: models_to_load["medium"] = MODEL_PATHS["medium"]
    if bad_count > 0: models_to_load["bad"] = MODEL_PATHS["bad"]
    
    loaded_models = {}
    for tag, path in models_to_load.items():
        model = load_model(path, DEVICE)
        if model:
            loaded_models[tag] = model
        else:
            print(f"⏭️ Skipping '{tag}' due to model load failure.")
    
    # Generate trajectories with GPU optimization
    start_time = time.time()
    
    for tag in ["medium", "bad"]:
        if tag in loaded_models:
            count = new_trajectory_counts[tag]
            print(f"\n--- Generating '{tag}' trajectories with GPU optimization ---")
            seed = hash(tag) % (2**32)
            
            trajectories = generate_trajectory_data_gpu_optimized(
                model=loaded_models[tag],
                env_id=ENV_ID,
                num_trajectories=count,
                return_range=RETURN_RANGES[tag],
                tag=tag,
                num_envs=NUM_ENVS,
                base_seed=seed
            )
            
            all_trajectories.extend(trajectories)
            if tag == "medium":
                medium_count = len(trajectories)
            elif tag == "bad":
                bad_count = len(trajectories)
            
            # Cleanup between models
            cleanup_gpu_memory()
    
    generation_time = time.time() - start_time
    print(f"\n⏱️ Total generation time: {generation_time:.2f} seconds")
    
    # Final save
    try:
        with open(final_output_filename, "wb") as f:
            pickle.dump(all_trajectories, f)
        print(f"\n🎉 Saved dataset to {final_output_filename} with:")
        print(f"   Expert: {expert_count_from_file} | Medium: {medium_count} | Bad: {bad_count}")
        print(f"   Total trajectories: {len(all_trajectories)}")
        print(f"   Generation rate: {len(all_trajectories)/generation_time:.2f} trajectories/second")
    except Exception as e:
        print(f"❌ Final save failed: {e}")
    
    # Final cleanup
    cleanup_gpu_memory()
    print("🏁 Done!")

if __name__ == "__main__":
    os.makedirs("datasets_3", exist_ok=True)
    os.makedirs("models_3", exist_ok=True)
    main()