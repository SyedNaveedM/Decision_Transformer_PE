import math
import gymnasium
import random
import numpy as np
from gymnasium.utils.step_api_compatibility import step_api_compatibility
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import pickle
import time
import shutil
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt

# Use maximum CPU cores for parallel environments
NUM_ENVS = min(32, mp.cpu_count())

def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0

    Returns:
        a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.unwrapped.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a

def update_random_factor(old_rf, alpha, current_reward, expected_reward):
    error = current_reward - expected_reward
    
    # Sharper non-linear transformation using tanh
    sharp_error = math.tanh(5 * error / (abs(expected_reward) + 1e-8))
    
    # Update with alpha
    delta = alpha * sharp_error
    
    new_rf = old_rf + delta
    return max(0.0, min(1.0, new_rf))

def demo_heuristic_lander(env, target = 50, alpha = 1, random_factor = 0.05, seed=None):
    random_factor = min(1.0, max(0.0, random_factor))
    total_reward = 0
    steps = 0
    
    # Trajectory data collection
    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = []
    dones_list = []

    s, info = env.reset(seed=seed)
    states_list.append(s) # Store initial state

    while True:
        a = heuristic(env, s)
        if random.random() < random_factor:
            # Ensure random action is valid for discrete space (0, 1, 2, 3)
            a = random.randint(0, env.action_space.n - 1)
        
        next_s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
        states_list.append(next_s)  # Store next state
        actions_list.append(a)
        rewards_list.append(r)
        next_states_list.append(next_s.copy())
        dones_list.append(terminated or truncated)
        
        total_reward += r
        s = next_s

        steps += 1
        
        if terminated or truncated:
            break
        
        random_factor = update_random_factor(random_factor, alpha, current_reward=total_reward, expected_reward=target)
        
    # Construct trajectory dictionary
    trajectory = {
        'seed': seed,
        'states': np.array(states_list[:-1]), # Exclude the last state as it's a next_state
        'actions': np.array(actions_list),
        'rewards': np.array(rewards_list),
        'next_states': np.array(next_states_list),
        'dones': np.array(dones_list),
        'total_reward': total_reward
    }
    
    return trajectory

def bucketize(numbers, ranges):
    """
    Buckets numbers into custom ranges provided.
    ranges: list of {'category': int, 'range': (min, max)} dictionaries
    """
    buckets = {range_info['category']: [] for range_info in ranges}
    
    for num in numbers:
        found = False
        for range_info in ranges:
            category = range_info['category']
            min_r, max_r = range_info['range']
            if min_r <= num < max_r:
                buckets[category].append(num)
                found = True
                break
        # Handle numbers outside defined ranges if necessary, or just ignore
        # if not found:
        #    print(f"Warning: Reward {num} fell outside defined ranges.")
            
    return buckets

class MultiProgressDisplay:
    def __init__(self, target_ranges_info, num_trajectories_per_range):
        self.target_ranges_info = target_ranges_info
        self.num_trajectories_per_range = num_trajectories_per_range
        self.collected_by_range = {range_info['category']: 0 for range_info in target_ranges_info}
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        self.terminal_width = shutil.get_terminal_size().columns
        
        self.total_target = len(target_ranges_info) * num_trajectories_per_range
        
        desc_width = min(30, max(20, self.terminal_width // 5))
        bar_width = max(25, self.terminal_width - desc_width - 50)
        
        self.overall_pbar = tqdm(
            total=self.total_target,
            desc=f"🚀 Collecting FlatLander Trajectories",
            unit="traj",
            position=0,
            leave=True,
            bar_format=f'{{desc:<{desc_width}}} {{percentage:3.0f}}%|{{bar:{bar_width}}}| {{n}}/{{total}} [{{elapsed}}<{{remaining}}, {{rate_fmt}}]',
            colour='green'
        )
        
        self.range_pbars = {}
        for i, range_info in enumerate(target_ranges_info):
            category = range_info['category']
            range_name = get_category_label(range_info['range'])
            range_desc = f"  📊 {range_name}"
            self.range_pbars[category] = tqdm(
                total=num_trajectories_per_range,
                desc=range_desc,
                unit="traj",
                position=i + 1,
                leave=True,
                bar_format=f'{{desc:<{desc_width}}} {{percentage:3.0f}}%|{{bar:{bar_width}}}| {{n}}/{{total}} [{{rate_fmt}}]',
                colour='blue'
            )
    
    def update_progress(self, collected_by_range, trajectory_added=False):
        current_time = time.time()
        
        if trajectory_added or (current_time - self.last_update_time >= 0.5):
            total_collected = sum(collected_by_range.values())
            
            progress_made = total_collected - self.overall_pbar.n
            if progress_made > 0:
                self.overall_pbar.update(progress_made)
            
            for range_info in self.target_ranges_info:
                category = range_info['category']
                current_count = collected_by_range.get(category, 0)
                range_progress = current_count - self.range_pbars[category].n
                if range_progress > 0:
                    self.range_pbars[category].update(range_progress)
                
                range_name = get_category_label(range_info['range'])
                if current_count >= self.num_trajectories_per_range:
                    status_icon = "✅"
                elif current_count > 0:
                    status_icon = "🔄"
                else:
                    status_icon = "⏳"
                
                range_desc = f"  {status_icon} {range_name}"
                self.range_pbars[category].set_description(range_desc)
            
            active_ranges = sum(1 for cat in self.collected_by_range 
                                 if collected_by_range.get(cat, 0) < self.num_trajectories_per_range)
            completed_ranges = len(self.collected_by_range) - active_ranges
            
            self.overall_pbar.set_postfix_str(
                f'✅{completed_ranges}/{len(self.collected_by_range)} | Active:{active_ranges}',
                refresh=True
            )
            
            self.last_update_time = current_time
            self.collected_by_range = collected_by_range.copy()
    
    def finalize(self, collected_by_range):
        total_collected = sum(collected_by_range.values())
        
        remaining_overall = self.total_target - self.overall_pbar.n
        if remaining_overall > 0:
            self.overall_pbar.update(remaining_overall)
        
        for range_info in self.target_ranges_info:
            category = range_info['category']
            current_count = collected_by_range.get(category, 0)
            remaining_range = min(self.num_trajectories_per_range, current_count) - self.range_pbars[category].n
            if remaining_range > 0:
                self.range_pbars[category].update(remaining_range)
            
            range_name = get_category_label(range_info['range'])
            if current_count >= self.num_trajectories_per_range:
                range_desc = f"  ✅ {range_name}"
            else:
                range_desc = f"  ⚠️  {range_name}"
            self.range_pbars[category].set_description(range_desc)
    
    def close(self):
        self.overall_pbar.close()
        for pbar in self.range_pbars.values():
            pbar.close()

def get_category_label(reward_range):
    min_r, max_r = reward_range
    if max_r == np.inf:
        return f"[{int(min_r)}+]"
    return f"[{int(min_r)},{int(max_r-1)}]" # Adjust for exclusive upper bound

def run_episode_for_pool(seed, alpha, initial_random_factor, current_target_reward):
    """Function to be run by each process in the pool."""
    env = gymnasium.make("LunarLander-v3", continuous=False, render_mode=None)
    
    # We pass the target to the demo_heuristic_lander for its internal random_factor adjustment
    trajectory = demo_heuristic_lander(env=env, alpha=alpha, target=current_target_reward,
                                      seed=seed, random_factor=initial_random_factor)
    env.close()
    
    # Return trajectory along with its total reward and the target it was aiming for
    return trajectory

def main():
    min_ret = int(input("Minimum reward for bucketing: "))
    max_ret = int(input("Maximum reward for bucketing: "))
    step = int(input("Step size for bucketing: "))

    # USE 0.5 FOR GOOD PRACTICE
    alpha = float(input("Alpha (0.0 - 1.0) for random factor adjustment: "))

    # 1 IS GOOD FOR THIS
    initial_random_factor = float(input("Initial random factor (0.0 - 1.0): "))
    num_trajectories_per_category = int(input("Min number of UNIQUE trajectories per category: "))

    # UPPER LIMIT OF TRAJECTORIES PER CATEGORY
    upper_limit_count = int(input("Max number of UNIQUE trajectories per category: "))

    print(f"\n🚀 Using {mp.cpu_count()} CPU cores for parallel collection")
    print(f"📊 Bucketing rewards from {min_ret} to {max_ret} with step {step}")
    print(f"🎯 Aiming for {num_trajectories_per_category} UNIQUE trajectories per category")

    # Define the target reward ranges for bucketing
    target_ranges_info = []
    current_category_idx = 0
    for i in range(min_ret, max_ret + 1, step):
        target_ranges_info.append({'category': current_category_idx, 'range': (i, i + step)})
        current_category_idx += 1
    
    # Ensure the last bucket covers up to max_ret
    if target_ranges_info and target_ranges_info[-1]['range'][1] < max_ret:
        target_ranges_info[-1]['range'] = (target_ranges_info[-1]['range'][0], max_ret + 1) # Make sure it includes max_ret

    # Initialize data structures
    all_collected_trajectories = {range_info['category']: [] for range_info in target_ranges_info}
    unique_rewards_per_bucket = {range_info['category']: set() for range_info in target_ranges_info}
    
    progress_display = MultiProgressDisplay(target_ranges_info, num_trajectories_per_category)

    iteration = 0
    # The batch size per target. Adjust as needed.
    # It's good to have it larger than NUM_ENVS if you want to submit many tasks at once
    # However, `NUM_ENVS` is used directly as the number of parallel processes for `ProcessPoolExecutor`
    # Let's adjust this to be compatible with `ProcessPoolExecutor` logic.
    # We will submit a fixed number of tasks per iteration, distributed across targets.
    batch_size_per_target = 25 # Number of episodes to run for each target reward in one iteration

    try:
        with ProcessPoolExecutor(max_workers=NUM_ENVS) as executor:
            while True:
                iteration += 1
                # print(f"\n--- Iteration {iteration} ---")
                
                futures = []
                for range_info in target_ranges_info:
                    category = range_info['category']
                    min_r, max_r = range_info['range']
                    
                    # Only collect if this bucket still needs unique trajectories
                    if len(unique_rewards_per_bucket[category]) < num_trajectories_per_category:
                        # Choose a target reward within this range. For simplicity, let's use the midpoint or start.
                        current_target_reward = (max_r) # Or just min_r or max_r, depending on desired behavior
                        
                        for _ in range(batch_size_per_target):
                            seed = random.randint(0, 4294967290)
                            # Partial function for the executor to map over
                            func = partial(run_episode_for_pool, seed, alpha, initial_random_factor, current_target_reward)
                            futures.append(executor.submit(func))

                if not futures: # All buckets are satisfied
                    break

                trajectories_this_iteration = []
                for future in tqdm(futures, desc=f"Running Iteration {iteration} Episodes", leave=False):
                    trajectories_this_iteration.append(future.result())

                trajectory_added_to_category_this_iteration = False

                for traj in trajectories_this_iteration:
                    total_reward = traj['total_reward']
                    
                    for range_info in target_ranges_info:
                        category = range_info['category']
                        min_r, max_r = range_info['range']
                        
                        if min_r <= total_reward < max_r:
                            # Add to unique rewards set if not already present
                            if total_reward not in unique_rewards_per_bucket[category]:
                                unique_rewards_per_bucket[category].add(total_reward)
                                all_collected_trajectories[category].append(traj)
                                progress_display.update_progress(
                                    {cat: len(unique_rewards_per_bucket[cat]) for cat in unique_rewards_per_bucket},
                                    trajectory_added=True # Signal that a trajectory was added to a category
                                )
                                trajectory_added_to_category_this_iteration = True
                            break # Trajectory fits only one category

                # Check if all buckets are satisfied after this iteration
                satisfied = True
                for category, unique_rewards in unique_rewards_per_bucket.items():
                    if len(unique_rewards) < num_trajectories_per_category:
                        satisfied = False
                        break
                
                # Update progress display even if no new unique trajectory was added in this specific batch,
                # to show overall status and iteration count.
                progress_display.update_progress(
                    {cat: len(unique_rewards_per_bucket[cat]) for cat in unique_rewards_per_bucket},
                    trajectory_added=trajectory_added_to_category_this_iteration # Only update if actual unique trajectories were added
                )
                
                if satisfied:
                    break

    finally:
        progress_display.finalize(
            {cat: len(unique_rewards_per_bucket[cat]) for cat in unique_rewards_per_bucket}
        )
        progress_display.close()

    print(f"\n🎉 Trajectory collection complete! All buckets now have at least {num_trajectories_per_category} unique entries.")
    print(f"Total iterations: {iteration}")

    # Combine and save all trajectories
    final_combined_trajectories = []
    for category in all_collected_trajectories:
        if len(all_collected_trajectories[category]) > upper_limit_count:
            all_collected_trajectories[category] = random.sample(
                all_collected_trajectories[category], upper_limit_count
            )

    for category, trajectories in all_collected_trajectories.items():
        print(f"Category {get_category_label(next(item['range'] for item in target_ranges_info if item['category'] == category))}: Collected {len(trajectories)} trajectories (Unique Rewards: {len(unique_rewards_per_bucket[category])})")
        final_combined_trajectories.extend(trajectories)

    # Create ULTRA_MAX_TRAJECTORIES folder if it doesn't exist
    output_dir = "TRAJECTORIES"
    os.makedirs(output_dir, exist_ok=True)

    final_filename = os.path.join(
        output_dir,
        f"Trajectories_{min_ret}_{max_ret}_1.pkl",
    )

    with open(final_filename, "wb") as f:
        pickle.dump(final_combined_trajectories, f)
    print(f"\n💾 Combined {len(final_combined_trajectories)} trajectories saved to {final_filename}")

    # Optionally, save reward bucket stats to CSV
    csv_filename = os.path.join(output_dir, "flatlander_reward_buckets_rugged_stats.csv")
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Category",
            "Range",
            "Unique Reward Count",
            "Total Trajectories Collected (for this category)",
        ])
        for range_info in target_ranges_info:
            category = range_info["category"]
            range_label = get_category_label(range_info["range"])
            unique_count = len(unique_rewards_per_bucket[category])
            total_collected_in_category = len(all_collected_trajectories[category])
            writer.writerow(
                [category, range_label, unique_count, total_collected_in_category]
            )
    print(f"📊 Reward bucket statistics saved to: {csv_filename}")

    # 📈 Plot reward distribution across buckets
    bucket_labels = [get_category_label(r["range"]) for r in target_ranges_info]
    bucket_counts = [len(all_collected_trajectories[r["category"]]) for r in target_ranges_info]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(bucket_labels, bucket_counts, color="skyblue", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Reward Range Buckets")
    plt.ylabel("Number of Trajectories")
    plt.title("Trajectories Collected per Reward Bucket")
    plt.tight_layout()

    # Annotate bars with counts
    for bar, count in zip(bars, bucket_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plot_filename = os.path.join(output_dir, "flatlander_bucket_counts.png")
    plt.savefig(plot_filename)
    plt.show()
    print(f"📉 Bucket count plot saved to: {plot_filename}")


if __name__ == "__main__":
    main()