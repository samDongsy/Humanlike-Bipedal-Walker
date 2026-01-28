import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_wrapper import CustomRewardWrapper
from new_env_wrapper import new_custom_reward_step
from functools import partial

def main():
    """
    Main training function for PPO BipedalWalker with curriculum learning.
    Sets up environment, callbacks, and trains the model with custom reward wrapper.
    """

    """
    Setup: Define constants, create directories for logs and models.
    """
    log_dir = "ppo_bipedalwalker_logs/"
    model_dir = "ppo_bipedalwalker_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    num_cpu = 20
    TOTAL_TIMESTEPS = 15_000_000

    """
    Environment Creation: Create vectorized training and evaluation environments
    with custom reward wrapper for curriculum learning.
    """
    env_id = "BipedalWalker-v3"
    env = make_vec_env(env_id,
                       n_envs=num_cpu,
                       vec_env_cls=SubprocVecEnv,
                       wrapper_class=partial(new_custom_reward_step, total_training_steps=TOTAL_TIMESTEPS))

    eval_env = make_vec_env(env_id, n_envs=1, wrapper_class=partial(new_custom_reward_step, total_training_steps=TOTAL_TIMESTEPS))

    """
    Callback Setup: Configure evaluation callback with early stopping
    when reward threshold is reached.
    """
    reward_threshold = 100000
    stop_training_callback = StopTrainingOnRewardThreshold(
        reward_threshold=reward_threshold,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_training_callback,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )

    """
    Model Creation: Initialize PPO model with MLP policy.
    """
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device='cpu'
    )

    """
    Training: Train the model with evaluation callback and progress tracking.
    """
    print("--- Starting Training ---")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True
    )
    print("--- Training Finished ---")

    """
    Model Saving: Save the final trained model.
    """
    final_model_name = "ppo_bipedalwalker_new_curriculum_learning_7_11.zip"
    final_model_path = os.path.join(model_dir, final_model_name)
    print(f"Saving final model to: {final_model_path}")
    model.save(final_model_path)

    """
    Model Testing: Load and test the trained model in the environment.
    """
    print("\n--- Loading and Testing a Trained Model ---")
    test_env = make_vec_env(env_id, n_envs=1, wrapper_class=new_custom_reward_step)

    if os.path.exists(final_model_path):
        print(f"Loading final model from: {final_model_path}")
        test_new_model = PPO.load(final_model_path)

    new_obs = test_env.reset()
    new_terminated = False
    new_truncated = False
    new_total_reward = 0

    while not new_terminated and not new_truncated:
        action, _states = test_new_model.predict(new_obs, deterministic=True)
        new_obs, new_reward, new_done, new_info = test_env.step(action)
        new_total_reward += new_reward

if __name__ == '__main__':
    main()