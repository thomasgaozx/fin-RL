from market_env import MarketEnv

from stable_baselines import A2C
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback, \
    CheckpointCallback, CallbackList
from stable_baselines.common.vec_env import DummyVecEnv



N_TIMESTEP = int(3e6)
N_SAVE_EVERY = 100
N_EVAL_EPISODES = int(50)
N_EVAL_EVERY = 400
N_ENVS = 2048

SEED = 123
LOGDIR = "a2c017"

logger.configure(folder=LOGDIR)



envs = DummyVecEnv([MarketEnv for _ in range(N_ENVS)])
envs.seed(SEED)

model = A2C(MlpLnLstmPolicy, envs,
            gamma=0.95,  # TODO
            alpha=0.9,   # TODO
            momentum=0.05,
            learning_rate=8e-4, # TODO
            lr_schedule='double_middle_drop', # TODO
            policy_kwargs={'n_lstm': 32, 'layers': [16, 16]},    # TODO: bigger, much smaller
            # policy_kwargs=dict(
            #     n_lstm=128,
            #     net_arch=[dict(vf=[32,32], pi=[32,32])]
            #     ),
            # 000: 128
            # 001: 32
            # 002: {'n_lstm': 32, 'layers': [16, 16]}
            # 003: 002 + 100 bp + longer episode + introducing repeat=125
            # 004: {'n_lstm': 64, 'layers': [32, 32]} but repeat=20
            # 005: {'n_lstm': 4, 'layers': [16, 16]} otherwise same as above
            # 006: lower learn rate, more realistic #timesteps
            # 007: policy_kwargs={'n_lstm': 8, 'layers': [8, 8]} bp=50 repeat=10
            # 008: {'n_lstm': 4, 'layers': [8, 8]} same as 007, leverages GPU
            # 009: {'n_lstm': 6, 'layers': [8, 8]}, batch learning
            # 010: {'n_lstm': 64, 'layers': [16, 16]}, repeat=5
            # 011: same but longer
            # 012: {'n_lstm': 128, 'layers': [64, 64]},
            # 014: {'n_lstm': 8, 'layers': [8, 8]}, batch learning
            # 015: {'n_lstm': 4, 'layers': [8, 8]}, batch learning
            # 016: {'n_lstm': 16, 'layers': [16, 16]}, batch learning
            # 017: {'n_lstm': 32, 'layers': [16, 16]}, batch learning

            verbose=2)

eval_callback = EvalCallback(envs.envs[0],
                            best_model_save_path=LOGDIR,
                            log_path=LOGDIR,
                            eval_freq=N_EVAL_EVERY,
                            render=False,
                            deterministic=False,
                            n_eval_episodes=N_EVAL_EPISODES)

cp_callback = CheckpointCallback(save_freq=N_SAVE_EVERY,
                                save_path=LOGDIR,
                                name_prefix="a2c_model")

cb = CallbackList([cp_callback, eval_callback])

model.learn(total_timesteps=N_TIMESTEP, callback=cb)

import os
model.save(os.path.join(LOGDIR, "final_model"))

envs.close()
