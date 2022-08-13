from market_env import MarketEnv

from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback, \
    CheckpointCallback, CallbackList
from stable_baselines.common.vec_env import DummyVecEnv



N_TIMESTEP = int(5e4)
N_SAVE_EVERY = int(1e2)
N_EVAL_EPISODES = int(10)
N_EVAL_EVERY = int(1e2)
N_ENVS = 1

SEED = 123
LOGDIR = "lstm006"

logger.configure(folder=LOGDIR)


envs = DummyVecEnv([MarketEnv for _ in range(N_ENVS)])
envs.seed(SEED)

model = PPO2(MlpLnLstmPolicy, envs,
            # lstm001
            # gamma=0.95,
            # lam=0.9,
            # lstm002
            gamma=0.95,
            lam=0.9,
            nminibatches=N_ENVS,
            #policy_kwargs={'n_lstm': 16, 'layers': [4, 4]}, # 005
            policy_kwargs={'n_lstm': 32, 'layers': [8, 8]}, # 006
            # policy_kwargs=dict(
            #     n_lstm=128,
            #     net_arch=[dict(vf=[32,32], pi=[32,32])]
            #     ),
            verbose=2)

eval_callback = EvalCallback(envs,
                            best_model_save_path=LOGDIR,
                            log_path=LOGDIR,
                            eval_freq=N_EVAL_EVERY,
                            render=False,
                            n_eval_episodes=N_EVAL_EPISODES)

cp_callback = CheckpointCallback(save_freq=N_SAVE_EVERY,
                                save_path=LOGDIR,
                                name_prefix="lstm_model")

cb = CallbackList([cp_callback, eval_callback])

model.learn(total_timesteps=N_TIMESTEP, callback=cb)

import os
model.save(os.path.join(LOGDIR, "final_model"))

envs.close()
