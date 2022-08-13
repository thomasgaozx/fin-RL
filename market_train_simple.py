from market_env import MarketEnv

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback, \
    CheckpointCallback, CallbackList

N_TIMESTEP = int(5e4)
N_SAVE_EVERY = int(1e4)
N_EVAL_EPISODES = int(1e3)
N_EVAL_EVERY = int(1e4)

SEED = 0
LOGDIR = "testing000"

logger.configure(folder=LOGDIR)

env = MarketEnv()
env.seed(SEED)

model = PPO1(MlpPolicy, env,
            timesteps_per_actorbatch=256,
            clip_param=0.2,
            entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=1e-3,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
            policy_kwargs=dict(net_arch=[dict(vf=[128,128], pi=[128,128])]),
            verbose=2)

eval_callback = EvalCallback(env,
                            best_model_save_path=LOGDIR,
                            log_path=LOGDIR,
                            eval_freq=N_EVAL_EVERY,
                            n_eval_episodes=N_EVAL_EPISODES)

cp_callback = CheckpointCallback(save_freq=N_SAVE_EVERY,
                                save_path=LOGDIR,
                                name_prefix="simple_model")

cb = CallbackList([eval_callback, cp_callback])

model.learn(total_timesteps=N_TIMESTEP, callback=cp_callback)

import os
model.save(os.path.join(LOGDIR, "final_model"))

env.close()
