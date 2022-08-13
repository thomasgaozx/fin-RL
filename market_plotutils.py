import numpy as np
import glob
import matplotlib.pyplot as plt
from market_env import MarketEnv
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.ppo2 import PPO2
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy


def plot_market_debug(policy_func):
    """policy_func(t) returns action"""
    ts = []
    rs = []
    vs = []
    ps = []
    t = 0
    d = False
    model = MarketEnv()
    model.reset()

    while not d:
        p = policy_func(t)
        _, r, d, _ = model.step(np.array([p]))
        ts.append(t)
        rs.append(r)
        vs.append(model.dynamic.v)
        ps.append(p)
        t += 1

    plt.figure()
    plt.plot(ts, rs)
    plt.title("reward")
    plt.figure()
    plt.plot(ts, vs)
    plt.title("portforlio value")
    plt.figure()
    plt.plot(ts, ps)
    plt.title("policy")
    plt.figure()
    model.dynamic.df.plot()
    plt.title("episode price history")

    rs = np.array(rs)
    print("reward mean", rs.mean(), "reward deviation", rs.std())

def plot_comparison(ModelType:BaseRLModel, modelfile, title, test_csv=0):
    """single episode asset comparison"""
    model = ModelType.load(modelfile, env=None)
    v0s = []
    v1s = []
    r0s = []
    r1s = []
    a1s = []

    env0 = MarketEnv(train=False)
    envs = DummyVecEnv([lambda: MarketEnv(train=False) for _ in range(model.n_envs)])
    for _ in range(test_csv + 1):
        env0.reset()
        obs = envs.reset()

    s = None
    ds = np.array([False for _ in range(model.n_envs)])
    while not ds[0]:
        a, s = model.predict(obs, state=s, mask=ds, deterministic=True)
        _, r0, _, _ = env0.step(np.array([1]))
        obs, rs, ds, _ = envs.step(a)
        r0s.append(r0)
        r1s.append(rs.mean())
        v0s.append(env0.dynamic.v)
        v1s.append(sum(envs.get_attr('dynamic')[i].v for i in range(model.n_envs))/model.n_envs)
        a1s.append(a[0])
    ts = np.arange(len(r0s))
    print(f"Asset {test_csv}:")
    print("long reward", np.array(r0s).sum())
    print("policy reward", np.array(r1s).sum())
    fig, axs = plt.subplots(2,1,figsize=(16,9), gridspec_kw={'height_ratios': [1, 2]})
    axs[0].plot(ts, r0s, label="long policy")
    axs[0].plot(ts, r1s, label="model")
    axs[0].plot(ts, np.array(a1s) * abs(np.array(r0s)).mean(), color='red', label='model policy')
    axs[0].legend()
    axs[0].set_title(f"Reward plot of asset {test_csv}, " + title)
    #plt.figure()
    axs[1].plot(ts, v0s, label="long policy")
    axs[1].plot(ts, v1s, label="model")
    axs[1].legend()
    axs[1].set_title(f"Agent Networth, asset {test_csv}, " + title)
    # plt.figure()
    # env0.dynamic.df.plot()
    #plt.title(f"Asset {test_csv} quote")
    #plt.show()

def plot_model_validation(ModelType:BaseRLModel, model_dir, title):
    """return list of names and df's,
    will worry bout memory if it becomes a problem"""
    names = []
    dfs = []
    mfiles = glob.glob(f"{model_dir}/*.zip")
    mfiles = [ f for f in mfiles if "best_model" not in f and "final_model" not in f]
    mfiles = np.array(mfiles)

    ts = [ int(f.split('_')[2]) for f in mfiles]
    ts = np.array(ts)
    idx = ts.argsort()
    mfiles = mfiles[idx]
    ts = ts[idx]

    r0s = []
    r1s = []
    r2s = []    
    r3s = []    
    r4s = []    
    for fname in mfiles:
        print("====================== evaluating ", fname)
        model = ModelType.load(fname, env=None)
        envs = DummyVecEnv([lambda :MarketEnv(train=False)])
        #envs = DummyVecEnv([lambda: MarketEnv(train=False) for _ in range(model.n_envs)])
        rewards, _ = evaluate_policy(model, envs,
                            n_eval_episodes=5,
                            deterministic=True,
                            return_episode_rewards=True)
        rewards = np.array(rewards)
        print(rewards.shape, rewards)
        r0s.append(rewards[0].item())
        r1s.append(rewards[1].item())
        r2s.append(rewards[2].item())
        r3s.append(rewards[3].item())
        r4s.append(rewards[4].item())

    print(ts, r0s, r1s, r2s, r3s, r4s)

    plt.figure()
    plt.plot(ts, r0s, label="asset 0 reward")
    plt.plot(ts, r1s, label="asset 1 reward")
    plt.plot(ts, r2s, label="asset 2 reward")
    plt.plot(ts, r3s, label="asset 3 reward")
    plt.plot(ts, r4s, label="asset 4 reward")
    plt.xlabel("timesteps")
    plt.ylabel("reward")
    plt.legend()
    plt.title(title)

def plot_rewards(evalfile, title):
    """evalfile is evaluations.npz"""
    #np.savez_compressed(evalfile)
    obj = np.load(evalfile)
    Xs = obj['timesteps']
    Ys = obj['results'].mean(axis=1).reshape(-1)
    Es = obj['results'].std(axis=1).reshape(-1)

    plt.figure()
    plt.errorbar(Xs, Ys, Es)
    plt.xlabel("timesteps")
    plt.ylabel("mean reward")
    plt.title(title)
    #plt.show()

if __name__ == "__main__":
    #plot_market_debug(lambda x:1)
    #plot_model_validation(A2C, "a2c002", "A2C Hyperparam [32 LSTM cells, 16x16 MLP, repeat=20, 100bp]")
    #plot_model_validation(A2C, "a2c005", "A2C Hyperparam [4 LSTM cells, 16x16 MLP, repeat=20, 100bp]")
    # plot_model_validation(A2C, "a2c006", "A2C Hyperparam [4 LSTM cells, 16x16 MLP, repeat=20, 100bp, learning_rate=8e-4]")
    # plot_model_validation(A2C, "a2c007", "A2C Hyperparam [8 LSTM cells, 8x8 MLP, repeat=10, 50bp, learning_rate=8e-4]")
    # plot_model_validation(A2C, "a2c007", "A2C Hyperparam [8 LSTM cells, 8x8 MLP, repeat=10, 50bp, learning_rate=8e-4]")
    #plot_model_validation(A2C, "a2c011", "A2C Hyperparam [64 LSTM cells, 16x16 MLP, repeat=5, 50bp, learning_rate=8e-4]")
    #plot_model_validation(A2C, "a2c012", "A2C Hyperparam [128 LSTM cells, 64x64 MLP, repeat=5, 50bp, learning_rate=8e-4]")
    #plot_model_validation(A2C, "a2c014", "A2C Hyperparam [8 LSTM cells, 8x8 MLP, repeat=5, 50bp, learning_rate=8e-4]")
    #plot_model_validation(A2C, "a2c015", "A2C Hyperparam [4 LSTM cells, 8x8 MLP, repeat=5, 50bp, learning_rate=8e-4]")
    #plot_model_validation(A2C, "a2c016", "A2C Hyperparam [16 LSTM cells, 16x16 MLP, repeat=5, 50bp, learning_rate=8e-4]")
    #plot_model_validation(A2C, "a2c017", "A2C Hyperparam [32 LSTM cells, 16x16 MLP, repeat=5, 50bp, learning_rate=8e-4]")

    #for i in range(5):#{'n_lstm': 64, 'layers': [16, 16]}, repeat=5
        #plot_comparison(A2C, "a2c010/final_model.zip", "A2C Hyperparam [64 LSTM cells, 16x16 MLP, repeat=5, 50bp]", test_csv=i)
        #plot_comparison(A2C, "a2c011/a2c_model_409600_steps.zip", "A2C Hyperparam [64 LSTM cells, 16x16 MLP, repeat=5, 50bp]", test_csv=i)
        #plot_comparison(A2C, "a2c009/final_model.zip", "A2C Hyperparam [6 LSTM cells, 8x8 MLP, repeat=5, 50bp], batch learning", test_csv=i)
        #plot_comparison(A2C, "a2c008/final_model.zip", "A2C Hyperparam [6 LSTM cells, 8x8 MLP, repeat=10, 50bp]", test_csv=i)
        #plot_comparison(A2C, "a2c002/a2c_model_11000_steps.zip", "A2C Hyperparam [32 LSTM cells, 16x16 MLP, repeat=20, 100bp]", test_csv=i)
        #plot_comparison(A2C, "a2c004/best_model.zip", "A2C Hyperparam [64 LSTM cells, 32x32 MLP, repeat=20, 100bp]", test_csv=i)
        #plot_comparison(A2C, "a2c005/a2c_model_500000_steps.zip", "A2C Hyperparam [4 LSTM cells, 16x16 MLP, repeat=20, 100bp]", test_csv=i)
        #plot_comparison(A2C, "a2c006/a2c_model_70000_steps.zip", "A2C Hyperparam [4 LSTM cells, 16x16 MLP, repeat=20, 100bp, learning_rate=8e-4]", test_csv=i)
        #plot_comparison(A2C, "a2c007/a2c_model_11000_steps.zip", "A2C Hyperparam [8 LSTM cells, 8x8 MLP, repeat=10, 50bp, learning_rate=8e-4]", test_csv=i)

    # for i in range(1,9):
    #     for j in range(5):
    #         plot_comparison(A2C, f"a2c005/a2c_model_{i}00000_steps.zip", f"A2C Hyperparam [4 LSTM cells, 16x16 MLP, repeat=20, 100bp, iter={i}00000", test_csv=j)
    #     plt.show()

    #plot_comparison(A2C, "a2c005/a2c_model_500000_steps.zip", "A2C Hyperparam [4 LSTM cells, 16x16 MLP, repeat=20, 100bp]", test_csv=0)
    #plot_comparison(A2C, "a2c007/a2c_model_17000_steps.zip", "A2C Hyperparam [8 LSTM cells, 8x8 MLP, repeat=10, 50bp, learning_rate=8e-4]", test_csv=0)
    #plot_comparison(A2C, "a2c000/a2c_model_50000_steps.zip", "A2C Hyperparam [128 LSTM cells]", test_csv=2)
    # for i in [2,4,8, 12, 20, 30, 50]:
    #     plot_comparison(PPO2, f"lstm004/lstm_model_{i}00_steps.zip", f"PPO Hyperparam [default], iter={i}00", test_csv=0)
    #plot_comparison(PPO2, "lstm004/lstm_model_500_steps.zip", "PPO Hyperparam [default], iter=500", test_csv=0)
    # for i in range(4,8):
    #     plot_rewards(f'a2c00{i}/evaluations.npz', f"A2C parameter set a2c00{i}")
    #plot_rewards(f'a2c011/evaluations.npz', f"A2C Hyperparam [64 LSTM cells, 16x16 MLP, repeat=5, 50bp, learning_rate=8e-4]")
    #plot_rewards(f'a2c012/evaluations.npz', f"A2C Hyperparam [128 LSTM cells, 64x64 MLP, repeat=5, 50bp, learning_rate=8e-4]")
    # a2c011\a2c_model_1254400_steps.zip
    for i in range(5):
        plot_comparison(A2C, "a2c014/a2c_model_409600_steps.zip", "A2C Hyperparam [8 LSTM cells, 8x8 MLP, repeat=5, 50bp]", test_csv=i)

    plt.show()

