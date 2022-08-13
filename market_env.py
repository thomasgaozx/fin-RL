import gym
from gym import spaces
import numpy as np
from pandas import DataFrame
from market_datautil import load_all, load_all_test, normalize0, randepisode
from stable_baselines.common.env_checker import check_env

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

class Dynamic:
    def __init__(self, df:DataFrame, bp, warmup):
        self.df = df
        self.bp = bp
        self.len = self.df.shape[0]
        self.pos: int = 0
        self.v = 100 # initial value
        self.t = 1
        self.warmup = warmup
        #print(df.head())

    def get_observation(self):
        return self.df.iloc[self.t - 1]

    def step(self, a: np.ndarray):
        info = self.df.iloc[self.t]
        _info = self.df.iloc[self.t - 1]

        a:int = round(a.item()) # just to be sure

        # NOTE: assume investing all your value
        # during training for more dramatic reward output
        if self.pos == 1:
            v_ = self.v * info['Close'] / _info['Close']
        else:
            v_ = self.v

        if a != self.pos:
            transaction_cost = v_ * self.bp * 1E-4
            v_ -= transaction_cost

        r = v_ / self.v - 1
        # NOTE: v_ - v works too but when the decision making 
        # sucks ass the v gets very small and punishment will diminish
        # for each bad decision!

        self.t += 1
        done = self.t >= self.len
        self.pos = a

        if self.warmup > 0:
            self.warmup -= 1
        else:
            self.v = v_
        return info, r, done

TRAIN_ASSET_NAMES, TRAIN_ASSET_DFS = load_all()
TEST_ASSET_NAMES, TEST_ASSET_DFS = load_all_test()

class MarketEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['console']}

    def __init__(self, eplen_min=320, eplen_max=None, bp=50, warmup=299, train=True, epsilon=0.05, repeat=4):
        super(MarketEnv, self).__init__()

        self.eplen_min = eplen_min
        self.eplen_max = eplen_max
        self.bp = bp
        self.warmup = warmup
        self.train = train
        self.epsilon = epsilon
        self.repeat = repeat
        self.repeat_cnt = 0

        if train:
            self.asset_names, self.asset_dfs = TRAIN_ASSET_NAMES, TRAIN_ASSET_DFS
        else:
            self.asset_names, self.asset_dfs = TEST_ASSET_NAMES, TEST_ASSET_DFS
        self.asset_sel = 0
        self.dynamic : Dynamic = None

        # 0: doesn't own
        # 1: hold
        self.action_space = spaces.MultiBinary(1)
        
        # state: [price, volume, position]
        # NOTE: apparently tuple observation space is not supported by
        # any algorithms, RIP trying to save training time and accuracy
        # self.observation_space = spaces.Tuple((
        #     spaces.Box(low=np.array([0, 0]),
        #                high=np.array([1, 1]), dtype=np.float32), # [price, volume]
        #     spaces.MultiBinary(1) # position [0,1]
        # ))
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]),
                       high=np.array([1, 1, 1]), dtype=np.float32)

    def step(self, action: np.ndarray):
        # Execute one time step within the environment
        info, reward, done = self.dynamic.step(action)

        moreinfo = {
            "portforlio-value": self.dynamic.v
        }
        s_ = np.array([info['Close'], info['Volume'], action.item()],
            dtype=np.float32)
              
        #print(s_)
        return s_, reward, done, moreinfo

    def reset(self):
        # Reset the state of the environment to an initial state
        if self.train:
            if self.repeat_cnt == 0:
                self.asset_sel = np.random.randint(0, len(self.asset_names))
                df = self.asset_dfs[self.asset_sel]
                hi = self.eplen_max
                if hi is None:
                    hi = df.shape[0]
                
                if np.random.random() > self.epsilon:
                    eplen = np.random.randint(round(self.eplen_min + 3* hi)/4,hi)
                    df = randepisode(df, eplen)
                else:
                    df = df.copy()
                f1, f2 = normalize0(df)
            else:
                df = self.dynamic.df
            self.repeat_cnt = (self.repeat_cnt + 1) % self.repeat
        else:
            df = self.asset_dfs[self.asset_sel].copy()
            f1, f2 = normalize0(df)

            self.asset_sel = (self.asset_sel + 1) % len(self.asset_names)

        self.dynamic = Dynamic(df, self.bp, self.warmup)

        info = self.dynamic.get_observation()
        return np.array([info['Close'], info['Volume'], 0], dtype=np.float32)

    def render(self, mode='console'):
        # if mode != 'console':
        #     raise NotImplementedError()

        print("portforlio value", self.dynamic.v)
        # TODO: wtf

if __name__ == "__main__":
    env = MarketEnv()
    #print(env.observation_space.sample()); exit()
    check_env(env, warn=False)
