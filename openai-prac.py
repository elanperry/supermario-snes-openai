#import gym
import sys
import retro
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
# multiprocess environment

def main():
    print('Main called!')

def train():
    n_cpu = 8
    env = SubprocVecEnv([lambda: retro.make('SuperMarioWorld-Snes', state='YoshiIsland1') for i in range(n_cpu)])
    #model = PPO2.load("ppo2_supermario", env)

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo2_supermario_yoshis-island1")

    del model # remove to demonstrate saving and loading

def test():
    n_cpu = 9
    model = PPO2.load("ppo2_supermario_yoshis-island1")
    env = SubprocVecEnv([lambda: retro.make('SuperMarioWorld-Snes', state='YoshiIsland1') for i in range(n_cpu)])
    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    if globals()[sys.argv[1]] is not None:
        globals()[sys.argv[1]]()
    else:
        main()
