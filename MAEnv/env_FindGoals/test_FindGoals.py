from env_FindGoals import EnvFindGoals
import random

if __name__ == '__main__':
    env = EnvFindGoals()
    max_iter = 10000
    for i in range(max_iter):
        print("iter= ", i)
        action_list = [random.randint(0,4), random.randint(0,4)]
        reward_list, done = env.step(action_list)
        print(env.agt1_pos, env.agt2_pos)
        env.render()


