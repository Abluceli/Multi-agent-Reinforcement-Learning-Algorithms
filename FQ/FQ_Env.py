from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import numpy as np
class FQ_Env(object):
    def __init__(self):
        self.engine_configuration_channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(side_channels=[self.engine_configuration_channel])
        self.engine_configuration_channel.set_configuration_parameters(
            # width = 84,
            # height = 84,
            # quality_level = 5, #1-5
            time_scale = 1  # 1-100
            # target_frame_rate = 60, #1-60
            # capture_frame_rate = 60 #default 60
        )

        self.reset()

        self.n = self.agent_num()

        self.state_shapes = [self.env.get_behavior_spec(behavior_name).observation_shapes[0][0] for behavior_name in
                             self.env.get_behavior_names()]
        self.action_dims = [self.env.get_behavior_spec(behavior_name).action_shape for behavior_name in
                             self.env.get_behavior_names()]
    def agent_num(self):
        behavior_names = self.env.get_behavior_names()
        agent_num = len(behavior_names)
        return agent_num

    def reset(self):
        self.env.reset()
        cur_state = []
        for behavior_name in self.env.get_behavior_names():
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_name)
            cur_state.append(DecisionSteps.obs[0][0])
        return cur_state

    def step(self, actions):
        next_state = []
        reward = []
        done = []
        for behavior_name_index, behavior_name in enumerate(self.env.get_behavior_names()):
            self.env.set_actions(behavior_name=behavior_name, action=np.asarray([actions[behavior_name_index]]))
        self.env.step()

        for behavior_name in self.env.get_behavior_names():
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_name)
            if len(TerminalSteps.reward) == 0:
                next_state.append(DecisionSteps.obs[0][0])
                reward.append(DecisionSteps.reward[0])
                done.append(False)
            else:
                next_state.append(TerminalSteps.obs[0][0])
                reward.append(TerminalSteps.reward[0])
                done.append(True)

        return next_state, reward, done

    def close(self):
        self.env.close()

def try_env():
    env = UnityEnvironment()
    env.reset()
    behavior_names = env.get_behavior_names()
    for behavior_name in behavior_names:
        BehaviorSpec = env.get_behavior_spec(behavior_name)
        print(behavior_name+" "+str(BehaviorSpec))

        DecisionSteps, TerminalSteps = env.get_steps(behavior_name)

        print(DecisionSteps.agent_id)
        print(DecisionSteps.reward)
        print(DecisionSteps.obs)
        print(TerminalSteps.agent_id)
        print(TerminalSteps.reward)
        print(TerminalSteps.obs)
    env.close()



if __name__ == "__main__":
    env = FQ_Env()
    cur_state = env.reset()
    #print(cur_state)
    step = 0
    while True:
        actions = np.random.normal(0, 1, (2, 2))
        next_state, reward, done = env.step(actions)
        step = step + 1
        if all(done):
            print(str(step), str(done))
            break


