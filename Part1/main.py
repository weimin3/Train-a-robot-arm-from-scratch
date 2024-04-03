"""
Build the basic framework for main.py,rl.py and env.py
"""
from Part1.env import ArmEnv
from Part1.rl import DDPG

MAX_EPISODES = 500
MAX_EP_STEPS = 200 #每个回合200步

# SET ENV
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound #动作范围，机器臂角度范围

# SET RL method
rl = DDPG(a_dim,s_dim,a_bound)

# start training
for i in range(MAX_EPISODES):
    s = env.reset()
    for j in range(MAX_EP_STEPS):
        env.render() #可视化

        a = rl.choose_action(s)

        s_,r,done = env.step(a)

        rl.store_transiton(s,a,r,s_) #返回值放到记忆库中进行离线学习

        if rl.memory_full: #如何记忆库存满了，则开始学习
            rl.learn()

        s = s_ #state变为下一个step的state

"""
summary:
env should at least have:
env.step()
env.reset()
env.render()

RL should at least have:
rl.choose_action()
rl.store_transition()
rl.learn()
rl.memory_full

"""
