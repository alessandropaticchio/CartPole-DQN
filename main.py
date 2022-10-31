import gym
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from itertools import count
from replay_memory import ReplayMemory
from utils import get_screen
from dqn import DQN
from policy import select_action
from utils import plot_durations
from train import optimize_model

env = gym.make('CartPole-v0').unwrapped
env.reset()
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_actions = env.action_space.n
init_screen = get_screen(env)
_, _, h, w = init_screen.shape

policy_net = DQN(h, w, n_actions).to(device)
target_net = DQN(h, w, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
episode_durations = []
steps_done = 0

num_episodes = 300
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state=state, policy_net=policy_net, n_actions=n_actions, eps_start=EPS_START,
                               eps_end=EPS_END, eps_decay=EPS_DECAY, device=device,
                               steps_done=steps_done)
        steps_done += 1
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model(memory=memory, policy_net=policy_net, target_net=target_net, optimizer=optimizer, device=device,
                       gamma=GAMMA, batch_size=BATCH_SIZE)
        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
