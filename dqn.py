import collections
import gym
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn.functional as F 
from tqdm import tqdm


class ReplayBuffer():
    def __init__(self, capacity) -> None:
        # first in, first out
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done =zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, 
                 learning_rate, gamma, epsilon, 
                 target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)

        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), 
                                          lr = learning_rate)
        
        self.gamma = gamma 
        self.epsilon = epsilon
        self.target_update = target_update  # ferquency to udate target net
        self.count = 0
        self.device = device

        self.EPS_START = 0.9
        self.EPS_END = 0.01
        self.EPS_DECAY = 2500
        self.TAU = 0.005

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.tensor(np.array([state], dtype=np.float32), dtype=torch.float32).to(self.device)
                action = self.q_net(state).argmax().item()

        return action
        
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], 
                              dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions'])\
            .view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], 
                               dtype=torch.float32)\
                                .view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float32).view(-1, 1).to(self.device)
        
        # q_net: [batch, action_dim]
        # q_values: [batch, 1], for q_value
        q_values = torch.gather(input=self.q_net(states), dim=1, index=actions)

        # target_q_net: [batch, action_dim]
        # max_q_values: [batch, 1]
        with torch.no_grad():
            max_next_q_values = torch.max(self.target_q_net(next_states), dim=1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            # Update the target network periodically.
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


def test_dqn():
    lr = 3e-4
    num_episodes = 600
    hidden_dim = 128
    gamma = 0.99
    epsilon = 0.01

    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    #env.seed(0)
    torch.manual_seed(0)

    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.n

    print('state_dim', state_dim)
    print('hidden_dim', hidden_dim)
    print('action_dim', action_dim)

    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0 
                state, _ = env.reset(seed=0)
                state = np.array(state, dtype=np.float32)
                done = False

                while not done:
                    action = agent.take_action(state)
                    #next_state, reward, done, _, _ = env.step(action)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    next_state = np.array(next_state, dtype=np.float32)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)

                        transition_dict = {
                            'states': b_s, 
                            'actions': b_a, 
                            'next_states': b_ns, 
                            'rewards': b_r, 
                            'dones': b_d
                        }
                        agent.update(transition_dict)

                return_list.append(episode_return)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': 
                        '%d' % (num_episodes / 10 * i + i_episode + 1), 
                        'return':
                        '%.3f' % np.mean(return_list[-10:])}
                    )
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.title("DQN")
    plt.savefig("dqn.png")
    #plt.close()


if __name__ == '__main__':
    test_dqn()



