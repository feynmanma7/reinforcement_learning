import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        # log \pi(a_t | s_t)
        log_probs = torch.log(self.actor(states).gather(1, actions))

        # policy gradient, gradient ascent, td_delta * \nabla log_prob
        # for gradient ascent, the loss is -td_delta * \nabla log_prob
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        # mean squared loss, mse(r + gamma * V_{s+1},  V(s_t)
        critic_loss = torch.mean(F.mse_loss(td_target.detach(), self.critic(states)))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []

    for i in range(10):
        with tqdm(total = int(num_episodes / 10),
                  desc = 'Iteration % d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [],
                                   'actions': [],
                                   'next_states': [],
                                   'rewards': [],
                                   'dones': []}
                state, _ = env.reset(seed=0)
                state = np.array(state, dtype=np.float32)
                done = False

                while not done:
                    action = agent.take_action(state)
                    # observation, reward, terminated, truncated, _
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    next_state = np.array(next_state, dtype=np.float32)

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                agent.update(transition_dict)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    return return_list


def test_actor_critic():
    actor_lr = 1e-3
    critic_lr = 1e-2  # why ?
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim,
                        actor_lr, critic_lr, gamma, device)
    return_list = train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format(env_name))
    plt.savefig('Actor-Critic.png')
    plt.close()


if __name__ == '__main__':
    test_actor_critic()
