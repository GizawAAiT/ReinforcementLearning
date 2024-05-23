import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Initialize the FrozenLake-v1 environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Define constants
num_actions = env.action_space.n
num_states = env.observation_space.n
gamma = 0.99  # Discount factor


def value_iteration(env, gamma=0.99, theta=1e-6):
    """
    Perform value iteration for the FrozenLake environment.
    
    :param env: FrozenLake environment.
    :param gamma: Discount factor.
    :param theta: Convergence threshold.
    :return: Optimal value function and policy.
    """
    V = np.zeros(num_states)
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            Q_s = [sum([prob * (reward + gamma * V[next_state])
                        for prob, next_state, reward, _ in env.P[s][a]]) for a in range(num_actions)]
            V[s] = max(Q_s)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q_s = [sum([prob * (reward + gamma * V[next_state])
                    for prob, next_state, reward, _ in env.P[s][a]]) for a in range(num_actions)]
        policy[s] = np.argmax(Q_s)
    return V, policy

# Example usage
optimal_values, optimal_policy = value_iteration(env)
print("Optimal Values:\n", optimal_values)
print("Optimal Policy:\n", optimal_policy)


def policy_iteration(env, gamma=0.99):
    """
    Perform policy iteration for the FrozenLake environment.
    
    :param env: FrozenLake environment.
    :param gamma: Discount factor.
    :return: Optimal value function and policy.
    """
    policy = np.random.choice(num_actions, size=num_states)
    V = np.zeros(num_states)

    def policy_evaluation(policy, V, gamma, theta=1e-6):
        while True:
            delta = 0
            for s in range(num_states):
                v = V[s]
                a = policy[s]
                V[s] = sum([prob * (reward + gamma * V[next_state])
                            for prob, next_state, reward, _ in env.P[s][a]])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        return V

    while True:
        V = policy_evaluation(policy, V, gamma)
        policy_stable = True
        for s in range(num_states):
            old_action = policy[s]
            Q_s = [sum([prob * (reward + gamma * V[next_state])
                        for prob, next_state, reward, _ in env.P[s][a]]) for a in range(num_actions)]
            policy[s] = np.argmax(Q_s)
            if old_action != policy[s]:
                policy_stable = False
        if policy_stable:
            break
    return V, policy

# Example usage
optimal_values, optimal_policy = policy_iteration(env)
print("Optimal Values:\n", optimal_values)
print("Optimal Policy:\n", optimal_policy)


def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Perform Q-learning for the FrozenLake environment.
    
    :param env: FrozenLake environment.
    :param num_episodes: Number of episodes for training.
    :param alpha: Learning rate.
    :param gamma: Discount factor.
    :param epsilon: Exploration rate.
    :return: Optimal Q-value function and derived policy.
    """
    Q = np.zeros((num_states, num_actions))
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)
            else:
                action = np.argmax(Q[state])
            next_state, reward, done, _, _ = env.step(action)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
    policy = np.argmax(Q, axis=1)
    return Q, policy

# Example usage
optimal_Q, optimal_policy = q_learning(env)
print("Optimal Q-Values:\n", optimal_Q)
print("Optimal Policy:\n", optimal_policy)


def ucb(env, num_episodes=1000, c=2):
    """
    Perform Upper Confidence Bound (UCB) algorithm for the FrozenLake environment.
    
    :param env: FrozenLake environment.
    :param num_episodes: Number of episodes for training.
    :param c: Exploration parameter.
    :return: Optimal Q-value function and derived policy.
    """
    Q = np.zeros((num_states, num_actions))
    N = np.zeros((num_states, num_actions))  # Number of times each state-action pair has been visited
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        t = 0
        while not done:
            t += 1
            ucb_values = Q[state] + c * np.sqrt(np.log(t + 1) / (N[state] + 1e-5))
            action = np.argmax(ucb_values)
            next_state, reward, done, _, _ = env.step(action)
            N[state, action] += 1
            Q[state, action] += (reward + gamma * np.max(Q[next_state]) - Q[state, action]) / N[state, action]
            state = next_state

    policy = np.argmax(Q, axis=1)
    return Q, policy

# Example usage
optimal_Q, optimal_policy = ucb(env)
print("Optimal Q-Values:\n", optimal_Q)
print("Optimal Policy:\n", optimal_policy)


def plot_policy(env, policy):
    """
    Plot the policy for the FrozenLake environment.
    
    :param env: FrozenLake environment.
    :param policy: Derived policy to visualize.
    """
    lake_size = int(np.sqrt(env.observation_space.n))
    policy_grid = np.reshape(policy, (lake_size, lake_size))
    
    fig, ax = plt.subplots()
    ax.imshow(policy_grid, cmap='cool', interpolation='nearest')

    for i in range(lake_size):
        for j in range(lake_size):
            text = ax.text(j, i, policy_grid[i, j],
                           ha="center", va="center", color="black")
    
    plt.title('Policy Grid')
    plt.show()

# Example usage for plotting the optimal policy from Q-Learning
plot_policy(env, optimal_policy)
