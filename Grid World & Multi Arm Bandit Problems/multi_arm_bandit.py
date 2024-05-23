import numpy as np

class MultiArmedBandit:
    def __init__(self, K):
        """
        Initialize the Multi-Armed Bandit environment.
        
        :param K: Number of arms (actions).
        """
        self.K = K
        self.means = np.random.rand(K)  # True mean rewards for each arm
        self.reset()

    def reset(self):
        """
        Reset the environment.
        
        :return: Initial state (always 0 for single-state).
        """
        return 0  # Only one state in the bandit problem

    def step(self, action):
        """
        Take an action (pull an arm) and return the reward.
        
        :param action: Action to take (which arm to pull).
        :return: Tuple of (next state, reward).
        """
        reward = np.random.randn() + self.means[action]
        return 0, reward

# Example usage
K = 10
bandit_env = MultiArmedBandit(K)


def value_iteration_bandit(env, gamma=0.9, theta=1e-6):
    """
    Perform value iteration for the multi-armed bandit problem.
    
    :param env: MultiArmedBandit environment.
    :param gamma: Discount factor.
    :param theta: Convergence threshold.
    :return: Optimal value function and policy.
    """
    V = np.zeros(env.K)
    
    while True:
        delta = 0
        for k in range(env.K):
            v = V[k]
            V[k] = env.means[k]  # In a bandit problem, the value is just the mean reward
            delta = max(delta, abs(v - V[k]))
        
        if delta < theta:
            break
    
    policy = np.argmax(V)
    return V, policy

# Example usage
optimal_values, optimal_policy = value_iteration_bandit(bandit_env)

print("Optimal Values:\n", optimal_values)
print("Optimal Policy:\n", optimal_policy)


def policy_iteration_bandit(env, gamma=0.9):
    """
    Perform policy iteration for the multi-armed bandit problem.
    
    :param env: MultiArmedBandit environment.
    :param gamma: Discount factor.
    :return: Optimal value function and policy.
    """
    policy = np.random.choice(env.K)
    V = np.zeros(env.K)
    
    while True:
        # Policy Evaluation
        for k in range(env.K):
            V[k] = env.means[policy]
        
        # Policy Improvement
        old_policy = policy
        policy = np.argmax(V)
        
        if policy == old_policy:
            break
    
    return V, policy

# Example usage
optimal_values, optimal_policy = policy_iteration_bandit(bandit_env)

print("Optimal Values:\n", optimal_values)
print("Optimal Policy:\n", optimal_policy)


def q_learning_bandit(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Perform Q-learning for the multi-armed bandit problem.
    
    :param env: MultiArmedBandit environment.
    :param num_episodes: Number of episodes for training.
    :param alpha: Learning rate.
    :param gamma: Discount factor.
    :param epsilon: Exploration rate.
    :return: Optimal Q-value function and derived policy.
    """
    Q = np.zeros(env.K)
    N = np.zeros(env.K)  # Number of times each arm has been pulled
    
    for episode in range(num_episodes):
        state = env.reset()
        
        if np.random.rand() < epsilon:
            action = np.random.choice(env.K)
        else:
            action = np.argmax(Q)
        
        next_state, reward = env.step(action)
        N[action] += 1
        Q[action] += alpha * (reward - Q[action])
    
    policy = np.argmax(Q)
    return Q, policy

# Example usage
optimal_Q, optimal_policy = q_learning_bandit(bandit_env)

print("Optimal Q-Values:\n", optimal_Q)
print("Optimal Policy:\n", optimal_policy)


def ucb_bandit(env, num_episodes=1000, c=2):
    """
    Perform Upper Confidence Bound (UCB) algorithm for the multi-armed bandit problem.
    
    :param env: MultiArmedBandit environment.
    :param num_episodes: Number of episodes for training.
    :param c: Exploration parameter.
    :return: Optimal Q-value function and derived policy.
    """
    Q = np.zeros(env.K)
    N = np.zeros(env.K)  # Number of times each arm has been pulled
    
    for episode in range(num_episodes):
        state = env.reset()
        
        ucb_values = Q + c * np.sqrt(np.log(episode + 1) / (N + 1e-5))
        action = np.argmax(ucb_values)
        
        next_state, reward = env.step(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
    
    policy = np.argmax(Q)
    return Q, policy

# Example usage
optimal_Q, optimal_policy = ucb_bandit(bandit_env)

print("Optimal Q-Values:\n", optimal_Q)
print("Optimal Policy:\n", optimal_policy)
