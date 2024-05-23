import numpy as np

class GridWorld:
    def __init__(self, n, m, start, goal, obstacles):
        """
        Initialize the Grid World environment.
        
        :param n: Number of rows.
        :param m: Number of columns.
        :param start: Starting cell as a tuple (row, col).
        :param goal: Goal cell as a tuple (row, col).
        :param obstacles: List of obstacle cells as tuples [(row, col), ...].
        """
        self.n = n
        self.m = m
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.grid = np.zeros((n, m))
        
        for obs in obstacles:
            self.grid[obs] = -10  # Obstacle penalty

        self.grid[goal] = 10  # Goal reward

    def is_valid(self, state):
        """
        Check if a state is within the grid and not an obstacle.
        
        :param state: Tuple (row, col).
        :return: True if valid, False otherwise.
        """
        r, c = state
        if 0 <= r < self.n and 0 <= c < self.m and self.grid[state] != -10:
            return True
        return False

    def get_next_state(self, state, action):
        """
        Get the next state given the current state and action.
        
        :param state: Tuple (row, col).
        :param action: Action to take (up, down, left, right).
        :return: Next state as a tuple (row, col).
        """
        actions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        r, c = state
        dr, dc = actions[action]
        next_state = (r + dr, c + dc)
        
        if self.is_valid(next_state):
            return next_state
        return state  # If invalid, remain in the same state

    def get_reward(self, state):
        """
        Get the reward for a given state.
        
        :param state: Tuple (row, col).
        :return: Reward value.
        """
        if state == self.goal:
            return 10
        elif self.grid[state] == -10:
            return -10
        else:
            return -1  # Step penalty

    def reset(self):
        """
        Reset the environment to the starting state.
        
        :return: Starting state.
        """
        return self.start

def value_iteration(env, gamma=0.9, theta=1e-6):
    """
    Perform value iteration algorithm to find the optimal value function.
    
    :param env: GridWorld environment.
    :param gamma: Discount factor.
    :param theta: Convergence threshold.
    :return: Optimal value function and policy.
    """
    V = np.zeros((env.n, env.m))
    policy = np.zeros((env.n, env.m), dtype=object)
    
    while True:
        delta = 0
        for r in range(env.n):
            for c in range(env.m):
                state = (r, c)
                if state == env.goal or env.grid[state] == -10:
                    continue
                
                v = V[state]
                action_values = []
                for action in ['up', 'down', 'left', 'right']:
                    next_state = env.get_next_state(state, action)
                    reward = env.get_reward(next_state)
                    action_values.append(reward + gamma * V[next_state])
                
                V[state] = max(action_values)
                policy[state] = ['up', 'down', 'left', 'right'][np.argmax(action_values)]
                delta = max(delta, abs(v - V[state]))
        
        if delta < theta:
            break
    
    return V, policy

# Example usage
n, m = 5, 5
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 1), (2, 2), (3, 3)]
env = GridWorld(n, m, start, goal, obstacles)
optimal_values, optimal_policy = value_iteration(env)

print("Optimal Values:\n", optimal_values)
print("Optimal Policy:\n", optimal_policy)

def policy_iteration(env, gamma=0.9):
    """
    Perform policy iteration algorithm to find the optimal policy.
    
    :param env: GridWorld environment.
    :param gamma: Discount factor.
    :return: Optimal value function and policy.
    """
    def policy_evaluation(policy, V, gamma, theta=1e-6):
        while True:
            delta = 0
            for r in range(env.n):
                for c in range(env.m):
                    state = (r, c)
                    if state == env.goal or env.grid[state] == -10:
                        continue

                    v = V[state]
                    action = policy[state]
                    next_state = env.get_next_state(state, action)
                    reward = env.get_reward(next_state)
                    V[state] = reward + gamma * V[next_state]
                    delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break
        return V

    policy = np.random.choice(['up', 'down', 'left', 'right'], size=(env.n, env.m))
    V = np.zeros((env.n, env.m))
    
    while True:
        V = policy_evaluation(policy, V, gamma)
        policy_stable = True
        
        for r in range(env.n):
            for c in range(env.m):
                state = (r, c)
                if state == env.goal or env.grid[state] == -10:
                    continue

                old_action = policy[state]
                action_values = []
                for action in ['up', 'down', 'left', 'right']:
                    next_state = env.get_next_state(state, action)
                    reward = env.get_reward(next_state)
                    action_values.append(reward + gamma * V[next_state])

                best_action = ['up', 'down', 'left', 'right'][np.argmax(action_values)]
                policy[state] = best_action
                
                if old_action != best_action:
                    policy_stable = False
        
        if policy_stable:
            break
    
    return V, policy

# Example usage
optimal_values, optimal_policy = policy_iteration(env)

print("Optimal Values:\n", optimal_values)
print("Optimal Policy:\n", optimal_policy)


def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Perform Q-learning algorithm to find the optimal action-value function.
    
    :param env: GridWorld environment.
    :param num_episodes: Number of episodes for training.
    :param alpha: Learning rate.
    :param gamma: Discount factor.
    :param epsilon: Exploration rate.
    :return: Optimal Q-value function and derived policy.
    """
    Q = np.zeros((env.n, env.m, 4))
    actions = ['up', 'down', 'left', 'right']
    
    def epsilon_greedy_policy(state):
        if np.random.rand() < epsilon:
            return np.random.choice(actions)
        else:
            return actions[np.argmax(Q[state[0], state[1], :])]

    for episode in range(num_episodes):
        state = env.reset()
        
        while state != env.goal:
            action = epsilon_greedy_policy(state)
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(next_state)
            action_idx = actions.index(action)
            
            best_next_action = np.argmax(Q[next_state[0], next_state[1], :])
            Q[state[0], state[1], action_idx] += alpha * (reward + gamma * Q[next_state[0], next_state[1], best_next_action] - Q[state[0], state[1], action_idx])
            
            state = next_state
    
    policy = np.zeros((env.n, env.m), dtype=object)
    for r in range(env.n):
        for c in range(env.m):
            policy[r, c] = actions[np.argmax(Q[r, c, :])]
    
    return Q, policy

# Example usage
optimal_Q, optimal_policy = q_learning(env)

print("Optimal Q-Values:\n", optimal_Q)
print("Optimal Policy:\n", optimal_policy)


def ucb(env, num_episodes=1000, alpha=0.1, gamma=0.9, c=2):
    """
    Perform Upper Confidence Bound (UCB) algorithm to balance exploration and exploitation.
    
    :param env: GridWorld environment.
    :param num_episodes: Number of episodes for training.
    :param alpha: Learning rate.
    :param gamma: Discount factor.
    :param c: Exploration parameter.
    :return: Optimal Q-value function and derived policy.
    """
    Q = np.zeros((env.n, env.m, 4))
    N = np.zeros((env.n, env.m, 4))  # Number of times each state-action pair has been visited
    actions = ['up', 'down', 'left', 'right']
    
    def ucb_policy(state, t):
        ucb_values = Q[state[0], state[1], :] + c * np.sqrt(np.log(t + 1) / (N[state[0], state[1], :] + 1e-5))
        return actions[np.argmax(ucb_values)]

    for episode in range(num_episodes):
        state = env.reset()
        t = 0
        
        while state != env.goal:
            t += 1
            action = ucb_policy(state, t)
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(next_state)
            action_idx = actions.index(action)
            
            N[state[0], state[1], action_idx] += 1
            Q[state[0], state[1], action_idx] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action_idx])
            
            state = next_state
    
    policy = np.zeros((env.n, env.m), dtype=object)
    for r in range(env.n):
        for c in range(env.m):
            policy[r, c] = actions[np.argmax(Q[r, c, :])]
    
    return Q, policy

# Example usage
n, m = 5, 5
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 1), (2, 2), (3, 3)]
env = GridWorld(n, m, start, goal, obstacles)
optimal_Q, optimal_policy = ucb(env)

print("Optimal Q-Values:\n", optimal_Q)
print("Optimal Policy:\n", optimal_policy)
