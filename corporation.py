import numpy as np

# Initialize parameters
alpha = 0.1  # Learning rate
beta = 0   # Discount rate
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.001
num_episodes = 5000
max_steps = 100

# State and action spaces
states = ['CC', 'CD', 'DC', 'DD']  # Example states
n_actions = 11  # Number of possible actions between 0 and 1 (inclusive)
  # Possible actions: probabilities of choosing C

# Function to initialize utilities based on G and L
def initialize_utilities(G):
    return {
        'CC': (1+G, 1+G),
        'CD': (0,1),
        'DC': (1,0),
        'DD': (1, 1)
    }

# Function to choose action using epsilon-greedy policy
def choose_action(Q, state_idx):
    actions = np.linspace(0, 1, n_actions)
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return actions[np.argmax(Q[state_idx])]

# Function to get state index
def get_state_index(state):
    return states.index(state)

# Function to simulate the environment's response
def take_step(state, action1_prob, action2_prob, utilities):
    action1 = 'C' if np.random.rand() < action1_prob else 'D'
    action2 = 'C' if np.random.rand() < action2_prob else 'D'
    next_state = action1 + action2
    reward1 = utilities[next_state][0]
    reward2 = utilities[next_state][1]
    return next_state, reward1, reward2

# Function to run Q-learning algorithm for given G and L
def run_q_learning(G):
    utilities = initialize_utilities(G)
    actions = np.linspace(0, 1, n_actions)
    # Initialize Q-tables for both players
    Q1 = np.zeros((len(states), len(actions)))
    Q2 = np.zeros((len(states), len(actions)))

    global epsilon  # Use the global epsilon variable
    epsilon = 1.0  # Reset epsilon for each run

    for episode in range(num_episodes):
        state = np.random.choice(states)
        for _ in range(max_steps):
            state_idx = get_state_index(state)
            action1_prob = choose_action(Q1, state_idx)
            action2_prob = choose_action(Q2, state_idx)
            
            next_state, reward1, reward2 = take_step(state, action1_prob, action2_prob, utilities)
            next_state_idx = get_state_index(next_state)
            
            # Q-learning update for player 1
            Q1[state_idx, np.where(actions == action1_prob)[0][0]] += alpha * (
                reward1 + beta * np.max(Q1[next_state_idx]) - Q1[state_idx, np.where(actions == action1_prob)[0][0]]
            )
            
            # Q-learning update for player 2
            Q2[state_idx, np.where(actions == action2_prob)[0][0]] += alpha * (
                reward2 + beta * np.max(Q2[next_state_idx]) - Q2[state_idx, np.where(actions == action2_prob)[0][0]]
            )
            
            state = next_state
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Extract the optimal policy
    optimal_policy1 = [actions[np.argmax(Q1[get_state_index(state)])] for state in states]
    optimal_policy2 = [actions[np.argmax(Q2[get_state_index(state)])] for state in states]

    return optimal_policy1, optimal_policy2

# List of G and L values to test
G_values = [0.5, 1, 1.5]

# Run Q-learning for each combination of G and L and print the results
for n_actions in range(11,15):
    print("co-ordination game")
for n_actions in range(11,15):
  print(n_actions)
  for G in G_values:
    optimal_policy1, optimal_policy2 = run_q_learning(G)
    print(f"Optimal Policies for G = {G}:")
    print("Player 1:")
    for state, action in zip(states, optimal_policy1):
        print(f"  State {state}: Probability of choosing C: {action:.2f}")
    print("Player 2:")
    for state, action in zip(states, optimal_policy2):
        print(f"  State {state}: Probability of choosing C: {action:.2f}")
    print("\n")