import os
import csv
from qgis.core import *
import numpy as np

path_to_root = "E:\\btp\\"


# Loading the centroid layer
path_to_centroid_layer = path_to_root + "centroid_layer.gpkg"
# print(os.path.exists(path_to_centroid_layer))
centroid_layer = QgsVectorLayer(path_to_centroid_layer, "Centroid Layer", "ogr")

if not centroid_layer.isValid():
    print("Centroid Layer failed to load!")
else:
    print("Centroid Layer loaded successfully!")

# Getting the coordinates of the centroids
features = centroid_layer.getFeatures()
centroid_coordinates = []
for i, feature in enumerate(features):
    g = feature.geometry()
    g.convertToSingleType()
    # print(g.asPoint())
    centroid_coordinates.append(g.asPoint())
    if i == 2:
        print(g.asPoint())

print(centroid_coordinates)

d = QgsDistanceArea()
# d.setEllipsoidalMode(True)
d.setEllipsoid('WGS84')
dist = d.measureLine(centroid_coordinates[0], centroid_coordinates[1])
dist = d.measureLine(QgsPointXY(76.75485, 30.73531), QgsPointXY(76.76292, 30.73590))
print(dist)

K_MAX=5

def robust_scaling(data):
    median = np.median(data)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    scaled_data = (data - median) / iqr
    k_step = (np.max(scaled_data) - np.min(scaled_data)) / (K_MAX - 1)
    min_val = np.min(scaled_data)
    print(scaled_data, k_step)
    k_values = list(map(lambda x: int((x - min_val) / k_step) + 1, scaled_data))
    return k_values
    
print(robust_scaling([0,3,4,6,8,15]))


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def render_final_policy(env, policy):
    """Renders the environment with the final policy"""
    state, info = env.reset()
    env.render()
    done = False
    steps = 0
    while not done:
        steps += 1
        action = policy[state]
        state, reward, done, _, info = env.step(action)
        env.render()
        plt.show()

# Implement SARSA on lunar lander environment
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from sklearn.cluster import KMeans

num_episodes = 1000
max_steps_per_episode = 1000
num_clusters = 10000
epsilon = 0.1
alpha = 0.01
gamma = 0.99
env = gym.make('LunarLander-v2')

def encode_state(state):
    state_str = '-'.join(map(str, state))
    state_hash = hashlib.sha256(state_str.encode()).hexdigest()
    return int(state_hash, 16)

kmeans = KMeans(n_clusters=num_clusters)

initial_state, _ = env.reset()
state_shape = initial_state.shape

num_samples = 1000000
state_samples = np.random.uniform(low=-1, high=1, size=(num_samples, state_shape[0]))

kmeans.fit(state_samples)

q_table = np.zeros((num_clusters, env.action_shape.n))

def epsilon_greedy(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])


# SARSA algorithm
episode_rewards = []
steps = []
for episode in range(num_episodes):
    state, _ = env.reset()
    action = epsilon_greedy(kmeans.predict([state])[0], epsilon)
    total_reward = 0
    step = 0
    frames = []
    for step in range(max_steps_per_episode):
        step += 1
        next_state, reward, done, _, info = env.step(action)
        next_action = epsilon_greedy(kmeans.predict([next_state])[0], epsilon)
        state_cluster = kmeans.predict([state])[0]
        next_state_cluster = kmeans.predict([next_state])[0]
        q_table[state_cluster][action] += alpha * (reward + gamma * q_table[next_state_cluster][next_action] - q_table[state_cluster][action])
        state = next_state
        action = next_action
        total_reward += reward
        if episode == 0 or episode == num_episodes/2 or episode == num_episodes-1:
            frames.append(env.render(mode='rgb_array'))
        else:
            env.render()
        if done:
            break
    animation = FuncAnimation(plt.gcf(), animate_frames, frames=frames, interval=50, blit=True)
    plt.show()
    steps.append(step)
    print("Episode {} completed in {} steps with reward {}".format(episode, steps, total_reward))
    episode_rewards.append(total_reward)

plt.subplot(211)
plt.plot(steps)
plt.xlabel('Episode')
plt.ylabel('Max Time Step')
plt.subplot(212)
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_frames(frame):
    plt.imshow(frame)
    plt.clf()
    return plt


from gymnasium.experimental.wrappers import RecordVideoV0

env = RecordVideoV0(env, path='./video', episode_trigger=lambda x: x==0 or x==num_episodes/2 or x==num_episodes-1)

def calculate_policy_from_Q(q_table):
    is_dict = isinstance(q_table, dict)
    if is_dict:
        policy = {}
        for state in q_table.keys():
            policy[state] = np.argmax(q_table[state])
        return policy
    else:
        policy = []
        for state in q_table:
            policy.append(np.argmax(state))
        return policy

import pickle
#save using pickle
with open(f'q_table_rp_alpha_{alpha}.pkl', 'wb') as f:
    pickle.dump(q_table, f)

env = gym.make('CliffWalking-v0', render_mode='rgb_array')

for alpha in alphas:
    with open('data.pickle', 'rb') as f:
        q_table = pickle.load(f)
    q_table = np.load(f'q_table_rp_alpha_{alpha}.npy')
    final_policy = calculate_policy_from_Q(q_table)
    render_final_policy(env, final_policy)

for epsilon in epsilons:
    avg_rewards = run_q_learning(epsilon)
    plt.plot(avg_rewards, label=f'epsilon={epsilon}')

plt.title('Q-learning with epsilon-Greedy Policy on CliffWalking Environment')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.show()