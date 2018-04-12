import numpy as np
import matplotlib.pyplot as plt

import gym
import tensorflow as tf

class PolicyGradientAgent(object):

    def __init__(self, input_size, output_size):

        self.graph = tf.Graph()
        self.session = tf.Session()

        self.state = tf.placeholder(tf.float32, [None, input_size])

        self.fc1 = tf.layers.dense(self.state, 32, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())
        self.fc2 = tf.layers.dense(self.fc1, 32, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal())

        self.output = tf.layers.dense(self.fc2, output_size, kernel_initializer=tf.keras.initializers.he_normal())
        self.action_probs = tf.nn.softmax(self.output)

        self.action_taken = tf.placeholder(tf.int32, [None, 2])
        self.action_prob = tf.gather_nd(self.action_probs, self.action_taken)

        self.advantage = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.reduce_mean(-tf.log(tf.reshape(self.action_prob, [-1, 1]))*self.advantage)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=.001)
        self.train_step = self.optimizer.minimize(loss=self.loss)

        self.init_op = tf.global_variables_initializer()
        self.session.run(self.init_op)


    def get_action_probs(self, state):

        feed_dict = {self.state: state.reshape([-1, 4])}
        return self.session.run(self.action_probs, feed_dict).flatten()


    def take_gradient_step(self, state, advantage, action_taken):

        state = np.array(state).reshape([-1, 4])
        advantage = np.array(advantage).reshape([-1, 1])
        action_taken = [[x, action_taken[x]] for x in range(len(action_taken))]
        feed_dict = {self.state: state,
                     self.advantage: advantage,
                     self.action_taken: action_taken}
        loss, _ = self.session.run([self.loss, self.train_step], feed_dict)
        return loss


def discount_rewards(rewards):

    discounted_rewards = []
    for index, reward in enumerate(rewards):
        discounted_reward = sum((discount_factor**t) * r for t, r in enumerate(rewards[index:]))
        discounted_rewards.append(discounted_reward)
    return discounted_rewards


env = gym.make('CartPole-v0')
agent = PolicyGradientAgent(input_size=4, output_size=2)

num_episodes = 1000
discount_factor = 0.8
sum_rewards = []
mean_rewards = []

for episode in range(num_episodes):

    states = []
    actions = []
    rewards = []

    state = env.reset()
    steps = []
    done = False
    steps = 0

    while not done:

        steps += 1
        action_probs = agent.get_action_probs(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

        discounted_rewards = discount_rewards(rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

    loss = agent.take_gradient_step(state=states,
                                  advantage=discounted_rewards,
                                 action_taken=actions)

    #for timestep in range(steps):
    #     loss = agent.take_gradient_step([states[timestep]], [discounted_rewards[timestep]], [actions[timestep]])

    sum_reward = sum(rewards)
    sum_rewards.append(sum_reward)

    if episode % 100 == 0:
        mean_reward = np.mean(sum_rewards[episode-100:])
        mean_rewards.append(mean_reward)
        print('Episode: {}'.format(episode))
        print('Mean reward: {}'.format(mean_reward))
        print('Loss: {}'.format(loss))

plt.plot(mean_rewards)


for _ in range(10):
    state = env.reset()
    done = False

    episode_reward = 0

    while not done:
        env.render()
        action_probs = agent.get_action_probs(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        state, reward, done, info = env.step(action)
        episode_reward += 1
    print(episode_reward)
env.close()