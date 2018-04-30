import gym
import numpy

epsilon_probability = 0.025 # this is for randomly selecting the current action very occationally for immediate rewards (exploration vs exploitation)
gamma = 1.0 # is the discount factor
iter = 201 # number of iterations for each episode


number_of_episodes = 5000 #this is the number of episodes


# Learning rate
learning_rate_threshold = 0.004
init_learning_rate = 1.0

# this gives us the number of buckets for each of our observation
num_states = 40



#This is out bucketing algorithm
#each observation is a state, Since it is continuous we discretize it into 40 states for each observation.
#that is 40 states for positions and 40 states for velocity
#this gives us a total combination of 40*40 = 2600 states (discrete velocities and positions)
def obs_to_state(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / num_states
    pos = int((obs[0] - env_low[0])/env_dx[0])
    vel = int((obs[1] - env_low[1])/env_dx[1])
    return pos, vel



def animate_solution(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(iter):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward


def main():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    numpy.random.seed(0)
    print('----- using Q Learning -----')

    # this is a three dimentional Q that is used for learning . Here the first 2 dimentions are for representing the state( position, velocity) and the third dimension is for the action

    Q = numpy.zeros((num_states, num_states, 3))
    for i in range(number_of_episodes):

        total_reward = 0
        obs = env.reset()
        ## eta: learning rate is decreased at each step
        eta = max(learning_rate_threshold, init_learning_rate * (0.85 ** (i // 100)))
        for j in range(iter):
            # this is the current state
            pos, vel = obs_to_state(env, obs)
            if numpy.random.uniform(0, 1) < epsilon_probability:
                action = numpy.random.choice(env.action_space.n)
            else:

                Q_value = Q[pos][vel]
                q_value_future = numpy.exp(Q_value)
                prob = q_value_future / numpy.sum(q_value_future)
                action = numpy.random.choice(env.action_space.n, p=prob)

            obs, reward, done, info  =  env.step(action)
            total_reward += reward
            # update q table
            a_, b_ = obs_to_state(env, obs)
            Q[pos][vel][action] = Q[pos][vel][action] + eta * (
                        reward + gamma * numpy.max(Q[a_][b_]) - Q[pos][vel][action])
            if done:

                break
        if i % 100 == 0:
            print('Iteration #%d -- Total reward = %d. -- moves = %d' % (i + 1, total_reward, j))

    print(Q)

    solution_policy = numpy.argmax(Q, axis=2)
    solution_policy_scores = [animate_solution(env, solution_policy, False) for _ in range(100)]
    print("Average Q Reward = ", numpy.mean(solution_policy_scores))
    animate_solution(env, solution_policy, True)

if __name__ == '__main__':
   main()