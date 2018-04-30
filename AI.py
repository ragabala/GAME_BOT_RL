import gym
import numpy

epsilon_probability = 0.025 # this is for randomly selecting the current action very occationally for immediate rewards (exploration vs exploitation)
gamma = 1.0 # is the discount factor
iterations = 201 # number of iterations for each episode
score_iterations = 100


number_of_episodes = 100 #this is the number of episodes


# Learning rate
learning_rate_threshold = 0.004
init_learning_rate = 1.0

# this gives us the number of buckets for each of our observation
num_states = 40



#This is our bucketing algorithm
#each observation is a state, Since it is continuous we discretize it into 40 states for each observation.
#that is 40 states for positions and 40 states for velocity
#this gives us a total combination of 40*40 = 2600 states (discrete velocities and positions)
def build_policy(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / num_states
    pos = int((obs[0] - env_low[0])/env_dx[0])
    vel = int((obs[1] - env_low[1])/env_dx[1])
    return pos, vel



def animate_solution(env, policy=None, render=False):
    '''This is the merhod that uses the policies at the end and animates the entire game
    This works based on starting from the initial state and '''
    obs = env.reset() # this gives the initial state for the game
    total_reward = 0
    step_idx = 0
    for _ in range(iterations):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = build_policy(env, obs)  # if the game is already in progress
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward


def main():
    ''' This is the main function that is used for updating our qvalues and finaly printing out the scores and then animating the game play'''
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    numpy.random.seed(0)
    print('----- using Q Learning -----')

    # this is a three dimentional Q that is used for learning . Here the first 2 dimentions are for representing the state( position, velocity) and the third dimension is for the action

    Q = numpy.zeros((num_states, num_states, 3))
    for i in range(number_of_episodes):
        # this loop is the number of game iterations/episode that are run.. Each episode ends when it comes to the termination state

        total_reward = 0
        obs = env.reset()
        ## eta: learning rate is decreased at each step
        eta = max(learning_rate_threshold, init_learning_rate * (0.85 ** (i // 100)))
        j = 0
        for j in range(iterations):
            # this is the current state
            pos, vel = build_policy(env, obs)
            # once in a while we explore randomly. This is good for not sudden rewards, but good hidden future rewards
            if numpy.random.uniform(0, 1) < epsilon_probability:
                action = numpy.random.choice(env.action_space.n)
            else:
                # else we take an action which is also random but with higher transitional probability
                q_value = Q[pos][vel]
                q_value_future = numpy.exp(q_value)
                prob = q_value_future / numpy.sum(q_value_future)
                action = numpy.random.choice(env.action_space.n, p=prob)

            # once we decide upon what action we will be taking we take that action and collect the reward and check the observations
            obs, reward, done, info  =  env.step(action)
            total_reward += reward
            # update q table
            pos_next, vel_next = build_policy(env, obs)

            # this is the all important step which computes our three dimentional Q table, based upon the actions which we take.
            Q[pos][vel][action] = Q[pos][vel][action] + eta * (
                        reward + gamma * numpy.max(Q[pos_next][vel_next]) - Q[pos][vel][action])
            if done:
                # if the episode ends... we stop this episode and start a new one
                break
        if i % 100 == 0:
            print('Iteration number : %d | moves for completion = %d' % (i + 1, j + 1))


    #This is the actions which are the highest in each state
    solution_policy = numpy.argmax(Q, axis=2)
    # the score for the game is the average of rewards it receives with the the optimal policy for
    average_reward_optimal = [animate_solution(env, solution_policy, False) for _ in range(score_iterations)]
    print("Average Q Reward = ", numpy.mean(average_reward_optimal))
    animate_solution(env, solution_policy, True)
    # we close the window that is opened
    env.close()

if __name__ == '__main__':
   main()