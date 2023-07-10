import gym
import tensorflow as tf
import tensorflow.keras as k 
from tensorflow.keras.layers import Input, Dense, InputLayer
from MDP_based.critic import Critic
from MDP_based.actorcritic import ActorCritic
from MDP_based.mdp_agent import MDPAgent
from MDP_based.valuefunction_approximator import *



ENVIRONMENT = 'CartPole-v1'
ACTION_SPACE_SIZE = 2
STATE_SPACE_DIM = 4
STATE_SPACE_SHAPE = (STATE_SPACE_DIM,)
MEMORY_LEN = int(5e4)
T = int(1e4)
DISCOUNT = 0.99999

"""
Gather experience for n_steps in n_environments, and then train the policy n_epochs times,
using samples from the current memory of size 'batch size'. 
This is repeated a specified number of times.
"""
n_envs = 1000
n_steps = 1
batch_size = int(1e2)
n_epochs = 10
repeat = 1000000


"""
 

 ----------------- TO DO ----------------


Create and implement custom loss function, so that we can apply entropy regularization.
This should also allow making use of the 'advantage' stuff.

Also, implement the better TD-lambda-estimator, which should be better.


----------------------------------------- 




"""
model = k.models.Sequential()
model.add(InputLayer(input_shape=(STATE_SPACE_DIM,)))
model.add(Dense(64, activation='selu'))
model.add(Dense(64, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))
model.compile(optimizer=k.optimizers.Adam(learning_rate=1e-3), loss='mse')


def main():
	print('creating policy\n')
	policy = Neural_Q_Learner(STATE_SPACE_DIM, ACTION_SPACE_SIZE, model, DISCOUNT, T=T, mode='normal')
	# print('creating actor critic\n')
	# policy = ActorCritic(STATE_SPACE_DIM, ACTION_SPACE_SIZE, actor_model, critic)
	# policy.T = T


	get_env = lambda : gym.make(ENVIRONMENT)
	print('creating agent\n')
	agent = MDPAgent(policy, get_env, memory_len=int(MEMORY_LEN))

	rendering_env = gym.make(ENVIRONMENT, render_mode='human')
	train = lambda : agent.train(batch_size, n_epochs, n_steps, n_envs, repeat=repeat)
	score = lambda : agent.score_episode(rendering_env, max_steps=-1)

	while True:
		try:
			answer = input('\ndo you want to train, score or quit?\n')
			match answer:
				case 'train':
					train()
				case 'score':
					rendering_env = gym.make(ENVIRONMENT, render_mode='human')
					score()
				case other:
					return
		except KeyboardInterrupt:
			continue



if __name__ == "__main__":
	main()
