import os

import gym
import numpy as np
import tensorflow as tf

import gym_pytris
from src.utils import plotLearning

MODEL_PATH = "models/"
PLOT_CURRENT = "current_model.png"


class ReplayBuffer():
	def __init__(self, max_size, input_dims):
		self.mem_size = max_size
		self.mem_cntr = 0

		self.state_memory = np.zeros((self.mem_size, *input_dims), 
																dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, *input_dims),
														dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

	def store_transition(self, state, action, reward, state_, done):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.reward_memory[index] = reward
		self.action_memory[index] = action
		self.terminal_memory[index] = 1 - int(done)
		self.mem_cntr += 1

	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size, replace=False)

		states = self.state_memory[batch]
		states_ = self.new_state_memory[batch]
		rewards = self.reward_memory[batch]
		actions = self.action_memory[batch]
		terminal = self.terminal_memory[batch]

		return states, actions, rewards, states_, terminal

class QAgent:
    
	def __init__(
		self,
		env_name="LunarLander-v2",
		hidden_layers=[128, 64, 32],
		learning_rate=.001,
		epoch=500,
		gamma=.99,
		epsilon=1.0,
		epsilon_dec=.001,
		epsilon_end=.01,
		mem_size=1000000,
		batch_size=64,
		activation_function = "softmax",
		render_during_training = False,
		live_plot = False
	):
		tf.compat.v1.disable_eager_execution()
		self.env_name = env_name
		if render_during_training:
			self.env = gym.make(env_name, render_mode="human")
		else:
			self.env = gym.make(env_name)
		self.hidden_layers = hidden_layers
		self.learning_rate = learning_rate
		self.epoch = epoch
		self.current_epoch = 0
		self.gamma = gamma
		self.init_epsilon = epsilon
		self.epsilon = epsilon
		self.epsilon_dec = epsilon_dec
		self.epsilon_end = epsilon_end
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.activation_function = activation_function
		self.render_during_training = render_during_training
		self.live_plot = live_plot

		self.training_done = False

		# On crée le modèle
		self.create_model()

		# Load model if exists
		self.load()

		# On crée la mémoire
		self.memory = ReplayBuffer(
			self.mem_size,
			self.env.observation_space.shape
		)

	def get_name(self):
		string = ""

		string += self.activation_function + "_"

		for layer in self.hidden_layers:
			string += str(layer) + "_"

		string += str(self.learning_rate) + "_"
		string += str(self.epoch) + "_"
		string += str(self.gamma) + "_"
		string += str(self.init_epsilon) + "_"
		string += str(self.epsilon_dec) + "_"
		string += str(self.epsilon_end) + "_"
		string += str(self.mem_size) + "_"
		string += str(self.batch_size)

		return string
	
	def load(self):
		if os.path.exists(MODEL_PATH + self.get_name()):
			self.model = tf.keras.models.load_model(MODEL_PATH + self.get_name() + "/model.h5")
			print("Model already exists, skipping training...")
			self.training_done = True
		else:
			print("No similar model found, training...")

	def create_model(self):
		# On crée le modèle
		self.model = tf.keras.models.Sequential()

		# On utilise la taille de l'input de l'environnement comme input de notre modèle
		input_size = self.env.observation_space.shape[0]

		# On ajoute les couches cachées
		for hidden_layer in self.hidden_layers:
			self.model.add(tf.keras.layers.Dense(hidden_layer, activation="relu", input_shape=(input_size,)))

		# On ajoute la couche de sortie
		self.model.add(tf.keras.layers.Dense(self.env.action_space.n, activation=self.activation_function))

		# On compile le modèle
		self.model.compile(
			optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate),
			loss='mean_squared_error'
		)

	# On entraine le modèle pour qu'il apprenne à jouer
	def train(self):

		self.scores = []
		self.eps_history = []

		for i in range(self.current_epoch, self.epoch):
			done = False
			score = 0
			observation = self.env.reset()[0]

			# On joue jusqu'à ce que le jeu soit terminé
			j = 0
			while not done and j < 1000:
				# On choisit une action
				action = self.choose_action(observation)

				# On joue l'action
				new_observation, reward, done, _, _ = self.env.step(action)

				# On l'affiche si demandé
				if self.render_during_training:
					self.env.render()

				# On sauvegarde l'experience
				self.memory.store_transition(observation, action, reward, new_observation, done)

				# On met à jour l'observation
				observation = new_observation

				# On entraine le modèle
				self.learn()

				# On met à jour le score
				score += reward

				j += 1

			# On met à jour epsilon
			self.eps_history.append(self.epsilon) 

			# On enregistre le score
			self.scores.append(score)

			self.current_epoch += 1

			self.avg_score = np.mean(self.scores[-50:])

			# On plot 
			if self.live_plot:
				self.plot(PLOT_CURRENT)

			print('episode: ', i, 'score %.2f' % score, # TODO: Fonction de log
					'average_score %.2f' % self.avg_score,
					'epsilon %.2f' % self.epsilon)


		self.training_done = True

		# On supprime le plot current
		if self.live_plot and os.path.exists(PLOT_CURRENT):
			os.remove(PLOT_CURRENT)
		# On ferme l'environnement
		self.env.close()

	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			return

		states, actions, rewards, states_, dones = \
						self.memory.sample_buffer(self.batch_size)

		q_eval = self.model.predict(states)
		q_next = self.model.predict(states_)


		q_target = np.copy(q_eval)
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		q_target[batch_index, actions] = rewards + \
										self.gamma * np.max(q_next, axis=1)*dones


		self.model.train_on_batch(states, q_target)

		self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > \
						self.epsilon_end else self.epsilon_end


	def choose_action(self, observation, train=True):
		# On choisit une action
		if np.random.random() < self.epsilon and train:
			action = self.env.action_space.sample()
		else:
			state = np.array([observation])
			actions = self.model.predict(state)
			action = np.argmax(actions)

		return action
	
	def play(self):
		self.env = gym.make(self.env_name, render_mode="human")

		# Joue pour le plaisir
		while True:
			done = False
			score = 0
			observation = self.env.reset()[0]

			# On joue jusqu'à ce que le jeu soit terminé
			j = 0
			while not done and j < 1000:
				# On choisit une action
				action = self.choose_action(observation, train=False)

				# On joue l'action
				new_observation, reward, done, _, _ = self.env.step(action)

				# On met à jour l'observation
				observation = new_observation

				# On met à jour le score
				score += reward
				j += 1

			print("Score: ", score)


	def save(self):
		# On crée un dossier pour sauvegarder le modèle
		if not os.path.exists(MODEL_PATH):
			os.makedirs(MODEL_PATH)

		# On crée le dossier pour ce modèle
		if not os.path.exists(MODEL_PATH + self.get_name()):
			os.makedirs(MODEL_PATH + self.get_name())

		# On sauvegarde le modèle
		self.model.save(MODEL_PATH + self.get_name() + "/model.h5")

		self.plot(MODEL_PATH + self.get_name() + "/plot.png")

		# On crée un fichier pour les détails du modèle
		file = open(MODEL_PATH + self.get_name() + "/details.txt", "w")

		# On écrit les détails du modèle
		file.write("activation_function: " + str(self.activation_function) + "\n")
		file.write("hidden_layers: " + str(self.hidden_layers) + "\n")
		file.write("learning_rate: " + str(self.learning_rate) + "\n")
		file.write("epoch: " + str(self.epoch) + "\n")
		file.write("gamma: " + str(self.gamma) + "\n")
		file.write("init_epsilon: " + str(self.init_epsilon) + "\n")
		file.write("epsilon_dec: " + str(self.epsilon_dec) + "\n")
		file.write("epsilon_end: " + str(self.epsilon_end) + "\n")
		file.write("mem_size: " + str(self.mem_size) + "\n")
		file.write("batch_size: " + str(self.batch_size) + "\n")

	def plot(self, name):
			x = [i+1 for i in range(self.current_epoch)]
			plotLearning(x, self.scores, self.eps_history, name)


