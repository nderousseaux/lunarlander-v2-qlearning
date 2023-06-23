import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import signal

from termcolor import colored
from reprint import output
import gym
import numpy as np
import tensorflow as tf

from src.utils import *


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
		# tf.compat.v1.enable_eager_execution()
		self.env_name = env_name
		if render_during_training:
			self.env = gym.make(env_name, render_mode="human")
		else:
			self.env = gym.make(env_name)
		self.hidden_layers = hidden_layers
		self.learning_rate = learning_rate
		self.epoch = epoch
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

		# On logue le modèle
		self.log_model()

		# On crée le modèle
		self.create_model()

		# Load model if exists
		self.load()

		# On crée la mémoire
		self.memory = ReplayBuffer(
			self.mem_size,
			self.env.observation_space.shape
		)

	# On crée le modèle
	def create_model(self):

		initializer = tf.keras.initializers.GlorotUniform(seed=42)

		next_layer = self.hidden_layers[0] if len(self.hidden_layers) > 0 else self.env.action_space.n

		self.model = tf.keras.models.Sequential([
			tf.keras.layers.Dense(
				next_layer,
				input_dim=self.env.observation_space.shape[0],
				activation="relu",
				kernel_initializer=initializer,
				bias_initializer=initializer
			)
		])

		for layer_size in self.hidden_layers[1:]:
			self.model.add(tf.keras.layers.Dense(
				layer_size,
				activation="relu",
				kernel_initializer=initializer,
				bias_initializer=initializer
			))

		self.model.add(tf.keras.layers.Dense(
			self.env.action_space.n,
			activation=self.activation_function,
			kernel_initializer=initializer,
			bias_initializer=initializer
		))

		self.model.compile(
			optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate),
			loss="mean_squared_error"
		)

	# On entraine le modèle pour qu'il apprenne à jouer
	def train(self):
		self.scores = []
		self.eps_history = []

		# Ctrl+C
		interrupt = False

		# Log
		self.start_time = time.time()
		with output(output_type='list', initial_len=16) as self.output:
			try:
				for self.current_epoch in range(self.epoch):
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

					# On plot 
					if self.live_plot:
						self.plot(PLOT_CURRENT)

					self.log_train()
			except KeyboardInterrupt:
				interrupt = True

			finally:
				self.training_done = True

				# On supprime le plot current
				if self.live_plot and os.path.exists(PLOT_CURRENT):
					os.remove(PLOT_CURRENT)
				# On ferme l'environnement
				self.env.close()

		if interrupt:
			exit(0)
		
		print("\nTraining done, start testing...")
		print(f"Info and perf is available at :\n{os.path.join(os.getcwd(), MODEL_PATH, self.get_name())}\n")

	# On fait apprendre le modèle
	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			return

		states, actions, rewards, states_, dones = \
						self.memory.sample_buffer(self.batch_size)

		q_eval = self.model.predict(states, verbose=0)
		q_next = self.model.predict(states_, verbose=0)


		q_target = np.copy(q_eval)
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		q_target[batch_index, actions] = rewards + \
										self.gamma * np.max(q_next, axis=1)*dones


		self.model.train_on_batch(states, q_target)

		self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > \
						self.epsilon_end else self.epsilon_end

	# On choisit une action
	def choose_action(self, observation, train=True):
		# On choisit une action
		if np.random.random() < self.epsilon and train:
			action = self.env.action_space.sample()
		else:
			state = np.array([observation])
			# On desactive les logs
			actions = self.model.predict(state, verbose=0)
			action = np.argmax(actions)

		return action
	
	# On fait jouer le modèle
	def play(self):
		self.env = gym.make(self.env_name, render_mode="human")

		self.scores_play = []
		self.number_of_games = 0

		with output(output_type='list', initial_len=16) as self.output:
			try:
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

					# On enregistre le score
					self.scores_play.append(score)
					self.number_of_games += 1

					self.log_play()
			except KeyboardInterrupt:
				exit(0)


	### Fonctions utiles ###

	# Choisi un nom pour le modèle
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
	
	# Charge un modèle
	def load(self):
		if os.path.exists(MODEL_PATH + self.get_name()):
			self.model = tf.keras.models.load_model(MODEL_PATH + self.get_name() + "/model.h5")
			print("\nModel already exists, skipping training...\n")
			self.training_done = True
		else:
			print("\nNo similar model found, start training...")
			if self.live_plot:
				# print ("Live plot will be available at : " + os.join(os.getcwd(), PLOT_CURRENT) + "\n")
				print("Live plot is available at : " + os.path.join(os.getcwd(), PLOT_CURRENT) + "\n")
			else:
				print("")

	# Sauvegarde le modèle
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

	# Plot les résultats
	def plot(self, name):
			x = [i+1 for i in range(self.current_epoch)]
			plotLearning(x, self.scores, self.eps_history, name)

	# On logue le modèle
	def log_model(self):
		print("╭───────────────────────────────── " + colored("PARAMETERS", "blue") + " ─────────────────────────────────╮")
		print(fill_line("", 80))
		print(fill_line(f"Name:           {colored(self.get_name(), 'blue')}", 80))
		print(fill_line(f"Activ. func.:   {colored(self.activation_function, 'blue')}", 80))
		print(fill_line(f"Hidden layers:  {colored(self.hidden_layers, 'blue')}", 80))
		print(fill_line(f"Learning rate:  {colored(self.learning_rate, 'blue')}", 80))
		print(fill_line(f"Epoch:          {colored(self.epoch, 'blue')}", 80))
		print(fill_line(f"Gamma:          {colored(self.gamma, 'blue')}", 80))
		print(fill_line(f"Init epsilon:   {colored(self.init_epsilon, 'blue')}", 80))
		print(fill_line(f"Epsilon dec:    {colored(self.epsilon_dec, 'blue')}", 80))
		print(fill_line(f"Epsilon end:    {colored(self.epsilon_end, 'blue')}", 80))
		print(fill_line(f"Memory size:    {colored(self.mem_size, 'blue')}", 80))
		print(fill_line(f"Batch size:     {colored(self.batch_size, 'blue')}", 80))
		print(fill_line("", 80))
		print("╰──────────────────────────────────────────────────────────────────────────────╯")

	# Logue la phase d'entrainement
	def log_train(self):
		n2 = self.scores[-2] if len(self.scores) > 1 else "None"
		n3 = self.scores[-3] if len(self.scores) > 2 else "None"
		n4 = self.scores[-4] if len(self.scores) > 3 else "None"

		time_elapsed = time.time() - self.start_time
		str_time = str(datetime.timedelta(seconds=time_elapsed)).split(".")[0]

		self.output[0] = "╭──────────────────────────────────── " + colored("LEARN", "red") + " ───────────────────────────────────╮"
		self.output[1] = fill_line("", 80)
		self.output[2] = fill_line(f"Time elapsed:          {colored(str_time, 'red')}", 80)
		self.output[3] = fill_line(f"Epoch:                 {colored(f'{self.current_epoch}/{self.epoch}', 'red')}", 80)
		self.output[4] = fill_line("", 80)
		self.output[5] = fill_line(loading(self.current_epoch, self.epoch, 60), 80)
		self.output[6] = fill_line("", 80)
		self.output[7] = fill_line(f"Current Epsilon:       {colored(round(self.epsilon, 2),'red')}", 80)
		self.output[8] = fill_line(f"Avg. Score (50):       {colored(round(np.mean(self.scores[-50:]),2), 'red')}", 80)
		self.output[9] = fill_line(f"Avg. Score (100):      {colored(round(np.mean(self.scores[-100:]), 2), 'red')}", 80)
		self.output[10] = fill_line(f"Score to n-1 game:     {colored(round(self.scores[-1], 2), 'red')}", 80)
		self.output[11] = fill_line(f"Score to n-2 game:     {colored(round(n2, 2) if n2 != 'None' else 'None', 'red')}", 80)
		self.output[12] = fill_line(f"Score to n-3 game:     {colored(round(n3, 2) if n3 != 'None' else 'None', 'red')}", 80)
		self.output[13] = fill_line(f"Score to n-4 game:     {colored(round(n4, 2) if n4 != 'None' else 'None', 'red')}", 80)
		self.output[14] = fill_line("", 80)
		self.output[15] = "╰──────────────────────────────────────────────────────────────────────────────╯"

	# Logue la phase de jeu
	def log_play(self):
		n2 = self.scores_play[-2] if len(self.scores_play) > 1 else "None"
		n3 = self.scores_play[-3] if len(self.scores_play) > 2 else "None"
		n4 = self.scores_play[-4] if len(self.scores_play) > 3 else "None"
		n5 = self.scores_play[-5] if len(self.scores_play) > 4 else "None"
		n6 = self.scores_play[-6] if len(self.scores_play) > 5 else "None"
		n7 = self.scores_play[-7] if len(self.scores_play) > 6 else "None"
		n8 = self.scores_play[-8] if len(self.scores_play) > 7 else "None"
		n9 = self.scores_play[-9] if len(self.scores_play) > 8 else "None"


		self.output[0] = "╭────────────────────────────────────  " + colored("PLAY", "green") + " ───────────────────────────────────╮"
		self.output[1] = fill_line("", 80)
		self.output[2] = fill_line(f"Game played:           {colored(self.number_of_games, 'green')}", 80)
		self.output[3] = fill_line(f"Avg. Score (50):       {colored(round(np.mean(self.scores_play[-50:]),2), 'green')}", 80)
		self.output[4] = fill_line(f"Avg. Score (100):      {colored(round(np.mean(self.scores_play[-100:]), 2), 'green')}", 80)
		self.output[5] = fill_line(f"Score to n-1 game:     {colored(round(self.scores_play[-1], 2), 'green')}", 80)
		self.output[6] = fill_line(f"Score to n-2 game:     {colored(round(n2, 2) if n2 != 'None' else 'None', 'green')}", 80)
		self.output[7] = fill_line(f"Score to n-3 game:     {colored(round(n3, 2) if n3 != 'None' else 'None', 'green')}", 80)
		self.output[8] = fill_line(f"Score to n-4 game:     {colored(round(n4, 2) if n4 != 'None' else 'None', 'green')}", 80)
		self.output[9] = fill_line(f"Score to n-5 game:     {colored(round(n5, 2) if n5 != 'None' else 'None', 'green')}", 80)
		self.output[10] = fill_line(f"Score to n-6 game:     {colored(round(n6, 2) if n6 != 'None' else 'None', 'green')}", 80)
		self.output[11] = fill_line(f"Score to n-7 game:     {colored(round(n7, 2) if n7 != 'None' else 'None', 'green')}", 80)
		self.output[12] = fill_line(f"Score to n-8 game:     {colored(round(n8, 2) if n8 != 'None' else 'None', 'green')}", 80)
		self.output[13] = fill_line(f"Score to n-9 game:     {colored(round(n9, 2) if n9 != 'None' else 'None', 'green')}", 80)
		self.output[14] = fill_line("", 80)
		self.output[15] = "╰──────────────────────────────────────────────────────────────────────────────╯"