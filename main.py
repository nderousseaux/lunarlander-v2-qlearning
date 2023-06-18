from src.QAgent import QAgent

def load_arguments():
  env_name = "LunarLander-v2"
  hidden_layers = [125, 100]
  learning_rate=.001
  epoch=500
  gamma=.99
  epsilon=0.9
  epsilon_dec=.995
  epsilon_end=0
  mem_size=1000000
  batch_size=64
  activation_function = "softmax"
  render_during_training = False
  live_plot = True

  return env_name, hidden_layers, learning_rate, epoch, gamma, epsilon, epsilon_dec, epsilon_end, mem_size, batch_size, activation_function, render_during_training, live_plot

if __name__ == "__main__":
  
  # On recupere les arguments
  env_name, hidden_layers, learning_rate, epoch, gamma, epsilon, epsilon_dec, epsilon_end, mem_size, batch_size, activation_function, render_during_training, live_plot = load_arguments()

  # On crée un modèle avec les arguments
  model = QAgent(
    env_name,
    hidden_layers,
    learning_rate,
    epoch,
    gamma,
    epsilon,
    epsilon_dec,
    epsilon_end,
    mem_size,
    batch_size,
    activation_function,
    render_during_training,
    live_plot
  )

  if not model.training_done:  
    # On entraine le modèle
    model.train()

    # On sauvegarde le modèle
    model.save()

  # On fait jouer le modèle
  model.play()

