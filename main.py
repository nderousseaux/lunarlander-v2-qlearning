import argparse
from termcolor import colored

print("\nLoading Tensorflow...", end="\r")
from src.QAgent import QAgent
print(f"Loading Tensorflow... {colored('Done', 'green')}\n")

def load_arguments():
  # Default arguments
  env_name = "LunarLander-v2"
  hidden_layers = [150, 120]
  learning_rate=.001
  epoch=2
  gamma=.99
  epsilon=1.0
  epsilon_dec=.996
  epsilon_end=0.01
  mem_size=1000000
  batch_size=64
  activation_function = "linear"
  render_during_training = False
  live_plot = True

  # On prent les arguments
  parser = argparse.ArgumentParser(description='Q-Learning')
  parser.add_argument('--env_name', type=str, default=env_name, help='Environment name')
  parser.add_argument('--hidden_layers', type=list, default=hidden_layers, help='Hidden layers')
  parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate')
  parser.add_argument('--epoch', type=int, default=epoch, help='Epoch')
  parser.add_argument('--gamma', type=float, default=gamma, help='Gamma')
  parser.add_argument('--epsilon', type=float, default=epsilon, help='Epsilon')
  parser.add_argument('--epsilon_dec', type=float, default=epsilon_dec, help='Epsilon dec')
  parser.add_argument('--epsilon_end', type=float, default=epsilon_end, help='Epsilon end')
  parser.add_argument('--mem_size', type=int, default=mem_size, help='Memory size')
  parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
  parser.add_argument('--activation_function', type=str, default=activation_function, help='Activation function')
  parser.add_argument('--render_during_training', type=bool, default=render_during_training, help='Render during training')
  parser.add_argument('--live_plot', type=bool, default=live_plot, help='Live plot')

 
  args = parser.parse_args()

  return args.env_name, args.hidden_layers, args.learning_rate, args.epoch, args.gamma, args.epsilon, args.epsilon_dec, args.epsilon_end, args.mem_size, args.batch_size, args.activation_function, args.render_during_training, args.live_plot

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

