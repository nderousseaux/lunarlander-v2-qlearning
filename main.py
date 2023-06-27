import argparse
from termcolor import colored

# Default arguments
ENV_NAME = "LunarLander-v2"
HIDDEN_LAYERS=[150, 120]
LEARNING_RATE=.001
EPOCH=30000
GAMMA=.99
EPSILON=1.0
EPSILON_DEC=.996
EPSILON_END=0.01
MEM_SIZE=1000000
BATCH_SIZE=64
ACTIVATION_FUNCTION = "linear"
RENDER_DURING_TRAINING = False
LIVE_PLOT = True


print("\nLoading Tensorflow...", end="\r")
from src.QAgent import QAgent
print(f"Loading Tensorflow... {colored('Done', 'green')}\n")

def load_arguments():

  # On prent les arguments
  parser = argparse.ArgumentParser(description='Q-Learning')
  parser.add_argument('--hidden_layers', type=list, default=HIDDEN_LAYERS, help='Hidden layers')
  parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
  parser.add_argument('--epoch', type=int, default=EPOCH, help='Epoch')
  parser.add_argument('--gamma', type=float, default=GAMMA, help='Gamma')
  parser.add_argument('--epsilon', type=float, default=EPSILON, help='Epsilon')
  parser.add_argument('--epsilon_dec', type=float, default=EPSILON_DEC, help='Epsilon dec')
  parser.add_argument('--epsilon_end', type=float, default=EPSILON_END, help='Epsilon end')
  parser.add_argument('--mem_size', type=int, default=MEM_SIZE, help='Memory size')
  parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
  parser.add_argument('--activation_function', type=str, default=ACTIVATION_FUNCTION, help='Activation function')
  parser.add_argument('--render_during_training', type=bool, default=RENDER_DURING_TRAINING, help='Render during training')
  parser.add_argument('--live_plot', type=bool, default=LIVE_PLOT, help='Live plot')

 
  args = parser.parse_args()

  return ENV_NAME, args.hidden_layers, args.learning_rate, args.epoch, args.gamma, args.epsilon, args.epsilon_dec, args.epsilon_end, args.mem_size, args.batch_size, args.activation_function, args.render_during_training, args.live_plot

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

