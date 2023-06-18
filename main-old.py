import os
import sys

from dotenv import load_dotenv

from src.Demo import Demo
from src.Agent import QAgent

MODEL_FOLDER = "models"
RANDOM_MODEL = "random"

def show_help(): # TODO : Faire un help avec les bons arguments
    str = """
    Usage: \"python main.py\" -> use env variables
    Usage: \"python main.py [model_name] [hidden_layers]\" -> for bypass env variables

    model_name: Name of the model to create. If the model already exists, it will be use for the demo. \"Random\" is a special model that will generate random play
    hidden_layers: List of hidden layers. Example: \"128 64 32\"
    """

    print(str)
    exit()

def load_env():
    load_dotenv()
    model_name = os.getenv("MODEL_NAME")
    hidden_layers = os.getenv("HIDDEN_LAYERS")
    hidden_layers = [int(hidden_layer) for hidden_layer in hidden_layers.split(" ")]
   
    return model_name, hidden_layers


def create_model(model_name, hidden_layers):

    # Si le modèle est random, on saute la création du modèle
    if model_name == RANDOM_MODEL:
        print("Random model, skipping model creation...")

    # Sinon, on regarde dans le dossier models si le modèle est un nouveau modèle
    elif not os.path.exists(f"{MODEL_FOLDER}/{model_name}"):
        print(f"Creating model {model_name}...")
        QAgent(model_name, hidden_layers)
    
    # Si oui, on skippe la création du modèle
    else:
        print(f"Model {model_name} already exists, skipping model creation...")

if __name__ == "__main__":

    # On récupère les arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "-h" or sys.argv[1] == "--help":
            show_help()

        model_name, hidden_layers = sys.argv[1], sys.argv[2:]
    else:
        model_name, hidden_layers = load_env()
    
    # On crée un nouveau modèle, si il existe pas déjà. Si il existe, on le charge
    model = create_model(model_name, hidden_layers)

    # On joue des parties de démonstration
    if model_name == RANDOM_MODEL:
        random = True
    else:
        random = False
    Demo(model_name, random=random)