import datetime
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import gym

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0", label="Epsilon")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1", label="Mean of last 20 games")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    # Plot la moyenne de 50 derniers scores
    N = len(scores)
    running_avg50 = np.empty(N)
    for t in range(N):
	    running_avg50[t] = np.mean(scores[max(0, t-50):(t+1)])
    ax2.plot(x, running_avg50, color="C2", label="Mean of last 50 games")

    # Plot la moyenne de 100 derniers scores
    N = len(scores)
    running_avg100 = np.empty(N)
    for t in range(N):
	    running_avg100[t] = np.mean(scores[max(0, t-100):(t+1)])
    ax2.plot(x, running_avg100, color="C3", label="Mean of last 100 games")

    # Legende des 4 courbes à l'extérieur du graphique
    ax2.legend(loc="upper left", bbox_to_anchor=(0, 1.1))
    

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    plt.close()

# Fonction qui renvoie un string, complété par des espaces, de la taille de nb_car
def fill_line(string, nb_car):
	space_start = 4
	return (
		"│" +
		" " * space_start
		+ string
		+ " " * (nb_car - real_len(string) - space_start - 2)
		+ "│"
	)

# Fonction qui renvoie la longueur réelle d'un string (sans les caractères d'échappement)
def real_len(string):	
	return len(re.sub(r"\x1b\[[0-9;]*m", "", string))

# Fonction qui retourne une barre de chargement
def loading(i, i_max, size):
	size -= 5

	nb = int((i / i_max) * size)
	return "[" + "#" * nb + "-" * (size - nb) + "]" + f" {round((i / i_max) * 100, 2)}%"
