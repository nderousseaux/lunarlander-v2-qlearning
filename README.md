# Lunar Lander Q-Learning attempt

This is my attempt to solve the Lunar Lander problem using Q-Learning.

## How to use this code

All the parameters can be indicated on the command line like this:
 
```bash
python main.py --hidden_layers="[64, 64]" --learning_rate=0.001 --epoch=1000 --gamma=0.99 --epsilon=1.0 --epsilon_dec=0.999 --epsilon_end=0.01 --mem_size=100000 --batch_size=64 --activation_function="relu" --render_during_training=True --live_plot=True
```

If you not indicate any parameter, the default value will be used. All the defaults the parameters is in the top of the main.py file.

If there is already a model saved, (on the folder `models`), the program will skip the training and will play the game using the model.