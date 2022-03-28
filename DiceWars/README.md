Implementation of agent playing DiceWars game.

There are two agents implemented.

Ferda is naive with reacitve architecture.

Kokos works as follows:

There is a neural netowrk (Graph convolution neural network) for Q-value estimation from the state of game, there is also some kind of BFS with pruning based on estimated heuristicf by the neural network. The whole code is focused on speed, so there might be present some "ugly" pieces of code.

The neural network and it's necessary tools are in estimator.py

For working implemantation of this agent and it's environmet, see https://gitlab.com/michal.glos99/sui-projekt