# alphaC4

This project began as an attempt to apply neural nets to the game of Connect 4, and over time morphed into an attempt to recreate the techniques used on the AlphaGo projects and apply them to Connect 4. The project approximately follows the techniques described in this [article](https://web.stanford.edu/~surag/posts/alphazero.html) and this [image](https://miro.medium.com/max/4000/1*0pn33bETjYOimWjlqDLLNw.png).

The agent uses a monte carlo tree search, which is driven by two seperate neural nets evaluating the favourability, as well as the move probabilities of a given board. After a user-selected amount of iterations of the monte carlo tree search, the move is selected either deterministically or stochastically according to variables contained in `config.py`. This config file contains various other variables which determine certain aspects of the agent behaviour including the use of temperatures when performing selfplay in order to ensure the agent explores possible game states.

By running the `run.py` file, the user is presented with 4 options.

1) The user can load an existing neural network, or can create their own via the terminal. The selfplay learning process is then started automatically until the terminal is closed. Any new champions created during this time will be saved under the folder name selected by the user.

2) The user can check the results of the selfplay learning by running a tournament between the current and previous saved champions by running seflplay.generationTournament().

3) The results of the tournament can be saved and subsequently accessed by running selfplay.loadTournamentResults().

4) The project also allows the user to compare the performance of different neural net structures by running a tournament between two existing neural net pairs using the head2Head.modelShowdown function.
