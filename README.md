# alphaC4

This project began as an attempt to train neural nets on connect 4, and over time morphed into an attempt to recreate the process used on the alpha go project and apply it to the game of connect 4. The project incorporates the use of multiple neural networks simultaneously, a monte carlo tree search function, and a self play learning process by which the user can train a neural net indefinitely.

By running the run.py file, a user can load an existing neural network, or can create their own via the terminal. The selfplay learning process is then started automatically until the terminal is closed. Any new champions created during this time will be saved under the folder name selected by the user.

A user can check the results of the selfplay learning by running a tournament between the current and previous saved champions by running seflplay.generationTournament(). The results of the tournament can be saved and subsequently accessed by running selfplay.loadTournamentResults().