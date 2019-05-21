import selfPlay, head2Head
choices = ["Run selfplay learning with a new or existing Neural Network", "Run a Generation Tournament on an existing Neural Network", "Load results from a previously run Generation Tournament", "Run a tournament between two models."]

print("What would you like to do")

for i, folder in enumerate(choices):
	print(str(i + 1) + ". " + folder)
	
while True:
	try:
		choiceNum = int(input(("Enter your choice number.\n> ")))
		if choiceNum > len(choices) or choiceNum < 1:
			print("Value out of range")
		else:
			break
	except ValueError:
		print("Enter your choice number.\n> ")

if choiceNum == 1:
	selfPlay.autoSelfplay()
elif choiceNum == 2:
	selfPlay.generationTournament()
elif choiceNum == 3:
	selfPlay.loadTournamentResults()
elif choiceNum == 4:
	head2Head.modelFaceoff()

input("Press enter to close the program.\n")