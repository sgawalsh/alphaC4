import mc, engine
from tqdm import tqdm

def head2Head(champVal, champPol, challVal, challPol, showDownSize, trainingRecursionCount1, trainingRecursionCount2 = None):# play selected amount of games between two models, return win counts
	if trainingRecursionCount2 == None:
		trainingRecursionCount2 = trainingRecursionCount1
	challWins = 0
	drawCount = 0
	champWins = 0
	champFirst = False
	
	print("Showdown!!!")
	for _ in tqdm(range(showDownSize)):
		champFirst = not champFirst# toggle first move
		isRedTurn = champFirst
		currBoardState = engine.board()
		while True:
			currPlayer = mc.monteTree(currBoardState, isRedTurn, champPol, champVal) if isRedTurn else mc.monteTree(currBoardState, isRedTurn, challPol, challVal)
			for _2 in range(trainingRecursionCount1 if isRedTurn else trainingRecursionCount2):
				currPlayer.nnSelectRec(currPlayer.root)
			currBoardState, rowNum, colNum = currPlayer.makeMove()
			if currBoardState.checkWin(rowNum, colNum, isRedTurn):
				if not isRedTurn:
					print("\nChallenger wins!")
					challWins += 1
				else:
					print("\nChampion wins!")
					champWins += 1
				break
			elif currBoardState.checkDraw():
				print("\nDraw!")
				drawCount += 1
				break
			else:
				isRedTurn = not isRedTurn
		print('''	Current Stats:
		Champion Wins: {}
		Challenger Wins: {}
		Draws: {}'''.format(champWins, challWins, drawCount))
	print('''	End Stats:
		Games Played: {}
		Champion Wins: {}
		Challenger Wins: {}
		Draws: {}'''.format(showDownSize, champWins, challWins, drawCount))
		
	return champWins, challWins, drawCount