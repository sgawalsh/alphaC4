import mc, engine, config, pdb
from tqdm import tqdm

def head2Head(isSelfPlayShowDown, name1, name2, champVal, champPol, challVal, challPol, showDownSize, trainingRecursionCount1, trainingRecursionCount2 = None): # play selected amount of games between two models, return win counts
	if trainingRecursionCount2 == None:
		trainingRecursionCount2 = trainingRecursionCount1
	challWins = 0
	drawCount = 0
	champWins = 0
	champFirst = False
	
	print("Showdown!!!")
	for _ in tqdm(range(showDownSize)):
		champFirst = not champFirst # toggle first move
		isRedTurn = champFirst
		currBoardState = engine.board()
		while True:
			currPlayer = mc.monteTree(currBoardState, True, champPol, champVal) if isRedTurn else mc.monteTree(currBoardState, False, challPol, challVal)
			for _2 in range(trainingRecursionCount1 if isRedTurn else trainingRecursionCount2):
				currPlayer.nnSelectRec(currPlayer.root)
			#print(currPlayer)
			currBoardState, rowNum, colNum = currPlayer.makeMove()
			currBoardState.printBoard()
			if currBoardState.checkWin(rowNum, colNum, isRedTurn):
				if isRedTurn:
					print("\n" + name1 + " wins!")
					champWins += 1
				else:
					print("\n" + name2 + " wins!")
					challWins += 1
				break
			elif currBoardState.checkDraw():
				print("\nDraw!")
				drawCount += 1
				break
			else:
				isRedTurn = not isRedTurn
		print('''	Current Stats:
		{} Wins: {}
		{} Wins: {}
		Draws: {}'''.format(name1, champWins, name2, challWins, drawCount))
		if isSelfPlayShowDown and ((champWins) * config.winRatio) > (showDownSize - champWins - drawCount):
			print("Challenger victory no longer possible. Ending showdown.")
			break
	print('''	End Stats:
		Games Played: {}
		{} Wins: {}
		{} Wins: {}
		Draws: {}'''.format(champWins + challWins + drawCount, name1, champWins, name2, challWins, drawCount))
		
	return champWins, challWins, drawCount