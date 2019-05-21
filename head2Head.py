import mc, engine, config, pdb, os, tensorflow as tf
from tqdm import tqdm

def head2Head(isSelfPlayShowDown, name1, name2, champVal, champPol, challVal, challPol, showDownSize, trainingRecursionCount1, trainingRecursionCount2 = None, custConst1 = None, custConst2 = None): # play selected amount of games between two models, return win counts
	if not custConst1 and not custConst2:
		custConst1, custConst2 = config.MCTSexploration, config.MCTSexploration
	elif custConst1 and not custConst2:
		custConst2 = custConst1
	if trainingRecursionCount2 == None:
		trainingRecursionCount2 = trainingRecursionCount1
	challWins = 0
	drawCount = 0
	champWins = 0
	# showTree = False
	# showBoard = True
	# treeDepth = 2
	
	print("Showdown!!!")
	for gameNum in tqdm(range(showDownSize)):
		isRedTurn = gameNum % 2 == 0 # toggle first move
		currBoardState = engine.board()
		turnCount = 0
		while True:
			currPlayer = mc.monteTree(currBoardState, True, champPol, champVal) if isRedTurn else mc.monteTree(currBoardState, False, challPol, challVal)
			for _2 in range(trainingRecursionCount1 if isRedTurn else trainingRecursionCount2):
				currPlayer.nnSelectRec(currPlayer.root, custConstMC = custConst1 if isRedTurn else custConst2)
			# if showTree:
				# print(currPlayer.__str__(treeDepth))
			temp = mc.monteTree.turnCountToTemp(turnCount)
			if temp < 1:
				currBoardState, rowNum, colNum = currPlayer.exploratoryMove(temp)
			else:
				currBoardState, rowNum, colNum = currPlayer.makeMove()
			# if showBoard:
			#currBoardState.printBoard()
			# pdb.set_trace()
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
				turnCount += 1
		print('''	Current Stats:
		{} Wins: {}
		{} Wins: {}
		Draws: {}'''.format(name1, champWins, name2, challWins, drawCount))
		if isSelfPlayShowDown:
			if ((champWins) * config.winRatio) > (showDownSize - champWins - drawCount):
				print("Challenger victory no longer possible. Ending showdown.")
				break
			elif ((challWins) / config.winRatio) > (showDownSize - challWins - drawCount):
				print("Challenger wins! Ending showdown.")
				break
	print('''	End Stats:
		Games Played: {}
		{} Wins: {}
		{} Wins: {}
		Draws: {}'''.format(champWins + challWins + drawCount, name1, champWins, name2, challWins, drawCount))
		
	return champWins, challWins, drawCount
	
def modelFaceoff():
	folderList = os.listdir(os.path.dirname(os.path.realpath(__file__)) + "\\models")
	if folderList:
		print("Models:")
		
		_ = 0
		modelArray = []
		while _ < 2:# load 2 models
			while True:
				for i, folder in enumerate(folderList):
					print(str(i + 1) + ". " + folder)
				try:
					modelInd = int(input(("Which model do you want to load for player {}?\n> ".format(_ + 1))))
					if modelInd > len(folderList) or modelInd < 1:
						print("Value out of range")
					else:
						break
				except ValueError:
					print("Which model do you want to load?\n> ")
					
			if modelInd:
				specList = os.listdir(os.path.dirname(os.path.realpath(__file__)) + "\\models\\" + folderList[modelInd - 1])
				while True:
					for i, folder in enumerate(specList):
						print(str(i + 1) + ". " + folder)
					try:
						specInd = int(input(("Which model specs do you want to load?\n> ")))
						if specInd > len(specList) or specInd <= 0:
							print("Value out of range")
						else:
							specArray = specList[specInd - 1].split("_")
							const = int(specArray[3])
							if specArray[4]:
								const = float(const + float("." + specArray[4]))
							break
					except ValueError:
						print("Which model specs do you want to load?\n> ")
				if modelInd:
					modelName = os.path.dirname(os.path.realpath(__file__)) + "\\models\\" + folderList[modelInd - 1] + "\\"+ specList[specInd - 1]
					pdb.set_trace()
					modelArray.append([tf.keras.models.load_model(modelName + "\\the_value_champ"), tf.keras.models.load_model(modelName + "\\the_policy_champ"), config.getIntResponse("What should the recursion count be?"), folderList[modelInd - 1], const if str(const)[:9] != str(config.MCTSexploration)[:9] else None])
					_ += 1
			else: 
				continue
		head2Head(False, modelArray[0][3], modelArray[1][3], modelArray[0][0], modelArray[0][1], modelArray[1][0], modelArray[1][1], config.getIntResponse("How many games should be played?"), modelArray[0][2], modelArray[1][2], modelArray[0][4], modelArray[1][4])
	else:
		print("No models exist")