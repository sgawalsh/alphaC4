import tensorflow as tf, mc, config, engine, pdb, random, _pickle, os, NNfunctions, numpy
from tqdm import tqdm
from keras.utils import np_utils

class gameSet():
	def __init__(self, gameID = 0):
		self.gameID = gameID
		self.gameBoards = []
		self.hasWinner = False
		self.redWins = False
		
def createTrainingSet(firstRun = False):
	#create config.trainingSetSize games by champion using MCTS against self, for each board store move probs, and eventual winner, append data to library, deleting old data
	if firstRun:
		gameID = 0
		trainingSet = []
	else:
		trainingSet = _pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + "\\trainingData\\trainingSet", "rb"))
		gameID = trainingSet[len(trainingSet) - 1].gameID
		gameID = gameID if gameID < 10000 else 1
	
	currSet = gameSet(gameID + 1)
	champVal = tf.keras.models.load_model("models/value3cat/simple/the_value_champ")
	champPol = tf.keras.models.load_model("models/policy/simple/the_simple_champ")
	
	print("Creating new training samples...")
	
	for _ in tqdm(range(config.trainingSetSize)):
		myTree = mc.monteTree(engine.board(), random.choice([True, False]), champPol, champVal)
		while True:
			boardMovePair = [myTree.root.board, myTree.root.isRedTurn]
			for _2 in range(config.trainingRecursionCount):#build tree
				myTree.nnSelectRec(myTree.root)
			resBoard = myTree.makeMove()[0]# return next board
			#myTree.root.board.printBoard()
			if resBoard.checkWin(myTree.root.rowNum, myTree.root.colNum, not myTree.root.isRedTurn):#check winner
				currSet.hasWinner = True
				currSet.redWins = not myTree.root.isRedTurn
				trainingSet.append(currSet)
				currSet = gameSet(currSet.gameID + 1)
				break
			elif resBoard.checkDraw():
				trainingSet.append[currSet]
				currSet = gameSet(currSet.gameID + 1)
				break
			else:
				boardMovePair.append(myTree.root.colNum)
				currSet.gameBoards.append(boardMovePair)
	#slice oldest games, pickle training set
	_pickle.dump(trainingSet[-config.fullTrainingSetSize:], open(os.path.dirname(os.path.realpath(__file__)) + "/trainingData/trainingSet", "wb"))
	
	print("Training samples updated.")

def createChallengerPair():
	#use sample board positions from library, use mcts to train current champion NN to create challenger NN and VNN
	trainingSet = _pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + "\\trainingData\\trainingSet", "rb"))
	boardList = []
	boardMoves = []
	boardResults = []
	
	print("Creating new challenger...")
	
	for _ in tqdm(range(config.challengerSamples)):
		set = random.choice(trainingSet)
		board = random.choice(set.gameBoards)
		boardList.append([board[0].board, board[1]])
		boardMoves.append(board[2])
		if set.hasWinner:
			if set.redWins == board[1]:#win
				boardResults.append(1)
			else:
				boardResults.append(0)
		else:#draw
			boardResults.append(.5)
	boardList = numpy.array(NNfunctions.boardArrayListToInputs(boardList))
	trainVal = tf.keras.models.load_model("models/value3cat/simple/the_value_champ")
	trainPol = tf.keras.models.load_model("models/policy/simple/the_simple_champ")

	trainPol.fit(boardList, np_utils.to_categorical(boardMoves, 7), epochs = 3, batch_size = 500)
	trainVal.fit(boardList, np_utils.to_categorical(boardResults, 3), epochs = 3, batch_size = 500)
	
	trainVal.save("models//value3cat//simple//the_value_challenger")
	trainPol.save("models//policy//simple//the_simple_challenger")
	print("Challenger ready and primed!")
	
def modelShowdown():
	#play config.showDownSize games using mcts, if challenger wins 55% challenger becomes new champion
	champVal = tf.keras.models.load_model("models/value3cat/simple/the_value_champ")
	challVal = tf.keras.models.load_model("models/value3cat/simple/the_value_challenger")
	champPol = tf.keras.models.load_model("models/policy/simple/the_simple_champ")
	challPol = tf.keras.models.load_model("models/policy/simple/the_simple_challenger")
	champFirst = False
	challWins = 0
	drawCount = 0
	champWins = 0
		
	print("Showdown!!!")
	for _ in tqdm(range(config.showDownSize)):
		champFirst = not champFirst# toggle first move
		testMode = False
		isRedTurn = champFirst
		currBoardState = engine.board()
		while True:
			currPlayer = mc.monteTree(currBoardState, isRedTurn, champPol, champVal) if isRedTurn else mc.monteTree(currBoardState, isRedTurn, challPol, challVal)
			for _2 in range(config.trainingRecursionCount):
				currPlayer.nnSelectRec(currPlayer.root)
				#currPlayer.root.board.printBoard()
				#print(currPlayer.root.__str__(15))
				if testMode:
					print(currPlayer.root.__str__(15))
					pdb.set_trace()
			currBoardState, rowNum, colNum = currPlayer.makeMove()
			currBoardState.printBoard()
			#print(currPlayer.root.__str__(0,2))
			pdb.set_trace()
			if currBoardState.checkWin(rowNum, colNum, isRedTurn):
				if not isRedTurn:
					print("Challenger wins!")
					challWins += 1
				else:
					print("Champion wins!")
					champWins += 1
				break
			elif currBoardState.checkDraw():
				print("Draw!")
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
		Draws: {}'''.format(config.showDownSize, champWins, challWins, drawCount))
	pdb.set_trace()
	if challWins > (champWins * config.winRatio):
		print("New champion!!")
		challVal.save("models//value3cat//simple//the_value_champ")
		challPol.save("models//policy//simple//the_simple_champ")
			
			
while True:
	#createTrainingSet()
	#createChallengerPair()
	modelShowdown()