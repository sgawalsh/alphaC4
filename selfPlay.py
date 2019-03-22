import tensorflow as tf, mc, config, engine, pdb, random, _pickle, os, NNfunctions, numpy
from tqdm import tqdm
from keras.utils import np_utils

#tf.enable_eager_execution()

class gameSet():
	def __init__(self, gameID = 0):
		self.gameID = gameID
		self.gameBoards = []
		self.hasWinner = False
		self.redWins = False
		
def createTrainingSet(modelName, firstRun = False):
	#create config.trainingSetSize games by champion using MCTS against self, for each board store move probs, and eventual winner, append data to library, deleting old data
	if firstRun:
		gameID = 0
		trainingSet = []
	else:
		trainingSet = _pickle.load(open(modelName + "\\trainingData\\trainingSet", "rb"))
		gameID = trainingSet[len(trainingSet) - 1].gameID
		gameID = gameID if gameID < 10000 else 0
	
	currSet = gameSet(gameID + 1)
	champVal = tf.keras.models.load_model(modelName + "\\the_value_champ")
	champPol = tf.keras.models.load_model(modelName + "\\the_policy_champ")
	
	print("Creating new training samples...")
	
	for _ in tqdm(range((int(config.challengerSamples / 5)) if firstRun else config.trainingSetSize)):
		myTree = mc.monteTree(engine.board(), random.choice([True, False]), champPol, champVal)
		while True:
			boardMovePair = [myTree.root.board, myTree.root.isRedTurn]
			for _2 in range(config.trainingRecursionCount):#build tree
				myTree.nnSelectRec(myTree.root)
			myTree.makeMove(True)# set root to next move
			#myTree.root.board.printBoard()
			if myTree.root.board.checkWin(myTree.root.rowNum, myTree.root.colNum, not myTree.root.isRedTurn):#check winner
				currSet.hasWinner = True
				currSet.redWins = not myTree.root.isRedTurn
				trainingSet.append(currSet)
				currSet = gameSet(currSet.gameID + 1)
				break
			elif myTree.root.board.checkDraw():
				trainingSet.append(currSet)
				currSet = gameSet(currSet.gameID + 1)
				break
			else:
				boardMovePair.append(myTree.root.colNum)
				currSet.gameBoards.append(boardMovePair)
	#slice oldest games, pickle training set
	_pickle.dump(trainingSet[-config.fullTrainingSetSize:], open(modelName + "\\trainingData\\trainingSet", "wb"))
	
	print("Training samples updated.")

def createChallengerPair(modelName):
	#use sample board positions from library, use mcts to train current champion NN to create challenger NN and VNN
	trainingSet = _pickle.load(open(modelName + "\\trainingData\\trainingSet", "rb"))
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
	trainVal = tf.keras.models.load_model(modelName + "/the_value_champ")
	trainPol = tf.keras.models.load_model(modelName + "/the_policy_champ")

	trainPol.fit(boardList, np_utils.to_categorical(boardMoves, 7), epochs = 3, batch_size = 500)
	trainVal.fit(boardList, np_utils.to_categorical(boardResults, 3), epochs = 3, batch_size = 500)
	
	trainVal.save(modelName + "//the_value_challenger")
	trainPol.save(modelName + "//the_policy_challenger")
	print("Challenger ready and primed!")
	
def modelShowdown(modelName):
	#play config.showDownSize games using mcts, if challenger wins 55% challenger becomes new champion
	champVal = tf.keras.models.load_model(modelName + "/the_value_champ")
	challVal = tf.keras.models.load_model(modelName + "/the_value_challenger")
	champPol = tf.keras.models.load_model(modelName + "/the_policy_champ")
	challPol = tf.keras.models.load_model(modelName + "/the_policy_challenger")
	champFirst = False
	challWins = 0
	drawCount = 0
	champWins = 0
		
	print("Showdown!!!")
	for _ in tqdm(range(config.showDownSize)):
		champFirst = not champFirst# toggle first move
		isRedTurn = champFirst
		currBoardState = engine.board()
		while True:
			currPlayer = mc.monteTree(currBoardState, isRedTurn, champPol, champVal) if isRedTurn else mc.monteTree(currBoardState, isRedTurn, challPol, challVal)
			for _2 in range(config.trainingRecursionCount):
				currPlayer.nnSelectRec(currPlayer.root)
				#currPlayer.root.board.printBoard()
				#print(currPlayer.root.__str__(15))
			currBoardState, rowNum, colNum = currPlayer.makeMove()
			#currBoardState.printBoard()
			#print(currPlayer.root.__str__(0,2))
			#pdb.set_trace()
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
		Draws: {}'''.format(config.showDownSize, champWins, challWins, drawCount))
	if (challWins / champWins) > config.winRatio:
		print("New champion!!")
		#save previous champions to folder
				
		champArray = [f for f in os.listdir(modelName) if os.path.isfile(os.path.join(modelName, f)) and f[0:9] == "prevChamp"]
		
		champArray = sorted(champArray, key=lambda x: int(x[12:]))
		
		while len(champArray) > (config.champArrayLength * 2):
			os.remove(join(modelName, champArray[0]))
			del champArray[0]
		
		champNum = (str(int(champArray[len(champArray) - 1][12:]) + 1) if champArray else "1")
		
		champVal.save(modelName + "//prevChampVal" + champNum)
		champPol.save(modelName + "//prevChampPol" + champNum)
		
		challVal.save(modelName + "//the_value_champ")
		challPol.save(modelName + "//the_policy_champ")

def loadOrCreateModel():
	folderList = os.listdir(os.path.dirname(os.path.realpath(__file__)) + "\\models")
	if folderList:
		print("Models:")
		
		for i, folder in enumerate(folderList):
			print(str(i + 1) + ". " + folder)
			
		while True:
			try:
				modelInd = int(input(("Which model do you want to load? Select number or press '0' to create a new model. \n> ")))
				if modelInd > len(folderList) or modelInd < 0:
					print("Value out of range")
				else:
					break
			except ValueError:
				print("Which model do you want to load? Select number or press '0' to create a new model. \n> ")
		
		if modelInd:
			return os.path.dirname(os.path.realpath(__file__)) + "\\models\\" + folderList[modelInd - 1]
		else:
			print("Creating new model")
			while True:
				data = input("What do you want to call the model?\n> ")
				if data:
					if data not in folderList:
						createModel(data)
						return os.path.dirname(os.path.realpath(__file__)) + "\\models\\" + data
					else:
						print("A model with that name already exists.")
			
	else:
		print("No models exist")
		while True:
			data = input("What do you want to call the model?\n> ")
			if data:
				createModel(data)
				return os.path.dirname(os.path.realpath(__file__)) + "\\models\\" + data

def createModel(modelName):
	os.mkdir("models\\" + modelName)
	print("Creating value model")
	model = tf.keras.models.Sequential()
	hasSimpleAsLast = [False]
	addLayer(model, (6,7,3), hasSimpleAsLast)
	while True:
		userChoice = getResponseFromList("Would you like to add another layer?", ('y', 'n'))
		if userChoice == 'y':
			try:
				addLayer(model, model.layers[len(model.layers) - 1].output_shape[1:], hasSimpleAsLast)
			except AttributeError:
				addLayer(model, model.layers[len(model.layers) - 1].units, hasSimpleAsLast)
		else:
			break

	if not hasSimpleAsLast[0]:
		model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(3, activation = tf.nn.softmax))#3 category value model
	print("Here's the value model:")
	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	model.summary()
	model.save("models//" + modelName + "//the_value_champ")
	
	print("Creating policy model")
	model = tf.keras.models.Sequential()
	hasSimpleAsLast = [False]
	addLayer(model, (6,7,3), hasSimpleAsLast)
	while True:
		userChoice = getResponseFromList("Would you like to add another layer?", ('y', 'n'))
		if userChoice == 'y':
			try:
				addLayer(model, model.layers[len(model.layers) - 1].output_shape[1:], hasSimpleAsLast)
			except AttributeError:
				addLayer(model, model.layers[len(model.layers) - 1].units, hasSimpleAsLast)
		else:
			break

	if not hasSimpleAsLast[0]:
		model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(7, activation = tf.nn.softmax))#policy value model
	print("Here's the policy model:")
	model.summary()
	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	model.save("models//" + modelName + "//the_policy_champ")
	
	os.mkdir("models//" + modelName + "//trainingData")
	createTrainingSet(os.path.dirname(os.path.realpath(__file__)) + "\\models\\" + modelName, True)
	
def addLayer(model, shape, hasSimpleAsLast):
	userChoice = getResponseFromList("Would you like to make a convolutional, or simple layer?", ('c', 's'))
	if userChoice == 'c':
		hasSimpleAsLast[0] = False
		filterNum = getIntResponse("How many filters should this layer have?")
		filterDimVert = getIntResponse("What should the height of the filter be?")
		filterDimHori = getIntResponse("What should the width of the filter be?")
		paddingChoice = getResponseFromList("Should the padding type be same (zero padded), or valid (non-zero padded)?", ('s', 'v'))
		model.add(tf.keras.layers.Conv2D(filterNum, (filterDimVert, filterDimHori), input_shape = shape, padding=("same" if paddingChoice == 's' else "valid")))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.Activation(getResponseFromList("Which activation function should the layer use?", ('elu', 'relu', 'selu', 'sigmoid', 'softplus', 'softmax', 'softsign', 'tanh'))))
	elif userChoice == 's':
		activationDict = {
			"el" : tf.nn.elu,
			"rl" : tf.nn.relu,
			"sl" : tf.nn.selu,
			"sg" : tf.nn.sigmoid,
			"sm" : tf.nn.softmax,
			"sp" : tf.nn.softplus,
			"ss" : tf.nn.softsign,
			"tn" : tf.nn.tanh
		}
		nuerons = getIntResponse("How many neurons should this layer have?")
		layerActivation = activationDict[getResponseFromList("What activation function should the layer use? elu, relu, selu, sigmoid, softplus, softmax, softsign, or tanh?", ('el', 'rl', 'sl', 'sg', 'sm', 'sp', 'ss', 'tn'))]
		if not hasSimpleAsLast[0]:
			model.add(tf.keras.layers.Flatten(input_shape=shape))
		model.add(tf.keras.layers.Dense(nuerons, activation = layerActivation))
		hasSimpleAsLast[0] = True
		
def getResponseFromList(question, answers):
	while True:
		data = input(question + " " + str(answers) + "\n> ")
		if data:
			if data in answers:
				return data
			else:
				print("Please choose an option from the list")

def getIntResponse(question):
	while True:
		try:
			i = int(input(question + "\n> "))
			if i <= 0:
				print("Value out of range")
			else:
				return i
		except ValueError:
			print("Please enter an integer.")
			
modelPath = loadOrCreateModel()

while True:
	createTrainingSet(modelPath)
	createChallengerPair(modelPath)
	modelShowdown(modelPath)